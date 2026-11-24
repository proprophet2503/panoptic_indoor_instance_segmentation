"""Assemble train_instance_seg_kaggle.ipynb from src/*.py (embedded via %%writefile) + config/run cells.

Usage: python build_notebook.py   (re-run after editing anything in src/)
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SRC_FILES = ["common.py", "common_torch.py", "train_yolo.py", "train_maskrcnn.py", "train_mask2former.py",
             "train_rfdetr.py", "worker.py", "prefetch.py", "orchestrator.py"]

INTRO = """# Instance Segmentation: CNN vs Transformer × Training-Set Size

**Dataset:** `panoptic_indoor_dataset_segmentation` (922 images, 74 classes, YOLO polygon labels)
**Hardware:** Kaggle **GPU T4 ×2** (2 × 16 GB). Two experiments run in parallel, one per GPU.

| Role | What |
|---|---|
| **Independent variable** | training fraction of the train pool, `TRAIN_FRACTIONS` (placeholder, default `[0.25, 0.5, 1.0]`) |
| **Constants (all 4 models)** | image size, optimizer-step budget, effective batch, augmentation (h-flip + scale jitter), COCO-pretrained init, fp16 AMP, seed, warmup, eval protocol |
| **Model-specific constants** | optimizer / LR / schedule from each model's paper or framework recipe, frozen across all splits |
| **Dependent variables** | COCO mask metrics on the **fixed test set**: AP, AP50, AP75, APs/m/l, AR, AP(thing), AP(stuff), per-class AP, FPS |

| Model | Family | Paradigm |
|---|---|---|
| `yolo26l_seg` | CNN | one-stage, NMS-free (Ultralytics) |
| `maskrcnn_r50_fpn` | CNN | two-stage (torchvision) |
| `mask2former_swin_t` | Transformer | query-based, Swin-T backbone (HF transformers) |
| `rfdetr_seg_l` | Transformer | real-time DETR, DINOv2 backbone (rfdetr) |

**Splits.** Fixed once with `SEED`: val = 10 % (best-checkpoint selection), test = 10 % (final metrics), pool = remaining 80 %.
`train(f)` = first `f` of the shuffled pool, so the subsets are nested (25 % ⊂ 50 % ⊂ 100 %). Same val/test for every run.

**Order.** Model-major: model 1 × all splits, then model 2, …, two jobs at a time (GPU0, GPU1).

### How to run / resume (Save Version mode)
1. Settings: **Accelerator = GPU T4 ×2**, **Internet = On**. Add the dataset as input.
2. W&B: *Add-ons → Secrets* → label `WANDB_API_KEY` (recommended), or paste the key in the W&B cell.
3. **Save Version → Save & Run All (Commit)**, and keep *Always save output* on.
4. The run pauses itself before `SESSION_BUDGET_HOURS` (every worker stops at an epoch boundary, so a checkpoint is on disk), and the version's output is saved.
5. **To continue**, add the previous version's output as input (*Add Input → Your Work → this notebook*, or turn the output into a dataset), make sure it's the **latest** version, then Save Version again.
   Finished experiments are skipped. Interrupted ones resume from their last epoch. Remaining ones start.

### Outputs (`/kaggle/working/outputs`)
```
split_{pct}/{model}/best.*  last.*  metrics_test.json  per_class_ap_test.csv  status.json
results.csv        # dependent variables, one row per (model, split)
wandb_runs.csv     # one W&B run per experiment: id + URL + status
splits.json        # the fixed val / test / pool file lists
logs/{model}__split_{pct}.log
```
`/kaggle/working/runs/` holds in-progress training state, it is deleted per experiment when that experiment finishes.
"""

CONFIG = r'''import time
SESSION_START_TS = time.time()

# ===================== INDEPENDENT VARIABLE (placeholder, edit me) =====================
TRAIN_FRACTIONS = [0.25, 0.5, 1.0]          # fractions of the train pool (80 % of the dataset)

# ===================== Session budget =====================
SESSION_BUDGET_HOURS = 11.0  # Kaggle hard limit is 12 h, set lower if your remaining weekly GPU quota is smaller
PAUSE_MARGIN_MIN = 40        # workers are told to pause this many minutes before the budget ends

# ===================== Paths =====================
DATASET_DIR = "/kaggle/input/datasets/jeremymattathiasmboe/panoptic-indoor-dataset-instance-segmentation"
WORKING = "/kaggle/working"

# ===================== Constants shared by ALL models (do not change between sessions) =====================
SHARED = dict(
    SEED=42,
    VAL_FRAC=0.10,                          # fixed val set (best-checkpoint selection)
    TEST_FRAC=0.10,                         # fixed test set (final metrics)
    IMG_SIZE=672,                           # long side, divisible by 32 (YOLO stride) and 24 (RF-DETR patch*windows)
    TRAIN_SCALES=list(range(552, 793, 24)), # scale jitter: long side in [552, 792] = IMG_SIZE x [0.82, 1.18]
    SCALE_JITTER=0.18,                      # same range expressed as Ultralytics `scale`
    HFLIP_P=0.5,                            # the only other augmentation
    EFFECTIVE_BATCH=8,                      # images per optimizer step (micro batch x grad accumulation)
    TOTAL_STEPS=4000,                       # optimizer-step budget per run -> epochs = ceil(TOTAL_STEPS / steps_per_epoch)
    NUM_EVALS=20,                           # ~validations per run (val every round(epochs/NUM_EVALS) epochs)
    WARMUP_STEPS=200,                       # linear LR warmup, in optimizer steps
    AMP=True,                               # fp16 mixed precision (T4 has no bf16)
    MAX_DET=100,                            # detections per image (COCO standard)
    SCORE_THRESH=0.001,                     # keep low-score detections for AP
)

# ===================== Model-specific recipes (constant across splits) =====================
# micro_batch only changes memory use (gradient accumulation keeps EFFECTIVE_BATCH), lower it on OOM.
MODELS = {
    "yolo26l_seg": dict(impl="yolo", family="CNN", weights="yolo26l-seg.pt", micro_batch=8,
                        # = Ultralytics optimizer='auto' choice for <10k iterations and nc=74, pinned explicitly
                        optimizer="AdamW", lr=0.000128, lrf=0.01, momentum=0.9, weight_decay=0.0005),
    "maskrcnn_r50_fpn": dict(impl="maskrcnn", family="CNN", micro_batch=8,
                             # Detectron/torchvision recipe: SGD 0.02 @ batch 16 -> linear scaling to batch 8
                             optimizer="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001,
                             lr_milestones=[0.667, 0.889], lr_gamma=0.1),
    "mask2former_swin_t": dict(impl="mask2former", family="Transformer",
                               checkpoint="facebook/mask2former-swin-tiny-coco-instance", micro_batch=4,
                               # paper recipe: AdamW 1e-4, wd 0.05, backbone x0.1, poly 0.9, clip 0.01
                               optimizer="AdamW", lr=0.0001, weight_decay=0.05, backbone_lr_mult=0.1,
                               poly_power=0.9, clip_norm=0.01),
    "rfdetr_seg_l": {"impl": "rfdetr", "family": "Transformer", "class": "RFDETRSegLarge", "micro_batch": 4,
                     # rfdetr defaults: AdamW 1e-4 (encoder 1.5e-4), wd 1e-4, EMA, constant LR
                     "lr": 0.0001, "lr_encoder": 0.00015, "weight_decay": 0.0001},
}
MODEL_ORDER = ["yolo26l_seg", "maskrcnn_r50_fpn", "mask2former_swin_t", "rfdetr_seg_l"]
NUM_WORKERS = 2            # dataloader workers per job (Kaggle has few CPU cores, 2 jobs run at once)
MAX_ATTEMPTS = 3           # a crashing experiment is retried in later sessions up to this many times
'''

WANDB = r'''# ===================== Weights & Biases =====================
# Recommended: Kaggle "Add-ons -> Secrets", label WANDB_API_KEY (the key is not stored in the notebook).
# Fallback: paste the key here. NOTE: a pasted key is saved inside this notebook version.
WANDB_API_KEY = ""                                 # <- optional manual paste
WANDB_PROJECT = "panoptic-indoor-instance-seg"
WANDB_ENTITY = ""                                  # team or username, empty = your default entity

import os
if not WANDB_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        WANDB_API_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:
        print("Kaggle secret WANDB_API_KEY not available:", exc)
WANDB_ENABLED = bool(WANDB_API_KEY)
if WANDB_ENABLED:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    import wandb
    wandb.login(key=WANDB_API_KEY, relogin=True)
    print("W&B enabled -> one run per experiment, links collected in outputs/wandb_runs.csv")
else:
    print("W&B DISABLED (no key). Training still runs, metrics are saved to outputs/.")
'''

INSTALL = r'''import torch, torchvision
# keep Kaggle's CUDA build of torch, versions below are the ones this pipeline was tested with
open("/tmp/constraints.txt", "w").write(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n")
!pip install -q -c /tmp/constraints.txt "ultralytics==8.4.31" "rfdetr[train]==1.12.1" "transformers==5.23.0" \
    "pytorch-lightning==2.6.6" "supervision==0.31.7" pycocotools wandb
!python -c "import ultralytics, rfdetr, transformers, pycocotools, torch"
!python -c "print('ultralytics', __import__('ultralytics').__version__, '| transformers', __import__('transformers').__version__, '| torch', __import__('torch').__version__, '| GPUs', __import__('torch').cuda.device_count())"
'''

WRITE_CONFIG = r'''import json, glob, sys
from pathlib import Path
import torch

SRC_DIR = f"{WORKING}/src"
if not Path(DATASET_DIR, "data.yaml").exists():  # fall back to wherever Kaggle mounted the dataset
    hits = [Path(p).parent for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True)
            if Path(p).parent.joinpath("images").is_dir() and Path(p).parent.joinpath("labels").is_dir()]
    assert hits, f"dataset not found at {DATASET_DIR} or anywhere under /kaggle/input"
    DATASET_DIR = str(hits[0])
print("dataset:", DATASET_DIR)

n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "No GPU visible: set Accelerator = GPU T4 x2"
cfg = dict(
    **SHARED, TRAIN_FRACTIONS=TRAIN_FRACTIONS, MODELS=MODELS, MODEL_ORDER=MODEL_ORDER,
    SHARED_KEYS=list(SHARED) + ["TRAIN_FRACTIONS"],
    NUM_WORKERS=NUM_WORKERS, MAX_ATTEMPTS=MAX_ATTEMPTS, LOG_EVERY_STEPS=20, CLEAN_RUN_DIRS=True,
    DATASET_DIR=DATASET_DIR, OUTPUT_DIR=f"{WORKING}/outputs", RUNS_DIR=f"{WORKING}/runs",
    DATA_DIR="/tmp/exp_data", WORK_CWD="/tmp/exp_cwd", WANDB_DIR="/tmp/wandb", STOP_FILE="/tmp/exp_STOP",
    INPUT_ROOT="/kaggle/input", CONFIG_PATH=f"{WORKING}/exp_config.json",
    WANDB_ENABLED=WANDB_ENABLED, WANDB_PROJECT=WANDB_PROJECT, WANDB_ENTITY=WANDB_ENTITY,
    GPUS=list(range(min(n_gpu, 2))), SESSION_START_TS=SESSION_START_TS,
    SESSION_BUDGET_HOURS=SESSION_BUDGET_HOURS, PAUSE_MARGIN_MIN=PAUSE_MARGIN_MIN,
)
for d in ("DATA_DIR", "WORK_CWD", "WANDB_DIR"):
    Path(cfg[d]).mkdir(parents=True, exist_ok=True)
Path(cfg["CONFIG_PATH"]).write_text(json.dumps(cfg, indent=1))
os.environ["EXP_CONFIG"] = cfg["CONFIG_PATH"]
sys.path.insert(0, SRC_DIR)
print(f"GPUs used: {cfg['GPUS']} | runs: {len(MODEL_ORDER)} models x {len(TRAIN_FRACTIONS)} splits")
'''

PREPARE = r'''import common as C, orchestrator as O

O.restore_previous_outputs(cfg)      # copies outputs/ + runs/ from an attached previous version (if any)
prev = C.read_json(Path(cfg["OUTPUT_DIR"]) / "PIPELINE_MARKER.json")
if prev:  # guard: constants must not change between sessions, otherwise runs are not comparable
    changed = {k: (prev["shared_config"].get(k), cfg[k]) for k in cfg["SHARED_KEYS"]
               if prev["shared_config"].get(k) != cfg[k]}
    assert not changed, f"Shared constants changed since the first session: {changed}"
info = C.prepare_data(cfg)
O.write_marker(cfg)
for f in TRAIN_FRACTIONS:
    sched = C.schedule(cfg, info["sizes"][f"train_{C.pct(f)}"], cfg["EFFECTIVE_BATCH"])
    print(f"split {C.pct(f):>3}% -> {sched['n_train']} train imgs, {sched['epochs']} epochs, "
          f"{sched['total_steps']} steps, val every {sched['val_every_epochs']} epochs")
'''

PREFETCH = r'''import subprocess
# download all pretrained weights once, before two workers start in parallel
subprocess.run([sys.executable, f"{SRC_DIR}/prefetch.py"], check=True, cwd=cfg["WORK_CWD"],
               env={**os.environ, "EXP_CONFIG": cfg["CONFIG_PATH"],
                    "PYTHONPATH": os.pathsep.join(filter(None, [SRC_DIR, os.environ.get("PYTHONPATH")]))})
'''

RUN = r'''# Runs every unfinished (model, split) pair, 2 at a time. Progress is printed every 10 min.
# Full per-experiment logs: outputs/logs/. Safe to re-run: finished experiments are skipped.
rows = O.run_all(cfg)
'''

REPORT = r'''import pandas as pd
import matplotlib.pyplot as plt

O.write_marker(cfg)
out = Path(cfg["OUTPUT_DIR"])
display(pd.read_csv(out / "wandb_runs.csv"))
if (out / "results.csv").exists():
    res = pd.read_csv(out / "results.csv")
    display(res)
    for metric in ["segm_AP", "segm_AP50", "segm_AP_thing", "segm_AP_stuff"]:
        display(res.pivot(index="model", columns="train_fraction", values=metric).round(4).rename_axis(metric))
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in res.groupby("model"):
        g = g.sort_values("train_fraction")
        ax.plot(g["train_fraction"], g["segm_AP"], marker="o", label=f"{model} ({g['family'].iloc[0]})")
    ax.set(xlabel="training fraction", ylabel="test mask AP (COCO)", title="Mask AP vs training-set size")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "ap_vs_train_fraction.png", dpi=150)
    plt.show()
'''


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(INTRO),
             md("## 1. Configuration (placeholders + constants)"), code(CONFIG),
             md("## 2. Install dependencies"), code(INSTALL),
             md("## 3. Weights & Biases API key"), code(WANDB),
             md("## 4. Pipeline source files\nEach experiment runs in its own process (`worker.py`) pinned to one GPU."),
             code("os.makedirs('/kaggle/working/src', exist_ok=True)")]
    cells += [code(f"%%writefile /kaggle/working/src/{n}\n{(HERE / 'src' / n).read_text()}") for n in SRC_FILES]
    cells += [md("## 5. Config file, restore previous session, build splits"), code(WRITE_CONFIG), code(PREPARE),
              md("## 6. Pre-download pretrained weights"), code(PREFETCH),
              md("## 7. Train + evaluate all experiments (2 GPUs in parallel)"), code(RUN),
              md("## 8. Results (dependent variables) and W&B links"), code(REPORT)]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}})
    path = HERE / "train_instance_seg_kaggle.ipynb"
    nbf.write(nb, path)
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
