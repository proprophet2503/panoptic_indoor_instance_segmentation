"""Shared utilities: config, fixed splits, COCO conversion, unified COCO evaluation, run state, W&B."""
import contextlib
import csv
import io
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXIT_PAUSED = 75  # worker stopped at an epoch boundary because the session deadline is near

# COCO-panoptic "stuff" categories present in this dataset (+ the stray "wall" class). Everything else is "thing".
STUFF_CLASSES = {
    "blanket", "cabinet-merged", "cardboard", "ceiling-merged", "counter", "curtain", "door-stuff",
    "floor-other-merged", "floor-wood", "flower", "food-other-merged", "light", "mirror-stuff", "paper-merged",
    "pillow", "rug-merged", "shelf", "stairs", "table-merged", "towel", "tree-merged", "wall", "wall-brick",
    "wall-other-merged", "wall-stone", "wall-tile", "wall-wood", "window-blind", "window-other",
}
METRIC_NAMES = ["segm_AP", "segm_AP50", "segm_AP75", "segm_APs", "segm_APm", "segm_APl",
                "segm_AR1", "segm_AR10", "segm_AR100", "segm_ARs", "segm_ARm", "segm_ARl"]


# ----------------------------------------------------------------------------- config / io
def load_config(path=None):
    return json.loads(Path(path or os.environ["EXP_CONFIG"]).read_text())


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def pct(frac):
    return int(round(float(frac) * 100))


def exp_name(model_key, frac):
    return f"{model_key}__split_{pct(frac)}"


def out_dir(cfg, model_key, frac):
    return Path(cfg["OUTPUT_DIR"]) / f"split_{pct(frac)}" / model_key


def run_dir(cfg, model_key, frac):
    return Path(cfg["RUNS_DIR"]) / model_key / f"split_{pct(frac)}"


def status_path(cfg, model_key, frac):
    return out_dir(cfg, model_key, frac) / "status.json"


def should_stop(cfg):
    return Path(cfg["STOP_FILE"]).exists()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- schedule shared by all models
def schedule(cfg, n_train, micro_batch):
    """Fixed optimizer-step budget -> epochs for this split. Same rule for every model.

    steps_per_epoch counts optimizer steps at EFFECTIVE_BATCH (micro_batch * accumulation).
    Validation runs every `val_every_epochs` so every run gets ~NUM_EVALS validations regardless of split size.
    """
    eff = cfg["EFFECTIVE_BATCH"]
    assert eff % micro_batch == 0, f"EFFECTIVE_BATCH {eff} must be a multiple of micro batch {micro_batch}"
    steps_per_epoch = math.ceil(n_train / eff)
    epochs = math.ceil(cfg["TOTAL_STEPS"] / steps_per_epoch)
    return {
        "n_train": n_train,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_steps": epochs * steps_per_epoch,
        "val_every_epochs": max(1, round(epochs / cfg["NUM_EVALS"])),
        "micro_batch": micro_batch,
        "accumulate": eff // micro_batch,
        "warmup_epochs": cfg["WARMUP_STEPS"] / steps_per_epoch,
    }


# ----------------------------------------------------------------------------- data
def class_names(cfg):
    names = yaml.safe_load((Path(cfg["DATASET_DIR"]) / "data.yaml").read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def make_splits(cfg):
    """val/test fixed once (seeded) and persisted, train(frac) = first frac of the shuffled pool (nested)."""
    path = Path(cfg["OUTPUT_DIR"]) / "splits.json"
    if path.exists():
        return read_json(path)
    files = sorted(p.name for p in (Path(cfg["DATASET_DIR"]) / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    rng = random.Random(cfg["SEED"])
    rng.shuffle(files)
    n_val = round(len(files) * cfg["VAL_FRAC"])
    n_test = round(len(files) * cfg["TEST_FRAC"])
    splits = {
        "seed": cfg["SEED"],
        "val": sorted(files[:n_val]),
        "test": sorted(files[n_val:n_val + n_test]),
        "pool": files[n_val + n_test:],  # keep shuffled order -> nested train subsets
    }
    atomic_write_json(path, splits)
    return splits


def train_files(splits, frac):
    pool = splits["pool"]
    return sorted(pool[:max(1, round(len(pool) * float(frac)))])


def yolo_to_coco(cfg, files, names):
    img_dir = Path(cfg["DATASET_DIR"]) / "images"
    lbl_dir = Path(cfg["DATASET_DIR"]) / "labels"
    images, anns = [], []
    for img_id, fn in enumerate(files, 1):
        with Image.open(img_dir / fn) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": fn, "width": w, "height": h})
        lbl = lbl_dir / (Path(fn).stem + ".txt")
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            vals = line.split()
            if len(vals) < 7 or (len(vals) - 1) % 2:
                continue  # fewer than 3 points or malformed
            xy = np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2) * [w, h]
            xy[:, 0] = xy[:, 0].clip(0, w)
            xy[:, 1] = xy[:, 1].clip(0, h)
            poly = xy.flatten().tolist()
            area = float(mask_utils.area(mask_utils.merge(mask_utils.frPyObjects([poly], h, w))))
            if area < 1:
                continue
            x0, y0 = xy.min(0)
            x1, y1 = xy.max(0)
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": int(vals[0]) + 1,
                         "segmentation": [poly], "area": area,
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "iscrowd": 0})
    cats = [{"id": i + 1, "name": n, "supercategory": "stuff" if n in STUFF_CLASSES else "thing"}
            for i, n in enumerate(names)]
    return {"images": images, "annotations": anns, "categories": cats}


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src, dst)


def prepare_data(cfg):
    """Build COCO json + YOLO dirs + RF-DETR dirs for every split. Idempotent, cheap to redo each session."""
    names = class_names(cfg)
    splits = make_splits(cfg)
    src_img = Path(cfg["DATASET_DIR"]) / "images"
    src_lbl = Path(cfg["DATASET_DIR"]) / "labels"
    data = Path(cfg["DATA_DIR"])
    sets = {"val": splits["val"], "test": splits["test"]}
    sets.update({f"train_{pct(f)}": train_files(splits, f) for f in cfg["TRAIN_FRACTIONS"]})

    for name, files in sets.items():
        cj = data / "coco" / f"{name}.json"
        if not cj.exists():
            atomic_write_json(cj, yolo_to_coco(cfg, files, names))
        ydir = data / "yolo" / name
        for fn in files:  # ultralytics maps .../images/x.jpg -> .../labels/x.txt by string replacement
            _link(src_img / fn, ydir / "images" / fn)
            lbl, dst = src_lbl / (Path(fn).stem + ".txt"), ydir / "labels" / (Path(fn).stem + ".txt")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if lbl.exists():
                    shutil.copy(lbl, dst)
                else:
                    dst.touch()

    for frac in cfg["TRAIN_FRACTIONS"]:
        p = pct(frac)
        (data / "yolo" / f"split_{p}.yaml").write_text(yaml.safe_dump({
            "path": str(data / "yolo"), "train": f"train_{p}/images", "val": "val/images",
            "names": dict(enumerate(names))}, sort_keys=False))
        for sub, name in (("train", f"train_{p}"), ("valid", "val"), ("test", "test")):
            rdir = data / "rfdetr" / f"split_{p}" / sub
            for fn in sets[name]:
                _link(src_img / fn, rdir / fn)
            shutil.copy(data / "coco" / f"{name}.json", rdir / "_annotations.coco.json")

    sizes = {k: len(v) for k, v in sets.items()}
    log(f"data ready: {sizes}")
    return {"names": names, "splits": splits, "sizes": sizes}


def coco_json(cfg, name):
    return Path(cfg["DATA_DIR"]) / "coco" / f"{name}.json"


def image_dir(cfg):
    return Path(cfg["DATASET_DIR"]) / "images"


def n_images(cfg, name):
    return len(read_json(coco_json(cfg, name))["images"])


# ----------------------------------------------------------------------------- unified evaluation
def coco_evaluate(gt_json, predict_fn, img_dir, max_det=100, fps_warmup=5, tag="eval"):
    """predict_fn(path) -> list of {"category_id", "score", "mask": bool HxW at original resolution}.

    Same pycocotools segm protocol for every model -> comparable dependent variables.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
    results, times = [], []
    img_ids = sorted(gt.getImgIds())
    for k, img_id in enumerate(img_ids):
        info = gt.imgs[img_id]
        t0 = time.perf_counter()
        preds = predict_fn(Path(img_dir) / info["file_name"])
        if k >= fps_warmup:
            times.append(time.perf_counter() - t0)
        for p in sorted(preds, key=lambda d: -d["score"])[:max_det]:
            m = np.asarray(p["mask"], dtype=np.uint8)
            assert m.shape == (info["height"], info["width"]), f"mask {m.shape} vs image {(info['height'], info['width'])}"
            if m.sum() == 0:
                continue
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            results.append({"image_id": img_id, "category_id": int(p["category_id"]),
                            "score": float(p["score"]), "segmentation": rle})
    fps = (len(times) / sum(times)) if times else float("nan")
    if not results:
        return {**{n: 0.0 for n in METRIC_NAMES}, "fps": fps, "num_images": len(img_ids)}, []
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(results), "segm")
        ev.params.maxDets = [1, 10, max_det]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    metrics = {n: float(v) for n, v in zip(METRIC_NAMES, ev.stats)}

    per_class = []
    prec = ev.eval["precision"]  # [T, R, K, A, M], A=0 -> all areas, M=-1 -> max_det
    for k, cid in enumerate(ev.params.catIds):
        p_all, p50 = prec[:, :, k, 0, -1], prec[0, :, k, 0, -1]
        cat = gt.cats[cid]
        per_class.append({"category_id": cid, "name": cat["name"], "type": cat["supercategory"],
                          "num_gt": len(gt.getAnnIds(catIds=[cid])),
                          "AP": float(p_all[p_all > -1].mean()) if (p_all > -1).any() else float("nan"),
                          "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")})
    for kind in ("thing", "stuff"):
        aps = [c["AP"] for c in per_class if c["type"] == kind and c["num_gt"] > 0 and not math.isnan(c["AP"])]
        metrics[f"segm_AP_{kind}"] = float(np.mean(aps)) if aps else float("nan")
    metrics["classes_in_eval"] = sum(c["num_gt"] > 0 for c in per_class)
    metrics["fps"] = fps
    metrics["num_images"] = len(img_ids)
    log(f"{tag}: segm_AP={metrics['segm_AP']:.4f} AP50={metrics['segm_AP50']:.4f} fps={fps:.1f}")
    return metrics, per_class


# ----------------------------------------------------------------------------- W&B
def wandb_init(cfg, model_key, frac, status, sched):
    """One W&B run per experiment, stable id -> a resumed session appends to the same run."""
    if not cfg.get("WANDB_ENABLED"):
        return None
    import wandb
    run = wandb.init(
        project=cfg["WANDB_PROJECT"], entity=cfg.get("WANDB_ENTITY") or None,
        id=status["wandb_id"], resume="allow", name=exp_name(model_key, frac),
        group=f"split_{pct(frac)}", job_type=model_key,  # files go to $WANDB_DIR (not saved as output)
        config={"model": model_key, "train_fraction": frac, "schedule": sched,
                "shared": {k: cfg[k] for k in cfg["SHARED_KEYS"]}, "model_hp": cfg["MODELS"][model_key]},
    )
    status["wandb_url"] = run.url
    atomic_write_json(status_path(cfg, model_key, frac), status)
    return run


def wandb_log(data, step=None):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(data, step=step)


def wandb_ensure(cfg, model_key, frac, status):
    """Some trainers (PTL) finish the run on exit, re-attach before logging test metrics."""
    if not cfg.get("WANDB_ENABLED"):
        return
    import wandb
    if wandb.run is None:
        wandb.init(project=cfg["WANDB_PROJECT"], entity=cfg.get("WANDB_ENTITY") or None,
                   id=status["wandb_id"], resume="allow")


def pause_and_exit():
    """Called at an epoch boundary (last checkpoint complete on disk) when the session deadline is near."""
    log("STOP file found -> pausing at epoch boundary")
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish(exit_code=EXIT_PAUSED)
    except Exception as exc:  # never block the pause on logging
        log(f"wandb finish failed: {exc}")
    sys.stdout.flush()
    os._exit(EXIT_PAUSED)


# ----------------------------------------------------------------------------- finalize
def finalize(cfg, model_key, frac, status, best_src, last_src, metrics, per_class, extra=None):
    od = out_dir(cfg, model_key, frac)
    od.mkdir(parents=True, exist_ok=True)
    for src, stem in ((best_src, "best"), (last_src, "last")):
        if src and Path(src).exists():
            shutil.copy(src, od / f"{stem}{Path(src).suffix}")
    atomic_write_json(od / "metrics_test.json", {**metrics, **(extra or {})})
    with open(od / "per_class_ap_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category_id", "name", "type", "num_gt", "AP", "AP50"])
        w.writeheader()
        w.writerows(per_class)

    wandb_ensure(cfg, model_key, frac, status)
    if cfg.get("WANDB_ENABLED"):
        import wandb
        if wandb.run is not None:
            wandb.run.summary.update({f"test/{k}": v for k, v in metrics.items()})
            if per_class:
                wandb.log({"test/per_class_AP": wandb.Table(columns=list(per_class[0].keys()),
                                                             data=[list(r.values()) for r in per_class])})
            status["wandb_url"] = wandb.run.url
            wandb.finish()

    status.update(status="done", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  test_segm_AP=metrics["segm_AP"], test_segm_AP50=metrics["segm_AP50"])
    atomic_write_json(status_path(cfg, model_key, frac), status)
    if cfg.get("CLEAN_RUN_DIRS", True):
        shutil.rmtree(run_dir(cfg, model_key, frac), ignore_errors=True)
    log(f"DONE {exp_name(model_key, frac)} -> {od}")
