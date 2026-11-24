"""RF-DETR-Seg (Roboflow rfdetr, COCO-pretrained, DINOv2 backbone) -- transformer, real-time."""
from pathlib import Path

import numpy as np
import torch

import common as C


def _install_trainer_hooks(cfg):
    """rfdetr builds its PTL Trainer internally, inject our epoch-boundary pause callback via extra_callbacks."""
    import rfdetr.training.trainer as T
    from pytorch_lightning import Callback

    class PauseAtEpochStart(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if C.should_stop(cfg):
                C.pause_and_exit()  # last.ckpt of the previous epoch is complete

    original = T.build_trainer
    if getattr(original, "_patched", False):
        return

    def build_trainer(*args, extra_callbacks=(), **kwargs):
        return original(*args, extra_callbacks=(*extra_callbacks, PauseAtEpochStart()), **kwargs)

    build_trainer._patched = True
    T.build_trainer = build_trainer
    import rfdetr.detr as D
    import rfdetr.training as TR
    for mod in (D, TR):
        if hasattr(mod, "build_trainer"):
            mod.build_trainer = build_trainer


def run(cfg, model_key, frac, status):
    import rfdetr

    hp = cfg["MODELS"][model_key]
    model_cls = getattr(rfdetr, hp["class"])
    p = C.pct(frac)
    rd = C.run_dir(cfg, model_key, frac)
    rd.mkdir(parents=True, exist_ok=True)
    last_ckpt, done_flag = rd / "last.ckpt", rd / "train_complete.json"
    sched = C.schedule(cfg, C.n_images(cfg, f"train_{p}"), hp["micro_batch"])
    C.wandb_init(cfg, model_key, frac, status, sched)  # PTL WandbLogger re-uses this active run
    C.log(f"{C.exp_name(model_key, frac)} schedule {sched}")

    if not done_flag.exists():
        _install_trainer_hooks(cfg)
        resume = str(last_ckpt) if last_ckpt.exists() else None
        if resume:
            C.log(f"resuming from {last_ckpt}")
        model = model_cls()
        model.train(
            dataset_dir=str(Path(cfg["DATA_DIR"]) / "rfdetr" / f"split_{p}"), dataset_file="roboflow",
            output_dir=str(rd), resume=resume,
            epochs=sched["epochs"], batch_size=hp["micro_batch"], grad_accum_steps=sched["accumulate"],
            lr=hp["lr"], lr_encoder=hp["lr_encoder"], weight_decay=hp["weight_decay"],
            warmup_epochs=sched["warmup_epochs"], lr_scheduler="step",
            lr_scheduler_kwargs={"lr_drop": sched["epochs"]},  # recipe default = constant LR (+EMA), no drop
            resolution=cfg["IMG_SIZE"],
            # shared augmentation spec: hflip (rfdetr default) + per-batch multi-scale, no random crop
            multi_scale=True, expanded_scales=True, scale_jitter=False, aug_config=None,
            eval_interval=sched["val_every_epochs"], checkpoint_interval=10**6, eval_max_dets=cfg["MAX_DET"],
            seed=cfg["SEED"], num_workers=cfg["NUM_WORKERS"], amp_dtype="fp16" if cfg["AMP"] else None,
            wandb=bool(cfg.get("WANDB_ENABLED")), project=cfg["WANDB_PROJECT"], run=C.exp_name(model_key, frac),
            tensorboard=False, progress_bar=None, early_stopping=False, run_test=False)
        C.atomic_write_json(done_flag, {"epochs": sched["epochs"]})

    best_pth = rd / "checkpoint_best_total.pth"
    model = model_cls.from_checkpoint(str(best_pth))
    names = C.class_names(cfg)
    cls_names = model.class_names
    cls_names = [cls_names[k] for k in sorted(cls_names)] if isinstance(cls_names, dict) else list(cls_names)
    name_to_cat = {n: i + 1 for i, n in enumerate(names)}
    idx_to_cat = {i: name_to_cat[n] for i, n in enumerate(cls_names) if n in name_to_cat}
    assert len(idx_to_cat) == len(names), f"class mapping mismatch: {cls_names[:5]}..."

    def predict(path):
        from PIL import Image
        det = model.predict(Image.open(path).convert("RGB"), threshold=cfg["SCORE_THRESH"])
        if det.mask is None or len(det) == 0:
            return []
        return [{"category_id": idx_to_cat[int(c)], "score": float(s), "mask": np.asarray(m, dtype=bool)}
                for c, s, m in zip(det.class_id, det.confidence, det.mask) if int(c) in idx_to_cat]

    with torch.no_grad():
        metrics, per_class = C.coco_evaluate(C.coco_json(cfg, "test"), predict, C.image_dir(cfg),
                                             max_det=cfg["MAX_DET"], tag="TEST")
    last_w = rd / "last_weights.pth"
    if last_ckpt.exists():  # weights only, full PTL state (optimizer, loops) is not needed after training
        sd = torch.load(last_ckpt, map_location="cpu", weights_only=False).get("state_dict", {})
        torch.save({"state_dict": sd}, last_w)
    C.finalize(cfg, model_key, frac, status, best_pth, last_w, metrics, per_class, extra={"schedule": sched})
