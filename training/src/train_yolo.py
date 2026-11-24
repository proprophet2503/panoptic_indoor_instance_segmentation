"""YOLO26-L-seg (Ultralytics, COCO-pretrained) -- CNN, one-stage, NMS-free."""
from pathlib import Path

import torch

import common as C


def _training_finished(last_pt):
    """Ultralytics strips the optimizer and sets epoch=-1 in last.pt once training completes."""
    return last_pt.exists() and torch.load(last_pt, map_location="cpu", weights_only=False).get("epoch", 0) == -1


def _add_callbacks(model, cfg, sched):
    validated = {"epoch": False}

    def on_train_epoch_start(trainer):
        if C.should_stop(cfg):
            C.pause_and_exit()  # last.pt of the previous epoch is complete

    def on_train_epoch_end(trainer):
        e = trainer.epoch + 1
        do_val = e % sched["val_every_epochs"] == 0 or e >= trainer.epochs
        trainer.args.val = do_val
        validated["epoch"] = do_val
        if not do_val:
            trainer.fitness = None  # otherwise save_model() would overwrite best.pt with unvalidated weights

    def on_fit_epoch_end(trainer):
        e = trainer.epoch + 1
        logs = {f"train/{k.split('/')[-1]}": float(v)
                for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()}
        logs.update({"train/lr": trainer.lr.get("lr/pg0", 0.0), "train/epoch": e})
        if validated["epoch"]:
            logs.update({f"val_ultralytics/{k.split('/', 1)[-1]}": float(v) for k, v in trainer.metrics.items()})
        C.wandb_log(logs, step=e * sched["steps_per_epoch"])

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def run(cfg, model_key, frac, status):
    from ultralytics import YOLO, settings
    if settings.get("wandb"):  # normally already disabled once by prefetch.py (avoids 2 workers writing at once)
        settings.update({"wandb": False})  # we log ourselves -> same W&B layout as the other models

    hp = cfg["MODELS"][model_key]
    p = C.pct(frac)
    rd = C.run_dir(cfg, model_key, frac)
    last_pt, best_pt = rd / "weights" / "last.pt", rd / "weights" / "best.pt"
    sched = C.schedule(cfg, C.n_images(cfg, f"train_{p}"), hp["micro_batch"])
    assert sched["accumulate"] == 1, "YOLO micro batch must equal EFFECTIVE_BATCH (nbs handles the rest)"
    C.wandb_init(cfg, model_key, frac, status, sched)
    C.log(f"{C.exp_name(model_key, frac)} schedule {sched}")

    if not _training_finished(last_pt):
        resume = last_pt.exists()
        model = YOLO(str(last_pt) if resume else hp["weights"])
        _add_callbacks(model, cfg, sched)
        if resume:
            C.log(f"resuming from {last_pt}")
            model.train(resume=True)
        else:
            model.train(
                data=str(Path(cfg["DATA_DIR"]) / "yolo" / f"split_{p}.yaml"),
                project=str(rd.parent), name=rd.name, exist_ok=True, device=0,
                epochs=sched["epochs"], imgsz=cfg["IMG_SIZE"], batch=hp["micro_batch"], nbs=cfg["EFFECTIVE_BATCH"],
                optimizer=hp["optimizer"], lr0=hp["lr"], lrf=hp["lrf"], momentum=hp["momentum"],
                weight_decay=hp["weight_decay"], warmup_epochs=sched["warmup_epochs"], cos_lr=False,
                # shared augmentation spec: horizontal flip + scale jitter only
                fliplr=cfg["HFLIP_P"], scale=cfg["SCALE_JITTER"], mosaic=0.0, close_mosaic=0, mixup=0.0,
                cutmix=0.0, copy_paste=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                shear=0.0, perspective=0.0, flipud=0.0, bgr=0.0, multi_scale=0.0,
                amp=cfg["AMP"], seed=cfg["SEED"], deterministic=False, workers=cfg["NUM_WORKERS"],
                patience=0, max_det=cfg["MAX_DET"], plots=False, val=True, save=True)

    best = YOLO(str(best_pt))
    best_val = torch.load(best_pt, map_location="cpu", weights_only=False).get("train_metrics", {})

    def predict(path):
        r = best.predict(str(path), imgsz=cfg["IMG_SIZE"], conf=cfg["SCORE_THRESH"], max_det=cfg["MAX_DET"],
                         retina_masks=True, half=cfg["AMP"], device=0, verbose=False)[0]
        if r.masks is None:
            return []
        masks = (r.masks.data > 0.5).cpu().numpy()
        return [{"category_id": int(c) + 1, "score": float(s), "mask": masks[i]}
                for i, (c, s) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()))]

    with torch.no_grad():
        metrics, per_class = C.coco_evaluate(C.coco_json(cfg, "test"), predict, C.image_dir(cfg),
                                             max_det=cfg["MAX_DET"], tag="TEST")
    C.finalize(cfg, model_key, frac, status, best_pt, last_pt, metrics, per_class,
               extra={"best_val_ultralytics": {k: float(v) for k, v in best_val.items()}, "schedule": sched})
