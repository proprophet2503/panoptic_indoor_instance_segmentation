"""Dataset + generic train loop shared by the two hand-written trainers (Mask R-CNN, Mask2Former)."""
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

import common as C


class CocoSegDataset(torch.utils.data.Dataset):
    """Shared augmentation spec: resize long side to a random TRAIN_SCALES value + horizontal flip (train only)."""

    def __init__(self, cfg, split_name, train):
        d = json.loads(C.coco_json(cfg, split_name).read_text())
        self.images = d["images"]
        self.anns = defaultdict(list)
        for a in d["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.img_dir = C.image_dir(cfg)
        self.train = train
        self.scales = cfg["TRAIN_SCALES"]
        self.base = cfg["IMG_SIZE"]
        self.hflip_p = cfg["HFLIP_P"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        w, h = img.size
        size = random.choice(self.scales) if self.train else self.base
        r = size / max(w, h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        flip = self.train and random.random() < self.hflip_p
        if flip:
            img = ImageOps.mirror(img)

        masks, labels = [], []
        for a in self.anns[info["id"]]:
            polys = []
            for poly in a["segmentation"]:
                xy = np.asarray(poly, dtype=np.float64).reshape(-1, 2) * r
                if flip:
                    xy[:, 0] = nw - xy[:, 0]
                polys.append(xy.flatten().tolist())
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polys, nh, nw)))
            if m.any():
                masks.append(m)
                labels.append(a["category_id"] - 1)  # 0-based class index
        masks = torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, nh, nw), dtype=torch.uint8)
        return TF.to_tensor(img), {"masks": masks, "labels": torch.tensor(labels, dtype=torch.int64)}


def list_collate(batch):
    return tuple(zip(*batch))


def save_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def warmup_factor(cfg, step):
    w = cfg["WARMUP_STEPS"]
    return 1.0 if step >= w else 0.001 + (1 - 0.001) * step / w


def poly_lambda(cfg, power):
    def f(step, total):
        return warmup_factor(cfg, step) * max(0.0, 1 - step / total) ** power
    return f


def multistep_lambda(cfg, milestones_frac, gamma):
    def f(step, total):
        return warmup_factor(cfg, step) * gamma ** sum(step >= math.floor(m * total) for m in milestones_frac)
    return f


def _seed_worker(base):
    def init(worker_id):
        random.seed(base + worker_id)
        np.random.seed((base + worker_id) % 2**32)
    return init


def train_loop(cfg, model_key, frac, status, adapter):
    """AMP train loop: grad accumulation, epoch-boundary pause, periodic val (unified COCO eval), atomic ckpts.

    adapter provides: micro_batch, clip_norm, build(), param_groups(model), lr_lambda(step, total),
    loss(model, images, targets, device) -> (loss, logs dict), predictor(model, device) -> predict_fn(path).
    """
    device = torch.device("cuda")
    hp = cfg["MODELS"][model_key]
    rd = C.run_dir(cfg, model_key, frac)
    rd.mkdir(parents=True, exist_ok=True)
    state_ckpt, best_w, last_w = rd / "last_state.pt", rd / "best.pt", rd / "last.pt"

    ds = CocoSegDataset(cfg, f"train_{C.pct(frac)}", train=True)
    sched = C.schedule(cfg, len(ds), adapter.micro_batch)
    C.wandb_init(cfg, model_key, frac, status, sched)
    C.log(f"{C.exp_name(model_key, frac)} schedule {sched}")

    model = adapter.build().to(device)
    groups = adapter.param_groups(model)
    if hp["optimizer"] == "SGD":
        opt = torch.optim.SGD(groups, lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"])
    else:
        opt = torch.optim.AdamW(groups, lr=hp["lr"], weight_decay=hp["weight_decay"])
    total = sched["total_steps"]
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: adapter.lr_lambda(s, total))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["AMP"])
    start_epoch, step, best_ap = 0, 0, -1.0

    if state_ckpt.exists():
        st = torch.load(state_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        lr_sched.load_state_dict(st["lr_sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, step, best_ap = st["epoch"], st["step"], st["best_ap"]
        C.log(f"resumed at epoch {start_epoch}, step {step}, best val AP {best_ap:.4f}")

    accum = sched["accumulate"]
    for epoch in range(start_epoch, sched["epochs"]):
        if C.should_stop(cfg):
            C.pause_and_exit()  # state_ckpt of the previous epoch is complete
        g = torch.Generator().manual_seed(cfg["SEED"] + epoch)  # same order after resume
        loader = torch.utils.data.DataLoader(
            ds, batch_size=adapter.micro_batch, sampler=torch.utils.data.RandomSampler(ds, generator=g),
            num_workers=cfg["NUM_WORKERS"], collate_fn=list_collate, pin_memory=True, drop_last=False,
            worker_init_fn=_seed_worker(cfg["SEED"] * 10_000 + epoch * 100), persistent_workers=False)
        model.train()
        n_micro = len(loader)
        for i, (images, targets) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["AMP"]):
                loss, logs = adapter.loss(model, images, targets, device)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum and (i + 1) != n_micro:
                continue
            if adapter.clip_norm:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), adapter.clip_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            lr_sched.step()
            step += 1
            if step % cfg["LOG_EVERY_STEPS"] == 0:
                C.wandb_log({"train/loss": float(loss), **{f"train/{k}": v for k, v in logs.items()},
                             "train/lr": opt.param_groups[0]["lr"], "train/epoch": epoch + 1}, step=step)

        if (epoch + 1) % sched["val_every_epochs"] == 0 or epoch + 1 == sched["epochs"]:
            model.eval()
            with torch.no_grad():
                m, _ = C.coco_evaluate(C.coco_json(cfg, "val"), adapter.predictor(model, device), C.image_dir(cfg),
                                       max_det=cfg["MAX_DET"], tag=f"val ep{epoch + 1}/{sched['epochs']}")
            C.wandb_log({**{f"val/{k}": v for k, v in m.items()}, "train/epoch": epoch + 1}, step=step)
            if m["segm_AP"] > best_ap:
                best_ap = m["segm_AP"]
                save_atomic({"model": model.state_dict(), "epoch": epoch + 1, "val_segm_AP": best_ap}, best_w)
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "lr_sched": lr_sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch + 1, "step": step, "best_ap": best_ap},
                    state_ckpt)

    if not last_w.exists():
        save_atomic({"model": model.state_dict(), "epoch": sched["epochs"]}, last_w)
    model.load_state_dict(torch.load(best_w, map_location="cpu", weights_only=False)["model"])
    model.eval()
    return model, best_w, last_w, sched, best_ap


def evaluate_test(cfg, model_key, frac, status, predict_fn, best_w, last_w, sched, best_val):
    with torch.no_grad():
        metrics, per_class = C.coco_evaluate(C.coco_json(cfg, "test"), predict_fn, C.image_dir(cfg),
                                             max_det=cfg["MAX_DET"], tag="TEST")
    C.finalize(cfg, model_key, frac, status, best_w, last_w, metrics, per_class,
               extra={"best_val_segm_AP": best_val, "schedule": sched})
