"""Run ONE experiment (model x train fraction) on the single GPU exposed via CUDA_VISIBLE_DEVICES.

Exit codes: 0 = finished (test metrics written), 75 = paused at an epoch boundary, other = failure.
"""
import argparse
import importlib
import random

import numpy as np
import torch

import common as C

TRAINERS = {"yolo": "train_yolo", "maskrcnn": "train_maskrcnn", "mask2former": "train_mask2former",
            "rfdetr": "train_rfdetr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frac", type=float, required=True)
    args = ap.parse_args()

    cfg = C.load_config()
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])
    torch.manual_seed(cfg["SEED"])
    status = C.read_json(C.status_path(cfg, args.model, args.frac))
    C.log(f"worker start {C.exp_name(args.model, args.frac)} on {torch.cuda.get_device_name(0)}")
    trainer = importlib.import_module(TRAINERS[cfg["MODELS"][args.model]["impl"]])
    trainer.run(cfg, args.model, args.frac, status)


if __name__ == "__main__":
    main()
