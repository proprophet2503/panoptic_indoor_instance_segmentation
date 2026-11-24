"""Mask R-CNN R50-FPN (torchvision, COCO-pretrained) -- CNN, two-stage."""
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import common as C
import common_torch as CT


class MaskRCNNAdapter:
    def __init__(self, cfg, hp):
        self.cfg, self.hp = cfg, hp
        self.num_classes = len(C.class_names(cfg))
        self.micro_batch = hp["micro_batch"]
        self.clip_norm = None
        self.lr_lambda = CT.multistep_lambda(cfg, hp["lr_milestones"], hp["lr_gamma"])

    def build(self):
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        nc = self.num_classes + 1  # + background
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, nc)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, nc)
        # Train images arrive already resized by the shared dataset, eval resizes long side to IMG_SIZE.
        m.transform._skip_resize = True  # torchvision only consults this in train mode
        m.transform.min_size = (self.cfg["IMG_SIZE"],)
        m.transform.max_size = self.cfg["IMG_SIZE"]
        m.roi_heads.score_thresh = self.cfg["SCORE_THRESH"]
        m.roi_heads.detections_per_img = self.cfg["MAX_DET"]
        return m

    def param_groups(self, model):
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                (no_decay if n.endswith(".bias") else decay).append(p)
        return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

    def loss(self, model, images, targets, device):
        imgs, tgts = [], []
        for img, t in zip(images, targets):
            masks = t["masks"]
            keep, boxes = [], []
            for k in range(masks.shape[0]):
                ys, xs = torch.where(masks[k] > 0)
                keep.append(k)
                boxes.append(torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float())
            imgs.append(img.to(device, non_blocking=True))
            tgts.append({
                "boxes": torch.stack(boxes).to(device) if boxes else torch.zeros((0, 4), device=device),
                "labels": (t["labels"][keep] + 1).to(device),  # 1..num_classes (0 = background)
                "masks": masks[keep].to(device),
            })
        losses = model(imgs, tgts)
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}

    def predictor(self, model, device):
        def predict(path):
            img = Image.open(path).convert("RGB")
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.cfg["AMP"]):
                out = model([TF.to_tensor(img).to(device)])[0]
            masks = (out["masks"][:, 0] > 0.5).cpu().numpy()
            # torchvision labels 1..N == our COCO category ids
            return [{"category_id": int(l), "score": float(s), "mask": masks[i]}
                    for i, (l, s) in enumerate(zip(out["labels"].tolist(), out["scores"].tolist()))]
        return predict


def run(cfg, model_key, frac, status):
    adapter = MaskRCNNAdapter(cfg, cfg["MODELS"][model_key])
    model, best_w, last_w, sched, best_val = CT.train_loop(cfg, model_key, frac, status, adapter)
    CT.evaluate_test(cfg, model_key, frac, status, adapter.predictor(model, torch.device("cuda")),
                     best_w, last_w, sched, best_val)
