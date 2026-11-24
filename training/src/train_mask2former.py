"""Mask2Former Swin-T (HF transformers, COCO-instance pretrained) -- transformer, query-based."""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import common as C
import common_torch as CT

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PAD_DIV = 32
MASK_CHUNK = 16


def _pad_batch(images, device):
    """Normalize, pad to a common size divisible by 32, pixel_mask marks valid pixels."""
    hmax = -(-max(i.shape[1] for i in images) // PAD_DIV) * PAD_DIV
    wmax = -(-max(i.shape[2] for i in images) // PAD_DIV) * PAD_DIV
    pv = torch.zeros((len(images), 3, hmax, wmax), device=device)
    pm = torch.zeros((len(images), hmax, wmax), dtype=torch.long, device=device)
    mean, std = MEAN.to(device), STD.to(device)
    for k, img in enumerate(images):
        _, h, w = img.shape
        pv[k, :, :h, :w] = (img.to(device) - mean) / std
        pm[k, :h, :w] = 1
    return pv, pm


class Mask2FormerAdapter:
    def __init__(self, cfg, hp):
        self.cfg, self.hp = cfg, hp
        self.names = C.class_names(cfg)
        self.micro_batch = hp["micro_batch"]
        self.clip_norm = hp["clip_norm"]
        self.lr_lambda = CT.poly_lambda(cfg, hp["poly_power"])

    def build(self):
        from transformers import Mask2FormerForUniversalSegmentation
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            self.hp["checkpoint"], num_labels=len(self.names),
            id2label=dict(enumerate(self.names)), label2id={n: i for i, n in enumerate(self.names)},
            ignore_mismatched_sizes=True)  # new class head, everything else COCO-pretrained

    def param_groups(self, model):
        """Paper recipe: backbone lr x0.1, no weight decay on norms / biases / embeddings."""
        groups = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            backbone = ".pixel_level_module.encoder." in f".{n}"
            no_wd = p.ndim == 1 or "embed" in n or "queries" in n or "norm" in n
            g = groups.setdefault((backbone, no_wd), {
                "params": [],
                "lr": self.hp["lr"] * (self.hp["backbone_lr_mult"] if backbone else 1.0),
                "weight_decay": 0.0 if no_wd else self.hp["weight_decay"]})
            g["params"].append(p)
        return list(groups.values())

    def loss(self, model, images, targets, device):
        pv, pm = _pad_batch(images, device)
        hp_, wp_ = pv.shape[-2:]
        mask_labels, class_labels = [], []
        for t in targets:
            m = t["masks"].to(device).float()
            mask_labels.append(F.pad(m, (0, wp_ - m.shape[-1], 0, hp_ - m.shape[-2])))
            class_labels.append(t["labels"].to(device))
        out = model(pixel_values=pv, pixel_mask=pm, mask_labels=mask_labels, class_labels=class_labels)
        return out.loss, {}

    def predictor(self, model, device):
        """Standard Mask2Former instance inference: top-k over (query, class), score = cls prob x mask quality."""
        size, max_det = self.cfg["IMG_SIZE"], self.cfg["MAX_DET"]

        def predict(path):
            img = Image.open(path).convert("RGB")
            w0, h0 = img.size
            r = size / max(w0, h0)
            nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
            pv, pm = _pad_batch([TF.to_tensor(img.resize((nw, nh), Image.Resampling.BILINEAR))], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.cfg["AMP"]):
                out = model(pixel_values=pv, pixel_mask=pm)
            cls = out.class_queries_logits[0].float().softmax(-1)[:, :-1]  # (Q, C) without "no object"
            n_cls = cls.shape[1]
            scores, idx = cls.flatten().topk(min(max_det, cls.numel()))
            q, labels = idx // n_cls, idx % n_cls
            ml = out.masks_queries_logits[0][q].float()[None]  # (1, K, h/4, w/4)
            ml = F.interpolate(ml, size=pv.shape[-2:], mode="bilinear", align_corners=False)[0, :, :nh, :nw]
            binm = ml > 0  # mask quality scored at network resolution (IMG_SIZE)
            mask_score = (ml.sigmoid() * binm).flatten(1).sum(1) / binm.flatten(1).sum(1).clamp(min=1)
            final = (scores * mask_score).tolist()
            # upsample to the original image in chunks: 100 full-res fp32 masks would not fit in memory
            binm = torch.cat([(F.interpolate(c[None], size=(h0, w0), mode="bilinear", align_corners=False)[0] > 0).cpu()
                              for c in ml.split(MASK_CHUNK)]).numpy()
            return [{"category_id": int(l) + 1, "score": s, "mask": binm[k]}
                    for k, (l, s) in enumerate(zip(labels.tolist(), final))]
        return predict


def run(cfg, model_key, frac, status):
    adapter = Mask2FormerAdapter(cfg, cfg["MODELS"][model_key])
    model, best_w, last_w, sched, best_val = CT.train_loop(cfg, model_key, frac, status, adapter)
    CT.evaluate_test(cfg, model_key, frac, status, adapter.predictor(model, torch.device("cuda")),
                     best_w, last_w, sched, best_val)
