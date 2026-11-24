"""Analysis on the own test set: hard-condition subsets, per-image failure diagnostics and the small experiment of the
failure analysis. Reuses the predictors and ground truth of evaluate_models.ipynb, so every number uses the same
models, labels and COCO protocol.

Run from anywhere with the evaluation environment (torch, ultralytics, rfdetr overlay in evaluation/.pylib):
    python evaluation/own_test_analysis.py
"""
import json
import os
import pickle
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
OUT = EVAL_DIR / "analysis_own_test"
CACHE = EVAL_DIR / ".cache" / "own_preds"


def load_notebook() -> dict[str, Any]:
    """Run every code cell of evaluate_models.ipynb before the run loop (setup, config, ground truth, predictors)."""
    os.chdir(ROOT)
    nb = json.loads((EVAL_DIR / "evaluate_models.ipynb").read_text())
    ns: dict[str, Any] = {"__name__": "evaluate_models"}
    for cell in nb["cells"][:11]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)
    return ns


NB = load_notebook()
np, pd, torch, cv2, C = NB["np"], NB["pd"], NB["torch"], NB["cv2"], NB["C"]
mask_utils, Image = NB["mask_utils"], NB["Image"]
from pycocotools.coco import COCO  # noqa: E402  (pycocotools comes from the notebook's sys.path)
from pycocotools.cocoeval import COCOeval  # noqa: E402

OWN = NB["TEST_SETS"]["own"]
GT = COCO(str(OWN["gt"]))
NAMES, MODELS, SPLITS = NB["NAMES"], NB["MODELS"], NB["SPLITS"]


# ----------------------------------------------------------------------------- predictions
def to_dets(preds, max_det):
    keep = sorted(preds, key=lambda p: -p["score"])[:max_det]
    return [{"category_id": int(p["category_id"]), "score": float(p["score"]),
             "rle": mask_utils.encode(np.asfortranarray(np.asarray(p["mask"], dtype=np.uint8)))}
            for p in keep if np.any(p["mask"])]


def predict_dir(predict, img_dir, files):
    with torch.no_grad():
        return {fn: to_dets(predict(img_dir / fn), NB["CFG"]["MAX_DET"]) for fn in files}


def cached_predictions(model_key, frac):
    """{file_name: [{"category_id", "score", "rle"}]} on the own test set, computed once per model and split."""
    path = CACHE / f"split_{C.pct(frac)}__{model_key}.pkl"
    if not path.exists():
        predict, model = NB["load_predictor"](model_key, frac)
        dets = predict_dir(predict, OWN["img_dir"], OWN["files"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(dets))
        del predict, model
        torch.cuda.empty_cache()
    return pickle.loads(path.read_bytes())


# ----------------------------------------------------------------------------- COCO metrics on (subsets of) images
def coco_metrics(dets, gt=GT, img_ids=None):
    """segm AP / AP50 / AR100 and mean AP of thing and stuff classes, restricted to img_ids when given."""
    fid = {gt.imgs[i]["file_name"]: i for i in gt.getImgIds()}
    res = []
    for fn, ds in dets.items():
        for d in ds:
            rle = dict(d["rle"])
            rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
            res.append({"image_id": fid[fn], "category_id": d["category_id"], "score": d["score"],
                        "segmentation": rle})
    ev = COCOeval(gt, gt.loadRes(res), "segm")
    ev.params.maxDets = [1, 10, NB["CFG"]["MAX_DET"]]
    if img_ids is not None:
        ev.params.imgIds = sorted(img_ids)
    with C.contextlib.redirect_stdout(C.io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    kinds = {"thing": [], "stuff": []}
    for k, cid in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        if (p > -1).any():
            kinds[gt.cats[cid]["supercategory"]].append(p[p > -1].mean())
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AR100": ev.stats[8],
            "AP_thing": float(np.mean(kinds["thing"])) if kinds["thing"] else float("nan"),
            "AP_stuff": float(np.mean(kinds["stuff"])) if kinds["stuff"] else float("nan")}


def reproducibility_check():
    """Re-run every model on the final labels and compare with the stored evaluate_models.ipynb results."""
    rows = []
    for k in MODELS:
        for f in SPLITS:
            stored = json.loads((OWN["out"] / f"split_{C.pct(f)}" / k / "metrics.json").read_text())
            ap = coco_metrics(cached_predictions(k, f))["AP"]
            rows.append({"model": k, "train_fraction": f, "segm_AP_rerun": round(ap, 4),
                         "segm_AP_stored": round(stored["segm_AP"], 4), "diff": round(ap - stored["segm_AP"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rerun_check.csv", index=False)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------------- hard conditions (test_set/hard_conditions.csv)
def hard_condition_subsets():
    tags = pd.read_csv(ROOT / "test_set" / "hard_conditions.csv").fillna("")
    fid = {GT.imgs[i]["file_name"]: i for i in GT.getImgIds()}
    subsets = {"semua": list(fid.values()),
               "sulit": [fid[f] for f in tags.file[tags.sulit == 1]],
               "normal": [fid[f] for f in tags.file[tags.sulit == 0]]}
    for t in ("cahaya_redup", "saling_menutupi", "sudut_ekstrem"):
        subsets[t] = [fid[f] for f, k in zip(tags.file, tags.kondisi_sulit) if t in k.split(";")]
    return subsets


def hard_condition_table():
    subsets = hard_condition_subsets()
    rows = []
    for k in MODELS:
        for f in SPLITS:
            dets = cached_predictions(k, f)
            for name, ids in subsets.items():
                m = coco_metrics(dets, img_ids=ids)
                rows.append({"model": k, "train_fraction": f, "subset": name, "num_images": len(ids),
                             **{key: round(v, 4) for key, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ap_hard_vs_normal.csv", index=False)
    print(df[df.train_fraction == 1.0].pivot(index="model", columns="subset", values="AP").round(3).to_string())
    return df


# ----------------------------------------------------------------------------- per-image failure diagnostics
VIS_THRESH = NB["VIS_THRESH"]  # same detections as the drawn images


def match_image(dets, img_id, gt=GT, thresh=VIS_THRESH):
    """Every ground-truth instance of one image gets one status, from the detections with score >= thresh:
    benar (same class IoU >= .5), salah_kelas (other class IoU >= .5), terpecah (>= 2 detections inside it cover it),
    bocor (one detection covers it but is > 2x larger), tidak_terdeteksi (< 30% covered), mask_buruk (the rest)."""
    info = gt.imgs[img_id]
    anns = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
    dt = [d for d in dets if d["score"] >= thresh]
    g_rle = [gt.annToRLE(a) for a in anns]
    d_rle = [d["rle"] for d in dt]
    g_area = mask_utils.area(g_rle).astype(float) if g_rle else np.zeros(0)
    d_area = mask_utils.area(d_rle).astype(float) if d_rle else np.zeros(0)
    iou = mask_utils.iou(d_rle, g_rle, [0] * len(g_rle)) if d_rle and g_rle else np.zeros((len(dt), len(anns)))
    inter = iou * (d_area[:, None] + g_area[None, :]) / (1 + iou)
    iog = inter / np.maximum(g_area[None, :], 1)
    iop = inter / np.maximum(d_area[:, None], 1)
    rows = []
    for j, a in enumerate(anns):
        same = [i for i, d in enumerate(dt) if d["category_id"] == a["category_id"]]
        best_same = max((iou[i, j] for i in same), default=0.0)
        best_any = iou[:, j].max() if len(dt) else 0.0
        inside = [i for i in range(len(dt)) if iop[i, j] >= 0.7 and iog[i, j] >= 0.1]
        if best_same >= 0.5:
            status = "benar"
        elif best_any >= 0.5:
            status = "salah_kelas"
        elif len(inside) >= 2 and iog[inside, j].sum() >= 0.5:
            status = "terpecah"
        elif any(iog[i, j] >= 0.7 and d_area[i] > 2 * g_area[j] for i in range(len(dt))):
            status = "bocor"
        elif (iog[:, j].max() if len(dt) else 0.0) < 0.3:
            status = "tidak_terdeteksi"
        else:
            status = "mask_buruk"
        top = int(iou[:, j].argmax()) if best_any > 0 else None
        rows.append({"file": info["file_name"], "ann_id": a["id"], "gt_class": NAMES[a["category_id"] - 1],
                     "gt_area_frac": round(g_area[j] / (info["width"] * info["height"]), 4),
                     "status": status, "best_iou_same_class": round(float(best_same), 3),
                     "best_iou_any": round(float(best_any), 3),
                     "best_pred_class": NAMES[dt[top]["category_id"] - 1] if top is not None else ""})
    n_fp = int(sum(iou[i].max() < 0.5 if len(anns) else True for i in range(len(dt))))
    return rows, n_fp


def failure_table(frac=1.0):
    """Per instance and per image status counts of every model at one split."""
    inst, per_img = [], []
    for k in MODELS:
        dets = cached_predictions(k, frac)
        for img_id in GT.getImgIds():
            fn = GT.imgs[img_id]["file_name"]
            rows, n_fp = match_image(dets[fn], img_id)
            inst += [{"model": k, **r} for r in rows]
            counts = pd.Series([r["status"] for r in rows]).value_counts()
            per_img.append({"model": k, "file": fn, "n_gt": len(rows), "false_positive": n_fp,
                            **{s: int(counts.get(s, 0)) for s in
                               ("benar", "salah_kelas", "terpecah", "bocor", "mask_buruk", "tidak_terdeteksi")}})
    inst, per_img = pd.DataFrame(inst), pd.DataFrame(per_img)
    per_img["recall50"] = (per_img.benar / per_img.n_gt).round(3)
    inst.to_csv(OUT / f"instances_split_{C.pct(frac)}.csv", index=False)
    per_img.to_csv(OUT / f"per_image_split_{C.pct(frac)}.csv", index=False)
    return inst, per_img


def matched_scores(dets, img_id, gt=GT):
    """Per ground-truth instance: highest score of a same-class detection with IoU >= .5 at any score (0 = none)."""
    anns = gt.loadAnns(gt.getAnnIds(imgIds=[img_id]))
    if not dets or not anns:
        return [0.0] * len(anns)
    iou = mask_utils.iou([d["rle"] for d in dets], [gt.annToRLE(a) for a in anns], [0] * len(anns))
    return [max((d["score"] for i, d in enumerate(dets)
                 if d["category_id"] == a["category_id"] and iou[i, j] >= 0.5), default=0.0)
            for j, a in enumerate(anns)]


# ----------------------------------------------------------------------------- experiment: simulated low light
LOW_LIGHT_STOPS = [0, 2, 3]  # 0 = the original photo


def simulate_low_light(img_bgr, stops, seed, full_well=4000.0, read_noise=0.002):
    """The same scene with 2**stops less light, amplified back to the same brightness like a phone raising ISO:
    Poisson shot noise and Gaussian read noise grow by 2**(stops/2) relative to the signal."""
    if stops == 0:
        return img_bgr
    rng = np.random.default_rng(seed)
    lin = (img_bgr.astype(np.float64) / 255.0) ** 2.2
    gain = 2.0 ** stops
    low = rng.poisson(lin / gain * full_well) / full_well + rng.normal(0.0, read_noise, lin.shape)
    return np.round(np.clip(low * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def low_light_experiment(frac=1.0):
    """Darken the photos that were not taken in dim light by 0, 2 and 3 stops and measure what the models lose.
    If dim light explains the failures, recall at the drawn threshold must drop with every stop."""
    tags = pd.read_csv(ROOT / "test_set" / "hard_conditions.csv").fillna("")
    files = [f for f, k in zip(tags.file, tags.kondisi_sulit) if "cahaya_redup" not in k.split(";")]
    fid = {GT.imgs[i]["file_name"]: i for i in GT.getImgIds()}
    dirs = {}
    for s in LOW_LIGHT_STOPS[1:]:
        d = EVAL_DIR / ".cache" / f"own_lowlight_{s}stops"
        (d / "images").mkdir(parents=True, exist_ok=True)
        for k, fn in enumerate(files):
            out = d / "images" / fn
            if not out.exists():
                img = simulate_low_light(cv2.imread(str(OWN["img_dir"] / fn)), s, seed=k)
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dirs[s] = d / "images"
    rows = []
    for model_key in MODELS:
        preds = {0: cached_predictions(model_key, frac)}
        cache = {s: CACHE / f"lowlight_{s}stops__split_{C.pct(frac)}__{model_key}.pkl" for s in dirs}
        if not all(p.exists() for p in cache.values()):
            predict, model = NB["load_predictor"](model_key, frac)
            for s, p in cache.items():
                if not p.exists():
                    p.write_bytes(pickle.dumps(predict_dir(predict, dirs[s], files)))
            del predict, model
            torch.cuda.empty_cache()
        preds.update({s: pickle.loads(p.read_bytes()) for s, p in cache.items()})
        for s in LOW_LIGHT_STOPS:
            dets = {fn: preds[s][fn] for fn in files}
            status = [r["status"] for fn in files for r in match_image(dets[fn], fid[fn])[0]]
            scores = [x for fn in files for x in matched_scores(dets[fn], fid[fn])]
            m = coco_metrics(dets, img_ids=[fid[f] for f in files])
            rows.append({"model": model_key, "stops": s, "num_images": len(files), "AP": round(m["AP"], 4),
                         "AP50": round(m["AP50"], 4), "recall50_at_vis": round(np.mean([x == "benar" for x in status]), 3),
                         "found_any_score": round(np.mean([x > 0 for x in scores]), 3),
                         "mean_score_found": round(float(np.mean([x for x in scores if x > 0])), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "experiment_low_light.csv", index=False)
    print(df.to_string(index=False))
    example = files[0]
    strip = [simulate_low_light(cv2.imread(str(OWN["img_dir"] / example)), s, seed=0) for s in LOW_LIGHT_STOPS]
    cv2.imwrite(str(OUT / "experiment_low_light_example.jpg"), np.hstack(strip), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return df


def real_dim_vs_normal(frac=1.0):
    """Same recall / score measures on the real photos, split into dim and not dim."""
    tags = pd.read_csv(ROOT / "test_set" / "hard_conditions.csv").fillna("")
    fid = {GT.imgs[i]["file_name"]: i for i in GT.getImgIds()}
    rows = []
    for model_key in MODELS:
        dets = cached_predictions(model_key, frac)
        for dim in (False, True):
            files = [f for f, k in zip(tags.file, tags.kondisi_sulit) if ("cahaya_redup" in k.split(";")) == dim]
            status = [r["status"] for fn in files for r in match_image(dets[fn], fid[fn])[0]]
            scores = [x for fn in files for x in matched_scores(dets[fn], fid[fn])]
            m = coco_metrics({f: dets[f] for f in files}, img_ids=[fid[f] for f in files])
            rows.append({"model": model_key, "foto": "redup" if dim else "tidak redup", "num_images": len(files),
                         "AP": round(m["AP"], 4), "recall50_at_vis": round(np.mean([x == "benar" for x in status]), 3),
                         "found_any_score": round(np.mean([x > 0 for x in scores]), 3),
                         "mean_score_found": round(float(np.mean([x for x in scores if x > 0])), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "real_dim_vs_normal.csv", index=False)
    print(df.to_string(index=False))
    return df


# chosen from per_image_split_100.csv (lowest recall over the three models) so that each shows another failure type
FAILURE_PHOTOS = ["20261123_115502.jpg", "20261123_233440.jpg", "20261121_200916.jpg", "20261123_233014.jpg",
                  "20261121_200651.jpg"]
MODEL_LABEL = {"yolo26l_seg": "YOLO26l-seg", "maskrcnn_r50_fpn": "Mask R-CNN R50-FPN", "rfdetr_seg_l": "RF-DETR-Seg-L"}


def failure_panel(fn, frac=1.0, width=640):
    """Ground truth next to the three models (detections >= VIS_THRESH) for one own test photo."""
    img = cv2.imread(str(OWN["img_dir"] / fn))
    img_id = next(i for i in GT.getImgIds() if GT.imgs[i]["file_name"] == fn)
    gt = [{"category_id": a["category_id"], "score": 1.0, "mask": GT.annToMask(a).astype(bool)}
          for a in GT.loadAnns(GT.getAnnIds(imgIds=[img_id]))]
    tiles = [("Ground truth", NB["draw_predictions"](img, gt))]
    for k in MODELS:
        preds = [{**d, "mask": mask_utils.decode(d["rle"]).astype(bool)}
                 for d in cached_predictions(k, frac)[fn] if d["score"] >= VIS_THRESH]
        tiles.append((MODEL_LABEL[k], NB["draw_predictions"](img, preds)))
    out = []
    for title, t in tiles:
        t = cv2.resize(t, (width, round(t.shape[0] * width / t.shape[1])))
        bar = np.full((34, width, 3), 255, np.uint8)
        cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        out.append(np.vstack([bar, t]))
    pad = np.full((out[0].shape[0], 8, 3), 255, np.uint8)
    grid = np.vstack([np.hstack([out[0], pad, out[1]]), np.full((8, 2 * width + 8, 3), 255, np.uint8),
                      np.hstack([out[2], pad, out[3]])])
    path = OUT / "failures" / f"{Path(fn).stem}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ----------------------------------------------------------------------------- black and white figures for the report
STYLE = {"yolo26l_seg": ("o", "-"), "maskrcnn_r50_fpn": ("s", "--"), "rfdetr_seg_l": ("^", ":")}


def report_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
                         "font.size": 11})
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def line(ax, x, y, k):
        m, ls = STYLE[k]
        ax.plot(x, y, color="black", marker=m, linestyle=ls, markerfacecolor="white", label=MODEL_LABEL[k])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".replace(".", ","))  # Indonesian decimal comma

    summary = pd.read_csv(EVAL_DIR / "summary_both_test_sets.csv")
    pcts = [C.pct(f) for f in SPLITS]
    panels = [("original", "segm_AP", "Test set asli (92 gambar)"), ("own", "segm_AP", "Test set sendiri (40 foto)"),
              ("original", "segm_AP_thing", "AP kelas thing, test set asli"),
              ("original", "segm_AP_stuff", "AP kelas stuff, test set asli")]
    for name, sel in (("fig_ap_vs_data_latih.png", panels[:2]), ("fig_thing_stuff_vs_data_latih.png", panels[2:])):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, (ts, col, title) in zip(axes, sel):
            for k in MODELS:
                d = summary[(summary.test_set == ts) & (summary.model == k)].sort_values("train_fraction")
                line(ax, pcts, d[col], k)
            ax.set_title(title)
            ax.set_xlabel("Data latih yang dipakai (%)")
            ax.set_xticks(pcts)
            ax.grid(color="0.85")
        axes[0].set_ylabel("Mask AP (IoU 0,50 sampai 0,95)")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=200)
        plt.close(fig)

    exp = pd.read_csv(OUT / "experiment_low_light.csv")
    real = pd.read_csv(OUT / "real_dim_vs_normal.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k in MODELS:
        d = exp[exp.model == k].sort_values("stops")
        line(ax, [0, 1, 2], d.recall50_at_vis, k)
        m, _ = STYLE[k]
        ax.plot([3.2], real[(real.model == k) & (real.foto == "redup")].recall50_at_vis, color="black", marker=m,
                linestyle="none", markersize=8)
    ax.set_xticks([0, 1, 2, 3.2], ["foto asli", "-2 EV", "-3 EV", "foto redup\nasli (9 foto)"])
    ax.axvline(2.6, color="0.6", linewidth=0.8)
    ax.set_ylabel("Recall instance (IoU 0,5, skor 0,5)")
    ax.set_xlabel("31 foto tidak redup dengan simulasi cahaya rendah")
    ax.set_ylim(0, 0.55)
    ax.grid(color="0.85")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_eksperimen_cahaya_rendah.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reproducibility_check()
    hard_condition_table()
    inst, per_img = failure_table(1.0)
    print(per_img.pivot(index="file", columns="model", values="recall50").assign(
        mean=lambda d: d.mean(axis=1)).sort_values("mean").head(12).to_string())
    print(inst.groupby(["model", "status"]).size().unstack(fill_value=0).to_string())
    for fn in FAILURE_PHOTOS:
        failure_panel(fn)
    real_dim_vs_normal()
    low_light_experiment()
    report_figures()
