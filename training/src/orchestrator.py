"""Experiment queue (model-major, then split) executed on N GPUs in parallel, resumable across Kaggle sessions."""
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import common as C

SRC = Path(__file__).resolve().parent
POLL_SEC = 20
REPORT_EVERY_SEC = 600
KILL_GRACE_SEC = 60


# ----------------------------------------------------------------------------- state
def init_status(cfg, model_key, frac):
    path = C.status_path(cfg, model_key, frac)
    status = C.read_json(path)
    if status is None:
        status = {"model": model_key, "train_fraction": frac, "family": cfg["MODELS"][model_key]["family"],
                  "status": "pending", "attempts": 0, "wandb_id": uuid.uuid4().hex[:12], "wandb_url": ""}
        C.atomic_write_json(path, status)
    return status


def set_status(cfg, model_key, frac, **kw):
    status = C.read_json(C.status_path(cfg, model_key, frac))
    status.update(kw)
    C.atomic_write_json(C.status_path(cfg, model_key, frac), status)
    return status


def pending_jobs(cfg):
    jobs = []
    for model_key in cfg["MODEL_ORDER"]:  # loop: model, then training split
        for frac in cfg["TRAIN_FRACTIONS"]:
            st = init_status(cfg, model_key, frac)
            if st["status"] == "done":
                continue
            if st["status"] == "failed" and st["attempts"] >= cfg["MAX_ATTEMPTS"]:
                C.log(f"skip {C.exp_name(model_key, frac)}: failed {st['attempts']}x (see its log)")
                continue
            jobs.append((model_key, frac))
    return jobs


# ----------------------------------------------------------------------------- reports
def write_reports(cfg):
    out = Path(cfg["OUTPUT_DIR"])
    rows, results = [], []
    for model_key in cfg["MODEL_ORDER"]:
        for frac in cfg["TRAIN_FRACTIONS"]:
            st = C.read_json(C.status_path(cfg, model_key, frac)) or {}
            rows.append({"model": model_key, "family": cfg["MODELS"][model_key]["family"], "train_fraction": frac,
                         "status": st.get("status", "pending"), "attempts": st.get("attempts", 0),
                         "wandb_run_id": st.get("wandb_id", ""), "wandb_url": st.get("wandb_url", ""),
                         "finished_at": st.get("finished_at", "")})
            m = C.read_json(C.out_dir(cfg, model_key, frac) / "metrics_test.json")
            if m:
                sched = m.get("schedule", {})
                results.append({"model": model_key, "family": cfg["MODELS"][model_key]["family"],
                                "train_fraction": frac, "n_train": sched.get("n_train"),
                                "epochs": sched.get("epochs"), "optimizer_steps": sched.get("total_steps"),
                                **{k: m.get(k) for k in C.METRIC_NAMES + ["segm_AP_thing", "segm_AP_stuff",
                                                                         "classes_in_eval", "fps"]}})
    for name, data in (("wandb_runs.csv", rows), ("results.csv", results)):
        if data:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    return rows, results


def _tail(path, n=15):
    try:
        return "".join(Path(path).read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- runner
def _launch(cfg, gpu, model_key, frac):
    log_path = Path(cfg["OUTPUT_DIR"]) / "logs" / f"{C.exp_name(model_key, frac)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = C.read_json(C.status_path(cfg, model_key, frac))["attempts"] + 1
    set_status(cfg, model_key, frac, status="running", gpu=gpu, attempts=attempts,
               started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "EXP_CONFIG": cfg["CONFIG_PATH"],
           "PYTHONPATH": f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "PYTHONUNBUFFERED": "1",
           "WANDB_DIR": cfg["WANDB_DIR"], "WANDB_SILENT": "true"}
    Path(cfg["WORK_CWD"]).mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a")
    logf.write(f"\n===== session {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {gpu} attempt {attempts} =====\n")
    logf.flush()
    proc = subprocess.Popen([sys.executable, str(SRC / "worker.py"), "--model", model_key, "--frac", str(frac)],
                            stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=cfg["WORK_CWD"],
                            start_new_session=True)
    C.log(f"GPU{gpu} <- {C.exp_name(model_key, frac)} (attempt {attempts}, log {log_path.name})")
    return {"proc": proc, "job": (model_key, frac), "log": log_path, "logf": logf}


def _finish(cfg, slot):
    model_key, frac = slot["job"]
    code = slot["proc"].returncode
    slot["logf"].close()
    st = C.read_json(C.status_path(cfg, model_key, frac))
    if code == 0 and st["status"] == "done":
        C.log(f"finished {C.exp_name(model_key, frac)}  test segm_AP={st.get('test_segm_AP', float('nan')):.4f}")
    elif code == C.EXIT_PAUSED:
        set_status(cfg, model_key, frac, status="paused")
        C.log(f"paused {C.exp_name(model_key, frac)} (resumes next session)")
    else:
        set_status(cfg, model_key, frac, status="failed", last_exit_code=code, error_tail=_tail(slot["log"], 30))
        C.log(f"FAILED {C.exp_name(model_key, frac)} exit={code}\n{_tail(slot['log'])}")


def _kill(slot):
    try:
        os.killpg(slot["proc"].pid, signal.SIGTERM)
        slot["proc"].wait(KILL_GRACE_SEC)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(slot["proc"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        slot["proc"].wait()


def run_all(cfg):
    stop_file = Path(cfg["STOP_FILE"])
    stop_file.unlink(missing_ok=True)
    budget_end = cfg["SESSION_START_TS"] + cfg["SESSION_BUDGET_HOURS"] * 3600
    pause_at, kill_at = budget_end - cfg["PAUSE_MARGIN_MIN"] * 60, budget_end
    queue = pending_jobs(cfg)
    C.log(f"{len(queue)} experiment(s) to run: {[C.exp_name(*j) for j in queue]}")
    C.log(f"pause signal at {time.strftime('%H:%M:%S', time.localtime(pause_at))}, "
          f"hard stop at {time.strftime('%H:%M:%S', time.localtime(kill_at))}")
    slots = {gpu: None for gpu in cfg["GPUS"]}
    last_report = time.time()

    while queue or any(slots.values()):
        now = time.time()
        if now >= pause_at and not stop_file.exists():
            stop_file.touch()
            C.log("session budget nearly used -> workers pause at their next epoch boundary")
        for gpu in slots:
            slot = slots[gpu]
            if slot and slot["proc"].poll() is not None:
                _finish(cfg, slot)
                slots[gpu] = slot = None
            if slot is None and queue and not stop_file.exists():
                slots[gpu] = _launch(cfg, gpu, *queue.pop(0))
        if stop_file.exists() and not any(slots.values()):
            break  # paused everything, remaining queue runs next session
        if now >= kill_at:
            for gpu, slot in slots.items():
                if slot:
                    C.log(f"hard stop: killing {C.exp_name(*slot['job'])} (resumes from last epoch checkpoint)")
                    _kill(slot)
                    slot["proc"].returncode = C.EXIT_PAUSED
                    _finish(cfg, slot)
                    slots[gpu] = None
            break
        if now - last_report >= REPORT_EVERY_SEC:
            last_report = now
            for gpu, slot in slots.items():
                if slot:
                    C.log(f"GPU{gpu} {C.exp_name(*slot['job'])} ...\n{_tail(slot['log'], 4)}")
            write_reports(cfg)
        time.sleep(POLL_SEC)

    stop_file.unlink(missing_ok=True)
    rows, _ = write_reports(cfg)
    left = [r for r in rows if r["status"] != "done"]
    C.log("ALL EXPERIMENTS DONE" if not left else
          f"{len(left)} experiment(s) not done -> Save Version, attach this version's output as input, run again")
    return rows


# ----------------------------------------------------------------------------- cross-session restore
MARKER = "PIPELINE_MARKER.json"


def restore_previous_outputs(cfg):
    """Copy outputs/ + runs/ of a previous notebook version (attached as input) into the working dir."""
    out = Path(cfg["OUTPUT_DIR"])
    if (out / MARKER).exists():
        C.log("working dir already has pipeline state -> nothing to restore")
        return False
    candidates = sorted(Path(cfg["INPUT_ROOT"]).glob(f"**/{out.name}/{MARKER}"))
    if not candidates:
        C.log("no previous outputs attached -> starting fresh")
        return False
    prev_out = max(candidates, key=lambda p: C.read_json(p).get("updated", "")).parent
    prev_runs = prev_out.parent / Path(cfg["RUNS_DIR"]).name
    C.log(f"restoring previous state from {prev_out.parent}")
    shutil.copytree(prev_out, out, dirs_exist_ok=True)
    if prev_runs.exists():
        shutil.copytree(prev_runs, cfg["RUNS_DIR"], dirs_exist_ok=True)
    for st_path in out.glob("split_*/*/status.json"):  # anything interrupted mid-run resumes
        st = C.read_json(st_path)
        if st["status"] == "running":
            st["status"] = "paused"
            C.atomic_write_json(st_path, st)
    return True


def write_marker(cfg):
    path = Path(cfg["OUTPUT_DIR"]) / MARKER
    prev = C.read_json(path, {})
    C.atomic_write_json(path, {"created": prev.get("created", time.strftime("%Y-%m-%d %H:%M:%S")),
                               "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "shared_config": {k: cfg[k] for k in cfg["SHARED_KEYS"]}})
