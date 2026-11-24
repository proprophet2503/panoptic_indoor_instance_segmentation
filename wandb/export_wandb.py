# Export all W&B runs (config + summary + full per-step history) to CSV.
# Run `wandb login` once first.
import pathlib

import pandas as pd
import wandb

PROJECT = "jeremymattathiasmboe-ai-research/panoptic-indoor-instance-seg"
out = pathlib.Path(__file__).parent / "wandb_export"
out.mkdir(exist_ok=True)

api = wandb.Api(timeout=60)
rows = []
for run in api.runs(PROJECT):
    rows.append({
        "id": run.id, "name": run.name, "state": run.state, "created": run.created_at,
        **{f"config.{k}": v for k, v in run.config.items() if not k.startswith("_")},
        **{f"summary.{k}": v for k, v in run.summary._json_dict.items() if not k.startswith("_")},
    })
    # scan_history = every logged step, run.history() keeps only ~500 sampled points
    pd.DataFrame(run.scan_history()).to_csv(out / f"{run.id}_history.csv", index=False)
    print("done", run.id, run.name, flush=True)

pd.DataFrame(rows).to_csv(out / "runs_summary.csv", index=False)
print(f"{len(rows)} runs -> {out}")
