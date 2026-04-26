"""Compute mDice / mIoU / per-class Dice for a trained mmseg checkpoint.

Re-runs evaluation on the chosen split (val/test) using mmseg's own
IoUMetric so the numbers match what's printed during training. Saves a
single-row CSV with per-class metrics next to the requested output path.

Usage:
    python practicum_work/src/analysis/compute_metrics.py \
        --config configs/student/baselines/unet_fcn_bs8_5k.py \
        --ckpt   work_dirs/unet_fcn_bs8_5k/best_mDice_iter_5000.pth \
        --split  val

Run from the `cv_seg_final_task/mmsegmentation/` directory (so config
paths resolve and `mmseg.datasets.student_dataset` can be imported).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Make sure the forked mmseg package is importable when running from
# any cwd (e.g. from the project root).
# Layout: mmsegmentation/practicum_work/src/analysis/compute_metrics.py
#                ^parents[3]      [2]   [1]      [0]
THIS_FILE = Path(__file__).resolve()
MMSEG_ROOT = THIS_FILE.parents[3]
if str(MMSEG_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSEG_ROOT))

from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.utils import register_all_modules

# PyTorch 2.6 defaults torch.load to weights_only=True; mmengine ckpts
# pickle objects (HistoryBuffer, numpy arrays, ...) that aren't on the
# default allowlist. We trust our own checkpoints, so monkey-patch
# torch.load to default weights_only=False.
import torch
_orig_load = torch.load


def _trusted_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(f, *args, **kwargs)


torch.load = _trusted_load


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="path to mmseg config")
    p.add_argument("--ckpt", type=Path, required=True, help="path to .pth checkpoint")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="where to write the per-class metrics CSV "
             "(default: practicum_work/supplementary/viz/baselines/<exp>_<split>.csv)",
    )
    return p.parse_args()


def evaluate(cfg_path: Path, ckpt_path: Path, split: str) -> dict:
    register_all_modules()
    cfg = Config.fromfile(str(cfg_path))
    cfg.load_from = str(ckpt_path)

    # Drop train pieces — Runner.test() doesn't need them, and dropping
    # them avoids accidentally building unused dataloaders.
    cfg.train_cfg = None
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None

    # Don't log this re-evaluation to ClearML (or any remote backend) —
    # we just want clean metric numbers locally.
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer',
    )

    # Drop the CheckpointHook — its `after_val_epoch` tries to save a
    # "best" ckpt and crashes in val-only mode (no train Runner state).
    if 'checkpoint' in cfg.default_hooks:
        cfg.default_hooks.pop('checkpoint')

    # If the user wants val metrics, just run val; otherwise run test.
    runner = Runner.from_cfg(cfg)
    if split == "val":
        metrics = runner.val()
    else:
        metrics = runner.test()
    return dict(metrics)


def write_csv(metrics: dict, exp_name: str, split: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "split", "metric", "value"])
        for k, v in metrics.items():
            w.writerow([exp_name, split, k, v])
    print(f"\nMetrics written to: {out_csv}")


def main() -> None:
    args = parse_args()

    exp_name = args.config.stem
    if args.out_csv is None:
        viz_dir = THIS_FILE.parents[2] / "supplementary" / "viz" / "baselines"
        args.out_csv = viz_dir / f"{exp_name}_{args.split}.csv"

    metrics = evaluate(args.config, args.ckpt, args.split)
    print("\n=== final metrics ===")
    for k, v in metrics.items():
        print(f"  {k:30s} {v}")
    write_csv(metrics, exp_name, args.split, args.out_csv)


if __name__ == "__main__":
    main()
