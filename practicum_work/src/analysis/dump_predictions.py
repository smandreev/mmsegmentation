"""Dump per-image predictions for a trained checkpoint and rank by Dice.

For each image in val/test we compute per-image macro-Dice across the
3 classes (background, cat, dog), sort, and save:
- top-K best (highest Dice) → viz/baselines/<exp>/best/
- top-K worst (lowest Dice) → viz/baselines/<exp>/worst/
- a CSV with all per-image scores, sorted.

Each saved figure shows: image | ground-truth overlay | prediction overlay.

Usage:
    python practicum_work/src/analysis/dump_predictions.py \
        --config configs/student/baselines/unet_fcn_bs8_5k.py \
        --ckpt   work_dirs/unet_fcn_bs8_5k/best_mDice_iter_5000.pth \
        --split  val \
        --topk   5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Layout: mmsegmentation/practicum_work/src/analysis/dump_predictions.py
#                ^parents[3]      [2]   [1]      [0]
THIS_FILE = Path(__file__).resolve()
MMSEG_ROOT = THIS_FILE.parents[3]
if str(MMSEG_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSEG_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

# PyTorch 2.6 default weights_only=True breaks mmengine ckpts. Trust our
# own ckpts and monkey-patch the default.
_orig_torch_load = torch.load


def _trusted_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _trusted_load

CLASSES = ("background", "cat", "dog")
PALETTE = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8
)
NUM_CLASSES = len(CLASSES)
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: practicum_work/supplementary/viz/baselines/<exp>/<split>/",
    )
    return p.parse_args()


def per_image_dice(pred: np.ndarray, target: np.ndarray) -> Tuple[float, dict]:
    """Macro-Dice across classes for one image.

    Classes that are absent from BOTH pred and target contribute Dice=1.
    Per-class breakdown is also returned for the report.
    """
    per_class = {}
    dices = []
    for cls_id, cls_name in enumerate(CLASSES):
        p = pred == cls_id
        t = target == cls_id
        if not p.any() and not t.any():
            d = 1.0
        else:
            inter = (p & t).sum()
            d = (2.0 * inter) / (p.sum() + t.sum() + EPS)
        per_class[cls_name] = float(d)
        dices.append(float(d))
    return float(np.mean(dices)), per_class


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = PALETTE[mask]
    return (img * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)


def collect_split_paths(cfg: Config, split: str) -> List[Tuple[Path, Path, str]]:
    """Return list of (image_path, mask_path, stem) for the chosen split."""
    dl_cfg = cfg.val_dataloader if split == "val" else cfg.test_dataloader
    ds_cfg = dl_cfg.dataset
    data_root = Path(ds_cfg.data_root)
    if not data_root.is_absolute():
        data_root = MMSEG_ROOT / data_root
    img_dir = data_root / ds_cfg.data_prefix.img_path
    msk_dir = data_root / ds_cfg.data_prefix.seg_map_path
    img_suffix = ds_cfg.get("img_suffix", ".jpg")
    msk_suffix = ds_cfg.get("seg_map_suffix", ".png")
    out = []
    for img_path in sorted(img_dir.glob(f"*{img_suffix}")):
        msk_path = msk_dir / (img_path.stem + msk_suffix)
        if not msk_path.exists():
            continue
        out.append((img_path, msk_path, img_path.stem))
    return out


def render(
    img: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, save_to: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    axes[0].imshow(img); axes[0].set_title("input"); axes[0].axis("off")
    axes[1].imshow(overlay(img, gt)); axes[1].set_title("ground truth"); axes[1].axis("off")
    axes[2].imshow(overlay(img, pred)); axes[2].set_title("prediction"); axes[2].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(str(args.config))

    # Strip ClearML/remote vis_backends to avoid network calls during
    # offline analysis. Write a temp config with the patched visualizer
    # and pass that to init_model.
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer',
    )
    import tempfile
    tmp_cfg = Path(tempfile.mkstemp(suffix='.py')[1])
    cfg.dump(str(tmp_cfg))

    exp_name = args.config.stem
    if args.out_dir is None:
        args.out_dir = (
            THIS_FILE.parents[2] / "supplementary" / "viz" / "baselines"
            / exp_name / args.split
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}")
    model = init_model(str(tmp_cfg), str(args.ckpt), device=args.device)
    tmp_cfg.unlink(missing_ok=True)

    samples = collect_split_paths(cfg, args.split)
    print(f"Found {len(samples)} samples in split '{args.split}'")

    rows = []
    for img_path, msk_path, stem in tqdm(samples, desc=f"infer {args.split}"):
        gt = np.asarray(Image.open(msk_path))
        result = inference_model(model, str(img_path))
        pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)

        dice_macro, per_class = per_image_dice(pred, gt)
        rows.append({
            "stem": stem,
            "dice_macro": dice_macro,
            **{f"dice_{k}": v for k, v in per_class.items()},
            "img": str(img_path),
            "mask": str(msk_path),
            "pred_arr_idx": len(rows),
        })

    rows.sort(key=lambda r: r["dice_macro"])
    csv_path = args.out_dir.parent / f"{args.split}_per_image_dice.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["stem", "dice_macro"] + [f"dice_{c}" for c in CLASSES]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved per-image scores: {csv_path}")

    # render top-K worst (lowest dice) and top-K best (highest dice)
    worst = rows[: args.topk]
    best = rows[-args.topk :][::-1]

    def render_set(items, kind: str) -> None:
        out_subdir = args.out_dir / kind
        for r in items:
            img = np.asarray(Image.open(r["img"]).convert("RGB"))
            gt = np.asarray(Image.open(r["mask"]))
            result = inference_model(model, r["img"])
            pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
            title = (
                f"{r['stem']} | dice_macro={r['dice_macro']:.3f} "
                f"(bg={r['dice_background']:.2f}, cat={r['dice_cat']:.2f}, "
                f"dog={r['dice_dog']:.2f})"
            )
            render(img, gt, pred, title, out_subdir / f"{r['stem']}.png")

    render_set(worst, "worst")
    render_set(best, "best")

    print(f"\nWrote top-{args.topk} best  to {args.out_dir / 'best'}")
    print(f"Wrote top-{args.topk} worst to {args.out_dir / 'worst'}")


if __name__ == "__main__":
    main()
