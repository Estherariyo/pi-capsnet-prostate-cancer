"""Runs Grad-CAM over a validation cache, computes IoU/Dice overlap between
heatmaps and ground-truth lesion masks on lesion-positive slices (the
standard validation practice for CNN-based prostate cancer explainability),
and saves a figure for every lesion-positive slice by default (--n-examples
caps this if you only want a handful).

CORRECTNESS FIX (post-hoc audit): "lesion-positive" used to also require
rank >= rank_threshold (default 3), which silently dropped rank-2 (ISUP 1)
slices that DO have an annotated lesion but aren't clinically significant
by that cutoff. Inclusion is now purely mask.sum() > 0 -- rank_threshold
still selects which ordinal bit's capsule activation Grad-CAM targets, but
no longer gates which slices get evaluated/saved."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grad_cam import GradCAM, overlap_metrics
from minisegcaps_model import MiniSegCaps


def load_batch(files):
    images, masks, ranks, names = [], [], [], []
    for f in files:
        d = np.load(f)
        images.append(d["image"])
        masks.append(d["mask"][0])
        ranks.append(int(d["rank"]))
        names.append(os.path.basename(f))
    return np.stack(images), np.stack(masks), np.array(ranks), names


def save_example(out_path, image, gt_mask, cam, rank, prob, title_extra=""):
    t2 = image[0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].imshow(t2, cmap="gray")
    axes[0].set_title("T2W input")
    axes[1].imshow(t2, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].contour(gt_mask, colors="lime", linewidths=1.2)
    axes[1].set_title("Grad-CAM + GT lesion outline")
    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground-truth lesion mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"rank={rank}  csPCa_prob={prob:.2f}  {title_extra}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/minisegcaps/gradcam/valid_cache")
    ap.add_argument("--checkpoint", default="/workspace/minisegcaps/gradcam/reference_checkpoint.pt")
    ap.add_argument("--rank-threshold", type=int, default=3, help="csPCa rank>=3 by default")
    ap.add_argument("--cam-threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="/workspace/minisegcaps/gradcam/results")
    ap.add_argument("--n-examples", type=int, default=-1,
                     help="cap on saved figures; <=0 means unlimited (save every lesion-positive slice)")
    ap.add_argument("--seg-first-cascade", action="store_true",
                     help="Approach B checkpoints: route Grad-CAM through the seg-first forward path")
    ap.add_argument("--gate-floor", type=float, default=1.0,
                     help="must match the checkpoint's own training --gate-floor, or the CAM reflects "
                          "the wrong gate strength")
    ap.add_argument("--gate-ceil", type=float, default=2.0)
    args = ap.parse_args()
    unlimited = args.n_examples <= 0

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "examples"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniSegCaps().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    cam_engine = GradCAM(model, device)

    files = sorted(glob.glob(os.path.join(args.data_root, "*.npz")))
    print(f"found {len(files)} cached slices")

    ious, dices = [], []
    example_saved = 0
    batch_size = 16
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        images, masks, ranks, names = load_batch(batch_files)
        x = torch.from_numpy(images).float()

        cams, probs = cam_engine(x, rank_threshold=args.rank_threshold, seg_first=args.seg_first_cascade,
                                  gate_floor=args.gate_floor, gate_ceil=args.gate_ceil)

        for j in range(len(batch_files)):
            is_positive = masks[j].sum() > 0
            if is_positive:
                iou, dice = overlap_metrics(cams[j], masks[j], cam_threshold=args.cam_threshold)
                if not np.isnan(iou):
                    ious.append(iou)
                    dices.append(dice)
                if unlimited or example_saved < args.n_examples:
                    bit_idx = {2: 0, 3: 1, 4: 2, 5: 3}[args.rank_threshold]
                    out_path = os.path.join(args.out_dir, "examples", f"example_{example_saved:03d}_{names[j]}.png")
                    save_example(out_path, images[j], masks[j], cams[j], ranks[j], probs[j][bit_idx])
                    example_saved += 1

        if i % (batch_size * 20) == 0:
            print(f"processed {i}/{len(files)} slices, positives so far: {len(ious)}, "
                  f"figures saved: {example_saved}", flush=True)

    summary = {
        "rank_threshold": args.rank_threshold,
        "cam_threshold": args.cam_threshold,
        "n_slices_total": len(files),
        "n_positive_slices_evaluated": len(ious),
        "n_example_figures_saved": example_saved,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "std_iou": float(np.std(ious)) if ious else None,
        "median_iou": float(np.median(ious)) if ious else None,
        "mean_dice": float(np.mean(dices)) if dices else None,
        "std_dice": float(np.std(dices)) if dices else None,
        "median_dice": float(np.median(dices)) if dices else None,
    }
    with open(os.path.join(args.out_dir, "gradcam_overlap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
