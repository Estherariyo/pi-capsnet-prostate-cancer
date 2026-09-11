"""Training loop matching the paper's optimization recipe: Adam(lr=2e-3),
step decay 0.8 every 20 epochs, 300 epochs, batch size scaled down from the
paper's 256 to fit slice count on this substitute dataset."""
import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from minisegcaps_model import MiniSegCaps
from losses import CombinedLoss
from dataset import SliceCacheDataset
from dataset_sequence import SliceSequenceDataset


def build_sample_weights(data_root, split, files):
    """Inverse-rank-frequency weight per sample, read from the manifest CSV
    (falls back to scanning the .npz files directly if no manifest exists).
    Corrects the severe class imbalance found in the unweighted PI-CAI run
    (~95% rank-1 slices), where plain random shuffling let the model learn
    to just predict the majority class."""
    manifest_path = os.path.join(data_root, f"{split}_manifest.csv")
    rank_by_name = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                rank_by_name[row["filename"]] = int(row["rank"])
    else:
        import numpy as np
        for fp in files:
            rank_by_name[os.path.basename(fp)] = int(np.load(fp)["rank"])

    ranks = [rank_by_name[os.path.basename(fp)] for fp in files]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    weights = [1.0 / counts[r] for r in ranks]
    return weights, counts


def run_epoch(model, loader, criterion, optimizer, device, train=True, max_steps=None, seg_first=False,
              gate_floor=1.0, gate_ceil=2.0, gate_bypass_prob=0.0):
    model.train(train)
    criterion.train(train)
    totals = {}
    n = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = batch["image"].to(device)
        seg_gt = batch["mask"].to(device)
        pirads_bits = batch["pirads_bits"].to(device)
        recon_mask = batch["recon_mask"].to(device)

        with torch.set_grad_enabled(train):
            if seg_first:
                # bypass only ever applies on training passes; model.train(False)
                # already guards eval, but passing 0.0 makes it explicit
                out = model.forward_seg_first(x, gate_floor=gate_floor, gate_ceil=gate_ceil,
                                               bypass_prob=gate_bypass_prob if train else 0.0)
            else:
                out = model.forward_single(x)
            loss, parts = criterion(out, seg_gt, pirads_bits, x, recon_mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * x.shape[0]
        n += x.shape[0]
    return {k: v / max(1, n) for k, v in totals.items()}


def run_epoch_sequence(model, loader, criterion, optimizer, device, train=True, max_steps=None,
                        gate_floor=1.0, gate_ceil=2.0):
    """Approach B v2, CapsGRU lever: same shape as run_epoch, but consumes
    SliceSequenceDataset's (B,T,C,H,W) batches and reads the loss inputs
    from the CENTER timestep's outputs -- pirads_prob comes from
    pirads_prob_refined (the whole point of CapsGRU is training against the
    GRU-refined call, not the un-refined per-slice one)."""
    model.train(train)
    criterion.train(train)
    totals = {}
    n = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x_seq = batch["image_seq"].to(device)
        seg_gt = batch["mask_center"].to(device)
        pirads_bits = batch["pirads_bits"].to(device)
        recon_target = batch["recon_target"].to(device)
        recon_mask = batch["recon_mask"].to(device)

        with torch.set_grad_enabled(train):
            out = model.forward_sequence_seg_first(x_seq, gate_floor=gate_floor, gate_ceil=gate_ceil)
            center = x_seq.shape[1] // 2
            adapted = {
                "seg_logits": out["seg_logits"][:, center],
                "pirads_prob": out["pirads_prob_refined"][:, center],
                "recon": out["recon"][:, center],
            }
            loss, parts = criterion(adapted, seg_gt, pirads_bits, recon_target, recon_mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bsz = x_seq.shape[0]
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * bsz
        n += bsz
    return {k: v / max(1, n) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/minisegcaps/data/picai_cache")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-decay", type=float, default=0.8)
    ap.add_argument("--lr-decay-every", type=int, default=20)
    ap.add_argument("--checkpoint-dir", default="/workspace/minisegcaps/checkpoints")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None,
                     help="cap steps/epoch, useful for smoke tests")
    ap.add_argument("--balanced", action="store_true",
                     help="use inverse-rank-frequency WeightedRandomSampler on the training set")
    ap.add_argument("--gated-cascade", action="store_true",
                     help="Approach A -- classify-then-segment cascade experiment: gate the "
                          "segmentation loss on the ground-truth csPCa label during training "
                          "(CombinedLoss's gate_seg_on_gt). Off by default -- this is a deliberate "
                          "architectural variant, not the paper's joint design, and should be run "
                          "as an explicit A/B against a non-gated fold before being trusted.")
    ap.add_argument("--gate-min-weight", type=float, default=0.0,
                     help="Approach A v2 fix: floor weight for GT-benign slices' segmentation loss "
                          "instead of the hard 0.0 used in v1 (which collapsed raw Dice to 0.33-0.54 "
                          "by never training the decoder to suppress false positives on benign "
                          "slices). Only meaningful with --gated-cascade.")
    ap.add_argument("--seg-first-cascade", action="store_true",
                     help="Approach B -- segment-then-classify cascade experiment: routes through "
                          "MiniSegCaps.forward_seg_first instead of forward_single (decoder runs "
                          "first with a neutral attention map, its own predicted mask gates the "
                          "capsule branch's input). Mutually exclusive with --gated-cascade in "
                          "practice -- only one cascade direction should be active per run.")
    ap.add_argument("--uncertainty-weighted", action="store_true",
                     help="Approach A v2 lever #3: learn the segmentation-vs-classification loss "
                          "balance via Kendall et al. 2018 uncertainty weighting instead of a fixed "
                          "1:1 ratio, to reduce shared-encoder gradient contention between the two "
                          "tasks. Adds two learnable scalars to CombinedLoss, included in the "
                          "optimizer alongside the model's own parameters.")
    ap.add_argument("--gate-floor", type=float, default=1.0,
                     help="Approach B v2 lever: forward_seg_first's lesion-prior gate multiplier for "
                          "non-lesion regions (lesion_prior=0). v1 default (1.0) never suppresses "
                          "background; <1.0 lets the classifier partially suppress it. Only "
                          "meaningful with --seg-first-cascade.")
    ap.add_argument("--gate-ceil", type=float, default=2.0,
                     help="Approach B v2 lever: forward_seg_first's gate multiplier for lesion "
                          "regions (lesion_prior=1). v1 default (2.0) preserved.")
    ap.add_argument("--capsgru", action="store_true",
                     help="Approach B v2 lever: train through forward_sequence_seg_first (the "
                          "built-but-previously-unused CapsGRU) using a SliceSequenceDataset instead "
                          "of SliceCacheDataset -- refines the classification call using --seq-len "
                          "neighbouring slices' capsule vectors, matching the paper's described "
                          "inter-slice-consistency mechanism. Segmentation stays per-slice (the "
                          "center slice of the window). Only meaningful with --seg-first-cascade.")
    ap.add_argument("--seq-len", type=int, default=3,
                     help="Number of neighbouring slices per training sequence when --capsgru is set. Must be odd.")
    ap.add_argument("--gate-bypass-prob", type=float, default=0.0,
                     help="stochastic gate bypass (seg-first cascade only): per-sample probability, "
                          "during training, of zeroing the lesion prior so the gate is the identity "
                          "for that sample. Keeps the classifier competent on ungated features, the "
                          "regime it faces whenever the decoder misses a lesion at inference. "
                          "Dropout/stochastic-depth-family regularisation; eval is never bypassed.")
    ap.add_argument("--positive-extra-aug", action="store_true",
                     help="Approach B v2 lever: apply intensity_jitter (brightness/contrast) on top "
                          "of the usual spatial augmentation, but only for lesion-positive (rank>1) "
                          "training samples, to give the minority class more diverse synthetic views "
                          "instead of just seeing the same few positive images more often via "
                          "--balanced resampling.")
    args = ap.parse_args()
    if args.gated_cascade and args.seg_first_cascade:
        raise ValueError("--gated-cascade and --seg-first-cascade are two different cascade "
                          "directions (Approach A vs Approach B) -- run them as separate experiments, "
                          "not combined in one run.")

    if args.capsgru and not args.seg_first_cascade:
        raise ValueError("--capsgru only makes sense with --seg-first-cascade (Approach B) -- "
                          "forward_sequence (the non-seg-first baseline path) exists but this "
                          "experiment is specifically about combining CapsGRU with the seg-first cascade.")

    if args.gate_bypass_prob > 0.0 and not (args.seg_first_cascade or args.gated_cascade):
        raise ValueError("--gate-bypass-prob requires --seg-first-cascade (forward-gate bypass) "
                          "or --gated-cascade (loss-gate bypass)")
    if args.gate_bypass_prob > 0.0 and args.capsgru:
        raise ValueError("--gate-bypass-prob is not implemented for the --capsgru sequence path")
    if not 0.0 <= args.gate_bypass_prob < 1.0:
        raise ValueError("--gate-bypass-prob must be in [0, 1)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.capsgru:
        train_ds = SliceSequenceDataset(args.data_root, split="train", seq_len=args.seq_len,
                                         augment=True, positive_extra_aug=args.positive_extra_aug)
        val_ds = SliceSequenceDataset(args.data_root, split="valid", seq_len=args.seq_len, augment=False)
    else:
        train_ds = SliceCacheDataset(args.data_root, split="train", augment=True,
                                      positive_extra_aug=args.positive_extra_aug)
        val_ds = SliceCacheDataset(args.data_root, split="valid", augment=False)

    # PERFORMANCE FIX (post-hoc audit): the CV run was I/O-bound on network
    # storage, not GPU- or CPU-bound (workers sat at ~10% CPU each while the
    # GPU idled). persistent_workers=True avoids re-forking + re-importing
    # all worker processes at the start of every single epoch (the default,
    # since num_workers>0 with persistent_workers unset tears them down each
    # time the DataLoader is iterated); prefetch_factor gives each worker
    # more read-ahead depth to better hide per-file network latency;
    # pin_memory speeds up the host->GPU transfer. num_workers default was
    # raised 4->8 in this same audit after confirming CPU had no headroom
    # pressure (255 cores, workers under 12% each even before this change).
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    if args.balanced:
        weight_files = train_ds.center_files() if args.capsgru else train_ds.files
        weights, counts = build_sample_weights(args.data_root, "train", weight_files)
        print(f"balanced sampling enabled, train rank counts: {counts}", flush=True)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   drop_last=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = MiniSegCaps().to(device)
    criterion = CombinedLoss(gate_seg_on_gt=args.gated_cascade, gate_min_weight=args.gate_min_weight,
                              uncertainty_weighted=args.uncertainty_weighted,
                              gate_bypass_prob=args.gate_bypass_prob if args.gated_cascade else 0.0).to(device)
    if args.gated_cascade:
        print(f"gated-cascade experiment enabled: segmentation loss gated on ground-truth csPCa "
              f"label, gate_min_weight={args.gate_min_weight}, "
              f"gate_bypass_prob={args.gate_bypass_prob}", flush=True)
    if args.seg_first_cascade:
        print("seg-first-cascade experiment enabled: classification gated on the model's own predicted mask", flush=True)
    if args.uncertainty_weighted:
        print("uncertainty-weighted loss enabled: learning the seg/classification loss balance", flush=True)
    if args.capsgru:
        print(f"capsgru experiment enabled: seq_len={args.seq_len}, gate_floor={args.gate_floor}, "
              f"gate_ceil={args.gate_ceil}", flush=True)
    if args.positive_extra_aug:
        print("positive-extra-aug enabled: intensity jitter on lesion-positive training samples", flush=True)
    # criterion.parameters() is empty unless --uncertainty-weighted added
    # log_var_seg/log_var_cls -- including it here is always safe and is
    # what lets those two scalars actually be learned when they exist.
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every,
                                                 gamma=args.lr_decay)

    epoch_fn = run_epoch_sequence if args.capsgru else run_epoch
    epoch_kwargs = dict(max_steps=args.max_steps_per_epoch, gate_floor=args.gate_floor, gate_ceil=args.gate_ceil)
    if not args.capsgru:
        epoch_kwargs["seg_first"] = args.seg_first_cascade
        epoch_kwargs["gate_bypass_prob"] = args.gate_bypass_prob

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = epoch_fn(model, train_loader, criterion, optimizer, device, train=True, **epoch_kwargs)
        val_metrics = epoch_fn(model, val_loader, criterion, optimizer, device, train=False, **epoch_kwargs)
        scheduler.step()
        dt = time.time() - t0

        uw_str = ""
        if args.uncertainty_weighted:
            uw_str = f" log_var_seg={val_metrics.get('log_var_seg', 0):.3f} log_var_cls={val_metrics.get('log_var_cls', 0):.3f}"
        print(f"epoch {epoch}/{args.epochs} lr={scheduler.get_last_lr()[0]:.5f} "
              f"train_total={train_metrics['total']:.4f} val_total={val_metrics['total']:.4f} "
              f"val_dice_loss={val_metrics['dice']:.4f}{uw_str} ({dt:.1f}s)", flush=True)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({"model": model.state_dict(), "criterion": criterion.state_dict(),
                        "epoch": epoch, "val_total": best_val},
                       os.path.join(args.checkpoint_dir, "best.pt"))

        torch.save({"model": model.state_dict(), "criterion": criterion.state_dict(), "epoch": epoch},
                   os.path.join(args.checkpoint_dir, "last.pt"))


if __name__ == "__main__":
    main()
