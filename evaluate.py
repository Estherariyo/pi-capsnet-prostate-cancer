"""Evaluation: Dice for segmentation; accuracy/sensitivity/specificity/F1 for
the PI-RADS>=4 (clinically significant) call, at slice- and patient-level,
matching the metrics reported in the paper."""
import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from minisegcaps_model import MiniSegCaps
from dataset import SliceCacheDataset
from dataset_sequence import SliceSequenceDataset
from utils import decode_pirads_ordinal


def dice_score(pred_mask, gt_mask, eps=1e-6):
    pred = pred_mask.flatten(1)
    gt = gt_mask.flatten(1)
    inter = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1)
    return ((2 * inter + eps) / (union + eps)).cpu().tolist()


def classification_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / max(1, len(y_true))
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2 * ppv * sens / max(1e-8, ppv + sens) if (ppv + sens) > 0 else 0.0
    return dict(accuracy=acc, sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1,
                n=len(y_true), tp=tp, tn=tn, fp=fp, fn=fn)


@torch.no_grad()
def evaluate(model, loader, device, seg_first=False, capsgru=False, gate_floor=1.0, gate_ceil=2.0,
             dual_pass=False):
    model.eval()
    dices = []
    gated_dices = []
    # bit index into the 4-dim ordinal vector for each ">=k" threshold: bit i is ">=(i+2)"
    thresholds = {"csPCa_ISUP>=2 (rank>=3)": (3, 1), "high_grade (rank>=4)": (4, 2)}
    slice_true = {k: [] for k in thresholds}
    slice_pred = {k: [] for k in thresholds}
    patient_gt = {k: {} for k in thresholds}
    patient_pred_maxprob = {k: defaultdict(float) for k in thresholds}

    for batch in loader:
        if capsgru:
            x_seq = batch["image_seq"].to(device)
            seg_gt = batch["mask_center"].to(device)
            pirads_bits_gt = batch["pirads_bits"].to(device)
            patient_ids = batch["patient_id"]
            full = model.forward_sequence_seg_first(x_seq, gate_floor=gate_floor, gate_ceil=gate_ceil)
            center = x_seq.shape[1] // 2
            out = {
                "seg_logits": full["seg_logits"][:, center],
                "pirads_prob": full["pirads_prob_refined"][:, center],
            }
        else:
            x = batch["image"].to(device)
            seg_gt = batch["mask"].to(device)
            pirads_bits_gt = batch["pirads_bits"].to(device)
            patient_ids = batch["patient_id"]
            if seg_first:
                out = model.forward_seg_first(x, gate_floor=gate_floor, gate_ceil=gate_ceil)
                if dual_pass:
                    # second, ungated pass: gate_ceil == gate_floor makes the gate a
                    # constant (identity at 1.0). Per-bit max of the two probability
                    # vectors -- a two-view test-time ensemble that is only
                    # legitimate for bypass-trained checkpoints, where the ungated
                    # regime was part of the training distribution. Segmentation
                    # output stays that of the gated pass (identical decoder either
                    # way: the gate sits after the decoder).
                    out_ungated = model.forward_seg_first(x, gate_floor=1.0, gate_ceil=1.0)
                    out = dict(out)
                    out["pirads_prob"] = torch.maximum(out["pirads_prob"], out_ungated["pirads_prob"])
            else:
                out = model.forward_single(x)
        pred_mask = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
        dices.extend(dice_score(pred_mask, seg_gt))

        # gated_seg_prob is always computed (see minisegcaps_model.py), even
        # for checkpoints trained without --gated-cascade -- reporting it here
        # for every run lets the cascade experiment's checkpoints be compared
        # against the baseline's checkpoints on the same metric.
        if "gated_seg_prob" in out:
            gated_pred_mask = (out["gated_seg_prob"] > 0.5).float()
            gated_dices.extend(dice_score(gated_pred_mask, seg_gt))

        gt_rank = decode_pirads_ordinal(pirads_bits_gt)
        pred_rank = decode_pirads_ordinal(out["pirads_prob"])

        for key, (rank_cut, bit_idx) in thresholds.items():
            gt_sig = (gt_rank >= rank_cut).long().cpu().tolist()
            pred_sig = (pred_rank >= rank_cut).long().cpu().tolist()
            pred_prob_sig = out["pirads_prob"][:, bit_idx].cpu().tolist()

            slice_true[key].extend(gt_sig)
            slice_pred[key].extend(pred_sig)
            for pid, gt, prob in zip(patient_ids, gt_sig, pred_prob_sig):
                patient_gt[key][pid] = max(patient_gt[key].get(pid, 0), gt)
                patient_pred_maxprob[key][pid] = max(patient_pred_maxprob[key][pid], prob)

    results = {"dice_mean": sum(dices) / max(1, len(dices))}
    if gated_dices:
        results["dice_mean_gated"] = sum(gated_dices) / max(1, len(gated_dices))
    for key in thresholds:
        patient_true_list = list(patient_gt[key].values())
        patient_pred_list = [1 if patient_pred_maxprob[key][pid] > 0.5 else 0 for pid in patient_gt[key]]
        results[key] = {
            "slice_level": classification_metrics(slice_true[key], slice_pred[key]),
            "patient_level": classification_metrics(patient_true_list, patient_pred_list),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/minisegcaps/data/picai_cache")
    ap.add_argument("--checkpoint", default="/workspace/minisegcaps/checkpoints/best.pt")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seg-first-cascade", action="store_true",
                     help="evaluate via MiniSegCaps.forward_seg_first (Approach B checkpoints)")
    ap.add_argument("--capsgru", action="store_true",
                     help="evaluate via MiniSegCaps.forward_sequence_seg_first with a "
                          "SliceSequenceDataset (Approach B v2 CapsGRU checkpoints)")
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--gate-floor", type=float, default=1.0)
    ap.add_argument("--gate-ceil", type=float, default=2.0)
    ap.add_argument("--dual-pass", action="store_true",
                     help="seg-first only: also run an ungated forward pass and take the per-bit "
                          "max of the two ordinal probability vectors (two-view test-time "
                          "ensemble). Only meaningful for checkpoints trained with "
                          "--gate-bypass-prob > 0; results are reported as a separate JSON "
                          "section alongside the standard single-pass metrics.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniSegCaps().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    if args.capsgru:
        ds = SliceSequenceDataset(args.data_root, split=args.split, seq_len=args.seq_len, augment=False)
    else:
        ds = SliceCacheDataset(args.data_root, split=args.split, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = evaluate(model, loader, device, seg_first=args.seg_first_cascade, capsgru=args.capsgru,
                        gate_floor=args.gate_floor, gate_ceil=args.gate_ceil)
    if args.dual_pass:
        if not args.seg_first_cascade or args.capsgru:
            raise ValueError("--dual-pass requires --seg-first-cascade (without --capsgru)")
        dual = evaluate(model, loader, device, seg_first=True, capsgru=False,
                         gate_floor=args.gate_floor, gate_ceil=args.gate_ceil, dual_pass=True)
        results = {"single_pass": results, "dual_pass": dual}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
