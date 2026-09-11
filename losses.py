"""Combined loss: Dice(0.5) + BCE(0.5) for segmentation, capsule margin loss
for the ordinal PI-RADS bits, and a masked-MSE reconstruction loss (weight
5e-4), matching the four loss terms described in the paper."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0, sample_weight=None):
    """sample_weight: optional (B,) tensor in [0,1] -- used by the cascade
    experiment's gate_seg_on_gt option to zero out the segmentation loss for
    slices the ground truth says are benign, so the decoder's gradient
    focuses on the ~24% of slices that are actually csPCa-positive instead
    of learning to predict an empty mask for the majority class."""
    prob = torch.sigmoid(logits)
    prob = prob.flatten(1)
    target = target.flatten(1)
    intersection = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    per_sample = 1 - dice
    if sample_weight is not None:
        denom = sample_weight.sum().clamp_min(1.0)
        return (per_sample * sample_weight).sum() / denom
    return per_sample.mean()


def margin_loss(lengths, targets, m_pos=0.9, m_neg=0.1, lam=0.5):
    """Multi-label margin loss (one term per ordinal bit), Sabour et al. 2017."""
    pos = targets * F.relu(m_pos - lengths) ** 2
    neg = (1 - targets) * lam * F.relu(lengths - m_neg) ** 2
    return (pos + neg).sum(dim=-1).mean()


def bce_loss(logits, target, sample_weight=None):
    """Per-sample-weighted counterpart to F.binary_cross_entropy_with_logits'
    default mean reduction, for the same gate_seg_on_gt use as dice_loss."""
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per_sample = per_pixel.flatten(1).mean(dim=1)
    if sample_weight is not None:
        denom = sample_weight.sum().clamp_min(1.0)
        return (per_sample * sample_weight).sum() / denom
    return per_sample.mean()


def masked_mse_loss(recon, target, mask):
    """mask broadcasts over channels; typically the (dilated) lesion mask so
    reconstruction is only scored where it matters."""
    diff2 = (recon - target) ** 2
    masked = diff2 * mask
    denom = mask.sum().clamp_min(1.0) * recon.shape[1] / mask.shape[1]
    return masked.sum() / denom


class CombinedLoss(nn.Module):
    """gate_seg_on_gt (classify-then-segment cascade experiment, opt-in,
    default off so the original joint-training behavior this model was
    verified against, mean Dice 0.863 across 5 folds, stays reproducible):
    when True, the segmentation loss for each sample is weighted by the
    GROUND-TRUTH csPCa indicator (pirads_bits[:, CSPCA_BIT]) rather than
    averaged over every slice uniformly. This is the "gate on ground truth
    during training" half of the cascade design -- it concentrates the
    decoder's gradient on the ~24% of slices that are actually csPCa-
    positive, instead of spending capacity learning to predict an empty mask
    for the majority-benign slices. It is independent of the model-side
    soft gate (gated_seg_prob / cspca_gate in minisegcaps_model.py), which
    always runs regardless of this flag and is not used as a loss target
    here specifically so the decoder's own gradient is never blocked by the
    classifier's (possibly still-learning) confidence.

    gate_min_weight (post-hoc fix, v2 of the cascade experiment): the first
    gated-cascade run used a hard {0,1} sample_weight (gate_min_weight=0.0),
    which zeroed segmentation loss entirely on GT-benign slices -- the
    decoder never learned to predict "empty" there, and raw seg_logits Dice
    collapsed to 0.33-0.54 across folds (vs. baseline 0.86) because it
    over-predicted lesions everywhere on benign slices. gate_min_weight>0
    gives benign slices a small but nonzero floor weight instead of a hard
    zero, so the decoder still receives a real "suppress false positives
    here" signal while still emphasizing positive slices ~1/gate_min_weight
    as strongly. Default 0.0 preserves the exact v1 behavior.

    uncertainty_weighted (v2, lever #3 -- reduce shared-encoder contention):
    Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics" (CVPR 2018). A fixed 1:1 weight
    between the segmentation loss and the margin (classification) loss lets
    whichever task's gradient happens to be larger dominate the shared
    6x6x256 bottleneck at any given point in training, in a way that can
    distort the features the other task needs. Instead of a hand-picked
    fixed ratio, this learns one scalar log-variance per task
    (log_var_seg, log_var_cls) jointly with the model: each task loss is
    weighted by exp(-log_var)/1 with a +log_var regularizer, so the model
    itself discovers the relative weighting each task's inherent difficulty
    calls for, and the regularizer stops the trivial collapse of just
    shrinking a task's weight to zero. Reconstruction stays fixed-weight
    (5e-4) -- it's a minor regularizer, not one of the two primary tasks in
    tension. train.py must include this module's own parameters in the
    optimizer (list(model.parameters()) + list(criterion.parameters())) for
    log_var_seg/log_var_cls to actually be learned."""

    def __init__(self, dice_w=0.5, bce_w=0.5, margin_w=1.0, recon_w=5e-4,
                 gate_seg_on_gt=False, gate_min_weight=0.0, uncertainty_weighted=False,
                 gate_bypass_prob=0.0):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.margin_w = margin_w
        self.recon_w = recon_w
        self.gate_seg_on_gt = gate_seg_on_gt
        self.gate_min_weight = gate_min_weight
        self.uncertainty_weighted = uncertainty_weighted
        # Stochastic loss-gate bypass (Approach A analogue of the seg-first
        # forward bypass): with probability gate_bypass_prob per sample,
        # during training only (self.training), the GT loss gate is bypassed
        # and the sample's segmentation loss gets full weight regardless of
        # its csPCa label. E[benign weight] = p, so this is the stochastic
        # counterpart of a deterministic gate_min_weight floor of the same
        # value -- run WITHOUT gate_min_weight to isolate the floor lever.
        # train.py must call criterion.train(train) alongside model.train()
        # so validation losses stay deterministic.
        self.gate_bypass_prob = gate_bypass_prob
        if uncertainty_weighted:
            self.log_var_seg = nn.Parameter(torch.zeros(()))
            self.log_var_cls = nn.Parameter(torch.zeros(()))

    def forward(self, outputs, seg_target, pirads_bits, recon_target, recon_mask):
        from minisegcaps_model import CSPCA_BIT
        sample_weight = None
        if self.gate_seg_on_gt:
            gt = pirads_bits[:, CSPCA_BIT]
            sample_weight = self.gate_min_weight + (1.0 - self.gate_min_weight) * gt
            if self.gate_bypass_prob > 0.0 and self.training:
                bypass = (torch.rand_like(gt) < self.gate_bypass_prob).float()
                sample_weight = torch.maximum(sample_weight, bypass)
        l_dice = dice_loss(outputs["seg_logits"], seg_target, sample_weight=sample_weight)
        l_bce = bce_loss(outputs["seg_logits"], seg_target, sample_weight=sample_weight)
        l_seg = self.dice_w * l_dice + self.bce_w * l_bce

        l_margin = margin_loss(outputs["pirads_prob"], pirads_bits)

        l_recon = masked_mse_loss(outputs["recon"], recon_target, recon_mask)

        extra = {}
        if self.uncertainty_weighted:
            # STABILITY FIX (post-hoc, found via a live-run intervention: the
            # first attempt's train_total went -0.32 -> -0.63 -> -0.94 over
            # epochs 7/12/18, an accelerating negative trend with no
            # corresponding task-quality improvement -- the textbook runaway
            # failure mode of unclamped Kendall et al. uncertainty weighting,
            # where log_var can drift to increasingly extreme values because
            # the +log_var regularizer alone doesn't bound it in practice.
            # Clamping to [-3, 3] (precision exp(-log_var) in [~0.05, ~20])
            # is the standard mitigation and keeps both task weights in a
            # sane range without preventing the mechanism from doing its job
            # (discovering a non-1:1 relative weighting).
            log_var_seg = self.log_var_seg.clamp(-3.0, 3.0)
            log_var_cls = self.log_var_cls.clamp(-3.0, 3.0)
            weighted_seg = torch.exp(-log_var_seg) * l_seg + log_var_seg
            weighted_cls = torch.exp(-log_var_cls) * l_margin + log_var_cls
            total = weighted_seg + weighted_cls + self.recon_w * l_recon
            extra = {"log_var_seg": log_var_seg.item(), "log_var_cls": log_var_cls.item()}
        else:
            total = l_seg + self.margin_w * l_margin + self.recon_w * l_recon

        return total, {
            "dice": l_dice.item(),
            "bce": l_bce.item(),
            "margin": l_margin.item(),
            "recon": l_recon.item(),
            "total": total.item(),
            **extra,
        }
