"""Grad-CAM for MiniSegCaps' capsule classification head.

Standard Grad-CAM (Selvaraju et al., 2017) is defined for a scalar class
score flowing from a convolutional feature map. This architecture's
classification head is a capsule, not a scalar logit, so the adaptation used
here follows the natural generalization used in the capsule-network
explainability literature: capsule *activation length* (in [0,1], the same
role as a softmax/sigmoid class probability) stands in for the class score.

Target layer: the shared 6x6x256 encoder bottleneck (see minisegcaps_model.py
Encoder.to_bottleneck) -- the feature map that directly feeds the capsule
branch, and the same "last conv layer before the classifier" role Grad-CAM
conventionally hooks into a CNN.

Recipe, per sample:
  1. Forward through the encoder to get the bottleneck feature map (B,256,6,6).
  2. Continue forward through the capsule branch to the target ordinal bit's
     activation length (scalar per sample).
  3. Backprop that scalar to the bottleneck; global-average-pool the gradient
     over space to get one importance weight per channel.
  4. CAM = ReLU(sum_k weight_k * bottleneck_k), upsampled to patch resolution.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from capsule_layers import CapsuleLength
from minisegcaps_model import MiniSegCaps

# bit index into the 4-dim ordinal vector for each ">=k" rank threshold
BIT_FOR_RANK = {2: 0, 3: 1, 4: 2, 5: 3}


class GradCAM:
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, x, rank_threshold=3, seg_first=False, gate_floor=1.0, gate_ceil=2.0):
        """x: (B,C,H,W) input patch batch. Returns cam (B,H,W) in [0,1] and
        the model's pirads_prob (B,4) for the same forward pass.

        seg_first=True targets Approach B checkpoints: the bottleneck this
        CAM attributes to is still the shared encoder output (retain_grad is
        on the pre-gate bottleneck), but the forward path in between runs
        through forward_seg_first's decoder-then-gate wiring instead of
        forward_single's, so the gradient correctly reflects what that
        checkpoint's classifier actually saw (the lesion-prior-gated
        features), not the ungated ones."""
        x = x.to(self.device).requires_grad_(False)
        bit_idx = BIT_FOR_RANK[rank_threshold]

        bottleneck, skips = self.model.encoder(x)
        bottleneck.retain_grad()

        if seg_first:
            b, c, h, w = bottleneck.shape
            neutral_attn = torch.ones(b, 1, h, w, device=x.device, dtype=x.dtype)
            seg_logits = self.model.decoder(bottleneck, skips, neutral_attn)
            lesion_prior = F.interpolate(torch.sigmoid(seg_logits), size=(h, w),
                                          mode="bilinear", align_corners=False)
            gated_bottleneck = bottleneck * (gate_floor + (gate_ceil - gate_floor) * lesion_prior)
            ordinal_caps, attn = self.model.caps_branch(gated_bottleneck)
        else:
            ordinal_caps, attn = self.model.caps_branch(bottleneck)
        pirads_prob = CapsuleLength()(ordinal_caps)  # (B,4)

        target = pirads_prob[:, bit_idx].sum()
        self.model.zero_grad(set_to_none=True)
        target.backward()

        grad = bottleneck.grad  # (B,256,6,6)
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (B,256,1,1)
        cam = F.relu((weights * bottleneck.detach()).sum(dim=1))  # (B,6,6)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1)  # (B,H,W)

        b = cam.shape[0]
        cam_flat = cam.view(b, -1)
        cmin = cam_flat.min(dim=1, keepdim=True)[0]
        cmax = cam_flat.max(dim=1, keepdim=True)[0]
        cam_norm = ((cam_flat - cmin) / (cmax - cmin + 1e-8)).view_as(cam)

        return cam_norm.detach().cpu().numpy(), pirads_prob.detach().cpu().numpy()


def overlap_metrics(cam, gt_mask, cam_threshold=0.5):
    """IoU and Dice between the thresholded CAM and the ground-truth lesion
    mask, for a single (H,W) pair."""
    pred = (cam >= cam_threshold).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    inter = (pred * gt).sum()
    union = ((pred + gt) > 0).sum()
    iou = inter / union if union > 0 else np.nan
    dice_denom = pred.sum() + gt.sum()
    dice = (2 * inter) / dice_denom if dice_denom > 0 else np.nan
    return float(iou), float(dice)
