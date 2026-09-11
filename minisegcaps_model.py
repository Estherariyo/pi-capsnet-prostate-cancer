"""MiniSegCaps: joint prostate lesion segmentation + PI-RADS classification.

Reconstruction of the architecture described in Jiang et al., "Joint Cancer
Segmentation and PI-RADS Classification on Multiparametric MRI Using
MiniSegCaps Network", Diagnostics 2023, 13(4), 615.

Three parts, matching the paper's description:
  1. MiniSeg backbone: lightweight conv encoder-decoder -> lesion mask.
  2. Capsule predictive branch: two conv-capsule layers on the encoder
     bottleneck + 3 FC layers + a final capsule layer producing 4 ordinal
     PI-RADS capsules, plus a reconstruction decoder for the masked-MSE loss.
  3. CapsGRU: a GRU run across neighbouring slices' capsule vectors for
     through-plane consistency.

No official code was released with the paper; this is a faithful from-scratch
implementation of the described components, not a copy of the authors' code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from capsule_layers import PrimaryCapsuleConv, ConvCapsuleLayer, CapsuleLength, squash

BOTTLENECK_HW = 6
BOTTLENECK_C = 256

# Index into the 4-bit ordinal vector for the csPCa (ISUP>=2, i.e. rank>=3)
# threshold -- see utils.py's [>=2,>=3,>=4,>=5] cumulative encoding. This is
# the bit the classify-then-segment cascade gate reads.
CSPCA_BIT = 1


def conv_block(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """4 downsampling stages, then adaptive-pool to the 6x6x256 bottleneck
    size the paper reports for the capsule branch input."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.stage1 = conv_block(in_channels, 32, stride=1)
        self.down1 = conv_block(32, 32, stride=2)
        self.stage2 = conv_block(32, 64, stride=1)
        self.down2 = conv_block(64, 64, stride=2)
        self.stage3 = conv_block(64, 128, stride=1)
        self.down3 = conv_block(128, 128, stride=2)
        self.stage4 = conv_block(128, 256, stride=1)
        self.down4 = conv_block(256, 256, stride=2)
        self.to_bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((BOTTLENECK_HW, BOTTLENECK_HW)),
            nn.Conv2d(256, BOTTLENECK_C, 1),
            nn.BatchNorm2d(BOTTLENECK_C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        s4 = self.stage4(self.down3(s3))
        bottleneck = self.to_bottleneck(self.down4(s4))
        skips = (s1, s2, s3, s4)
        return bottleneck, skips


class Decoder(nn.Module):
    """Upsamples the bottleneck back to full resolution, fusing encoder skips
    and a capsule-derived attention map at each stage, and predicts the
    lesion mask logits."""

    def __init__(self, out_channels=1):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(BOTTLENECK_C, 256, 2, stride=2)
        self.fuse4 = conv_block(256 + 256 + 1, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.fuse3 = conv_block(128 + 128 + 1, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.fuse2 = conv_block(64 + 64 + 1, 32)
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.fuse1 = conv_block(32 + 32 + 1, 32)
        self.head = nn.Conv2d(32, out_channels, 1)

    @staticmethod
    def _resize_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, bottleneck, skips, attn_map):
        s1, s2, s3, s4 = skips
        a4 = self._resize_to(attn_map, s4)
        x = self._resize_to(self.up4(bottleneck), s4)
        x = self.fuse4(torch.cat([x, s4, a4], dim=1))

        a3 = self._resize_to(attn_map, s3)
        x = self._resize_to(self.up3(x), s3)
        x = self.fuse3(torch.cat([x, s3, a3], dim=1))

        a2 = self._resize_to(attn_map, s2)
        x = self._resize_to(self.up2(x), s2)
        x = self.fuse2(torch.cat([x, s2, a2], dim=1))

        a1 = self._resize_to(attn_map, s1)
        x = self._resize_to(self.up1(x), s1)
        x = self.fuse1(torch.cat([x, s1, a1], dim=1))

        return self.head(x)


class CapsuleBranch(nn.Module):
    """Two conv-capsule layers on the bottleneck feature map, followed by
    3 FC layers + a final conv-capsule layer producing 4 PI-RADS ordinal
    capsules, plus a reconstruction decoder used for the masked-MSE loss."""

    NUM_ORDINAL = 4  # thresholds for PI-RADS >=2, >=3, >=4, >=5
    CAP_DIM = 8

    def __init__(self, recon_out_channels=3, recon_hw=100):
        super().__init__()
        self.recon_hw = recon_hw
        self.primary = PrimaryCapsuleConv(BOTTLENECK_C, num_caps=8, cap_dim=8,
                                           kernel_size=3, stride=1, padding=1)
        self.conv_caps1 = ConvCapsuleLayer(in_caps=8, in_dim=8, out_caps=8, out_dim=8,
                                            kernel_size=3, stride=1, padding=1, routing_iters=3)
        self.conv_caps2 = ConvCapsuleLayer(in_caps=8, in_dim=8, out_caps=8, out_dim=8,
                                            kernel_size=3, stride=2, padding=1, routing_iters=3)

        pooled_hw = BOTTLENECK_HW // 2  # from conv_caps2 stride=2
        flat_dim = pooled_hw * pooled_hw * 8 * self.CAP_DIM
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.NUM_ORDINAL * self.CAP_DIM),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.NUM_ORDINAL * self.CAP_DIM, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Linear(512, recon_out_channels * recon_hw * recon_hw), nn.Sigmoid(),
        )
        self.recon_out_channels = recon_out_channels

    def forward(self, bottleneck):
        # bottleneck: (B, 256, 6, 6) -> capsule grid (B, H, W, caps, dim)
        b = bottleneck.shape[0]
        primary = self.primary(bottleneck)  # (B, N, dim) flattened grid already
        h = w = bottleneck.shape[-1]
        primary = primary.view(b, h, w, 8, 8)

        caps1 = self.conv_caps1(primary)
        caps2 = self.conv_caps2(caps1)  # (B, h/2, w/2, 8, 8)

        flat = caps2.reshape(b, -1)
        ordinal_caps = self.fc(flat).view(b, self.NUM_ORDINAL, self.CAP_DIM)
        ordinal_caps = squash(ordinal_caps, dim=-1)

        # attention map: spatial activation strength from conv_caps1, upsampled
        # back to bottleneck resolution and used to steer the segmentation decoder
        attn = CapsuleLength()(caps1).mean(dim=-1, keepdim=True)  # (B,h,w,1)
        attn = attn.permute(0, 3, 1, 2)  # (B,1,h,w)

        return ordinal_caps, attn

    def reconstruct(self, ordinal_caps):
        b = ordinal_caps.shape[0]
        recon = self.decoder(ordinal_caps.reshape(b, -1))
        return recon.view(b, self.recon_out_channels, self.recon_hw, self.recon_hw)


class CapsGRU(nn.Module):
    """GRU across a sequence of neighbouring slices' ordinal-capsule vectors,
    enforcing through-plane (inter-slice) consistency of the PI-RADS call."""

    def __init__(self, num_ordinal=4, cap_dim=8, hidden=128):
        super().__init__()
        in_dim = num_ordinal * cap_dim
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, in_dim)
        self.num_ordinal = num_ordinal
        self.cap_dim = cap_dim

    def forward(self, caps_seq):
        # caps_seq: (B, T, num_ordinal, cap_dim) — T neighbouring slices
        b, t = caps_seq.shape[:2]
        flat = caps_seq.reshape(b, t, -1)
        out, _ = self.gru(flat)
        refined = self.proj(out).view(b, t, self.num_ordinal, self.cap_dim)
        return squash(refined, dim=-1)


class MiniSegCaps(nn.Module):
    def __init__(self, in_channels=3, patch_hw=100):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels=1)
        self.caps_branch = CapsuleBranch(recon_out_channels=in_channels, recon_hw=patch_hw)
        self.caps_gru = CapsGRU(num_ordinal=CapsuleBranch.NUM_ORDINAL, cap_dim=CapsuleBranch.CAP_DIM)

    def forward_single(self, x):
        """x: (B, C, H, W) single-slice patch.

        CASCADE GATE (classify-then-segment experiment): seg_logits is the
        segmentation decoder's raw, ungated output -- training losses read
        this field unchanged, so the decoder's own gradient signal is exactly
        what it was before this experiment. gated_seg_prob is a NEW field:
        sigmoid(seg_logits) multiplied by the classifier's own predicted
        csPCa confidence (pirads_prob[:, CSPCA_BIT]), broadcast spatially.
        This is a *soft* gate -- plain multiplication of two continuous,
        differentiable values -- so gradients flow through it in both
        directions (decoder <-> classifier) rather than being blocked by a
        hard if/else. It is intended as the reported/deployed mask (a
        genuinely-benign classification suppresses the mask smoothly) and as
        the hook for an inference-time hard threshold (see
        predict_with_hard_gate below); it deliberately does not replace
        seg_logits as the segmentation-loss target so this cascade experiment
        cannot silently degrade the existing, verified segmentation training
        signal -- see CombinedLoss's gate_seg_on_gt option for the paired
        change on the loss side.
        """
        bottleneck, skips = self.encoder(x)
        ordinal_caps, attn = self.caps_branch(bottleneck)
        seg_logits = self.decoder(bottleneck, skips, attn)
        recon = self.caps_branch.reconstruct(ordinal_caps)
        pirads_prob = CapsuleLength()(ordinal_caps)  # (B, 4) in [0,1]

        cspca_gate = pirads_prob[:, CSPCA_BIT].view(-1, 1, 1, 1)  # (B,1,1,1)
        gated_seg_prob = torch.sigmoid(seg_logits) * cspca_gate

        return {
            "seg_logits": seg_logits,
            "ordinal_caps": ordinal_caps,
            "pirads_prob": pirads_prob,
            "recon": recon,
            "cspca_gate": pirads_prob[:, CSPCA_BIT],
            "gated_seg_prob": gated_seg_prob,
        }

    @torch.no_grad()
    def predict_with_hard_gate(self, x, threshold=0.5):
        """Inference-only helper: hard-thresholds the classifier's csPCa
        confidence and reports an all-zero mask for slices below threshold,
        instead of the soft (always-computed) gate used in forward_single.
        Not used during training -- no gradients are needed here, so a hard
        if/else is safe (see the differentiability discussion this cascade
        experiment is built around: hard gating is fine once no backward
        pass will run through it). This is where the "real" compute-saving
        cascade behavior belongs; forward_single stays fully differentiable
        and soft for training."""
        out = self.forward_single(x)
        is_cspca = (out["cspca_gate"] > threshold).view(-1, 1, 1, 1).float()
        out["hard_gated_seg_prob"] = torch.sigmoid(out["seg_logits"]) * is_cspca
        return out

    def forward_seg_first(self, x, gate_floor=1.0, gate_ceil=2.0, bypass_prob=0.0):
        """Approach B: segment-then-classify cascade (the reverse direction
        from Approach A). Literature grounding (chat + Approach B report):
        a two-stage prostate segmentation-then-PIRADS-grading strategy has
        direct precedent, and the mechanism the literature actually uses is
        attention-gating a *predicted* mask onto classifier features ("
        localized lesion attention"), not concatenating a raw mask or -- most
        importantly -- ever gating on the ground-truth mask, which risks the
        classifier shortcut-learning from mask shape/pixel-count instead of
        genuine visual features (a documented failure mode for this cascade
        direction specifically, unlike Approach A's ground-truth loss gate,
        which never exposes the ground truth to the model itself).

        Decoder runs on the bottleneck FIRST, using a neutral (all-ones)
        attention map instead of one derived from the capsule branch, so
        segmentation here genuinely does not depend on classification (in
        forward_single, some coupling already exists via the capsule-derived
        attn map; this path removes it to make the segment-first ordering
        real, not just nominal). The decoder's own predicted probability map
        is then downsampled to bottleneck resolution and used to multiply
        (gate) the bottleneck features before they reach the capsule branch
        -- soft and differentiable, and using the model's own prediction,
        never ground truth.

        gate_floor/gate_ceil (v2, gate-strength tuning): the gate multiplier
        is gate_floor + (gate_ceil - gate_floor) * lesion_prior. v1 used the
        implicit defaults (1.0, 2.0) -- pure amplification of lesion regions,
        never suppressing background (floor=1.0 means non-lesion regions
        always pass through at full, unmodified strength). Since Dice stayed
        strong in v1 while classification accuracy/specificity lagged, the
        decoder isn't obviously the bottleneck -- a sharper gate (e.g.
        gate_floor<1.0) lets the classifier partially suppress non-lesion
        background instead of always keeping full context, which v1 never
        tested. Defaults reproduce v1's exact behavior.

        bypass_prob (stochastic gate bypass): during training only, each
        sample in the batch independently has this probability of its lesion
        prior being zeroed, which (with gate_floor=1.0) reduces the gate to
        the identity for that sample. The classifier is thereby forced to
        stay competent on UNGATED bottleneck features too -- the regime it
        faces at inference whenever the decoder misses a lesion, which is
        exactly the failure mode observed on small lesions (csPCa prob
        collapses to ~0 because unamplified positives are outside the
        classifier's training distribution). Same regularisation family as
        dropout / stochastic depth / modality dropout: randomly withhold a
        signal the downstream head would otherwise over-adapt to. Inference
        (model.eval()) is never bypassed, so evaluation stays deterministic
        and fully gated; the bypass uses the model's own prediction or
        nothing, never ground truth, preserving the leakage-free design."""
        bottleneck, skips = self.encoder(x)
        b, c, h, w = bottleneck.shape
        neutral_attn = torch.ones(b, 1, h, w, device=x.device, dtype=x.dtype)

        seg_logits = self.decoder(bottleneck, skips, neutral_attn)
        seg_prob = torch.sigmoid(seg_logits)
        lesion_prior = F.interpolate(seg_prob, size=(h, w), mode="bilinear", align_corners=False)

        if bypass_prob > 0.0 and self.training:
            keep = (torch.rand(b, 1, 1, 1, device=x.device) >= bypass_prob).to(lesion_prior.dtype)
            lesion_prior = lesion_prior * keep

        gate = gate_floor + (gate_ceil - gate_floor) * lesion_prior
        gated_bottleneck = bottleneck * gate

        ordinal_caps, _unused_attn = self.caps_branch(gated_bottleneck)
        pirads_prob = CapsuleLength()(ordinal_caps)
        recon = self.caps_branch.reconstruct(ordinal_caps)

        return {
            "seg_logits": seg_logits,
            "ordinal_caps": ordinal_caps,
            "pirads_prob": pirads_prob,
            "recon": recon,
            "lesion_prior": lesion_prior,
        }

    def forward_sequence_seg_first(self, x_seq, gate_floor=1.0, gate_ceil=2.0):
        """Approach B v2, CapsGRU lever: same idea as forward_sequence, but
        each slice's base pass goes through forward_seg_first (decoder-then-
        gate) instead of forward_single. Segmentation stays per-slice
        (unaffected by the GRU); the GRU refines the classification call
        using T neighbouring slices' capsule vectors -- this is the built-
        but-never-trained inter-slice-consistency mechanism from the paper's
        own described design, and PI-RADS is fundamentally a lesion-level
        (not single-2D-slice) call, so this is expected to be the largest
        remaining lever for Approach B specifically."""
        b, t, c, h, w = x_seq.shape
        flat = x_seq.view(b * t, c, h, w)
        out = self.forward_seg_first(flat, gate_floor=gate_floor, gate_ceil=gate_ceil)
        caps_seq = out["ordinal_caps"].view(b, t, CapsuleBranch.NUM_ORDINAL, CapsuleBranch.CAP_DIM)
        refined_caps = self.caps_gru(caps_seq)
        refined_prob = CapsuleLength()(refined_caps)  # (B, T, 4)

        out["seg_logits"] = out["seg_logits"].view(b, t, 1, h, w)
        out["pirads_prob"] = out["pirads_prob"].view(b, t, CapsuleBranch.NUM_ORDINAL)
        out["pirads_prob_refined"] = refined_prob
        out["ordinal_caps"] = caps_seq
        out["ordinal_caps_refined"] = refined_caps
        out["recon"] = out["recon"].view(b, t, c, h, w)
        # lesion_prior is at bottleneck resolution (BOTTLENECK_HW), not the
        # input's (h,w) -- reshape using its own actual shape, not the outer
        # function's h/w, which are the INPUT image's spatial size.
        lp_h, lp_w = out["lesion_prior"].shape[-2:]
        out["lesion_prior"] = out["lesion_prior"].view(b, t, 1, lp_h, lp_w)
        return out

    def forward_sequence(self, x_seq):
        """x_seq: (B, T, C, H, W) a stack of neighbouring slices; refines the
        PI-RADS call with CapsGRU while segmentation stays per-slice."""
        b, t, c, h, w = x_seq.shape
        flat = x_seq.view(b * t, c, h, w)
        out = self.forward_single(flat)
        caps_seq = out["ordinal_caps"].view(b, t, CapsuleBranch.NUM_ORDINAL, CapsuleBranch.CAP_DIM)
        refined_caps = self.caps_gru(caps_seq)
        refined_prob = CapsuleLength()(refined_caps)  # (B, T, 4)

        out["seg_logits"] = out["seg_logits"].view(b, t, 1, h, w)
        out["pirads_prob"] = out["pirads_prob"].view(b, t, CapsuleBranch.NUM_ORDINAL)
        out["pirads_prob_refined"] = refined_prob
        out["ordinal_caps"] = caps_seq
        out["ordinal_caps_refined"] = refined_caps
        out["recon"] = out["recon"].view(b, t, c, h, w)
        return out
