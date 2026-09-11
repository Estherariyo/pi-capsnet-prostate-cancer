"""Capsule layers with dynamic routing (Sabour et al. 2017), adapted to the
convolutional-capsule form used by the MiniSegCaps capsule branch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s, dim=-1, eps=1e-8):
    """Non-linearity that keeps short vectors near zero and long vectors near
    unit length, so capsule vector length can be read as an activation prob."""
    norm2 = (s ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm2 + eps)
    return (norm2 / (1.0 + norm2)) * (s / norm)


class PrimaryCapsuleConv(nn.Module):
    """Turns a conv feature map into a grid of primary capsules via a single
    conv whose output channels are reshaped into (num_caps, cap_dim)."""

    def __init__(self, in_channels, num_caps, cap_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.conv = nn.Conv2d(in_channels, num_caps * cap_dim, kernel_size, stride, padding)

    def forward(self, x):
        b = x.shape[0]
        out = self.conv(x)  # (B, num_caps*cap_dim, H, W)
        h, w = out.shape[-2:]
        out = out.view(b, self.num_caps, self.cap_dim, h, w)
        out = out.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, num_caps, cap_dim)
        out = out.view(b, h * w * self.num_caps, self.cap_dim)
        return squash(out, dim=-1)


class ConvCapsuleLayer(nn.Module):
    """Convolutional capsule layer with EM-free (vector) dynamic routing.

    Each input capsule type is transformed per output capsule type by a small
    conv (acting as the transformation matrix W_ij shared across spatial
    locations), then routed with the standard iterative agreement procedure.
    """

    def __init__(self, in_caps, in_dim, out_caps, out_dim, kernel_size=3,
                 stride=1, padding=1, routing_iters=3):
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        # one conv per (in_cap -> all out_caps*out_dim) keeps this tractable
        self.transform = nn.Conv2d(
            in_caps * in_dim, out_caps * out_dim, kernel_size, stride, padding
        )

    def forward(self, x):
        # x: (B, H, W, in_caps, in_dim)
        b, h, w, in_caps, in_dim = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(b, in_caps * in_dim, h, w)
        u_hat = self.transform(x_flat)  # (B, out_caps*out_dim, H', W')
        h2, w2 = u_hat.shape[-2:]
        u_hat = u_hat.view(b, self.out_caps, self.out_dim, h2, w2)
        u_hat = u_hat.permute(0, 3, 4, 1, 2).contiguous()  # (B,H',W',out_caps,out_dim)

        # dynamic routing treats each spatial cell independently, routing from
        # a single aggregated input capsule set (in_caps) to out_caps
        logits = torch.zeros(b, h2, w2, self.out_caps, device=x.device, dtype=x.dtype)
        v = None
        for r in range(self.routing_iters):
            c = F.softmax(logits, dim=-1)  # coupling coeffs over out_caps
            s = u_hat * c.unsqueeze(-1)
            v = squash(s, dim=-1)
            if r < self.routing_iters - 1:
                logits = logits + (u_hat * v).sum(dim=-1)
        return v  # (B, H', W', out_caps, out_dim)


class CapsuleLength(nn.Module):
    """Capsule vector length == activation probability for that capsule type."""

    def forward(self, x):
        return torch.sqrt((x ** 2).sum(dim=-1) + 1e-8)
