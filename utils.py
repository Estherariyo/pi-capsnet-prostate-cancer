"""PI-RADS <-> 4-bit ordinal (cumulative) encoding, per the paper's ordinal
label scheme: each of PI-RADS {1,2,3,4,5} maps to a 4-bit vector of cumulative
thresholds [>=2, >=3, >=4, >=5], so the network can learn the ordering between
categories instead of treating them as unrelated one-hot classes."""
import torch

PIRADS_MIN, PIRADS_MAX = 1, 5
NUM_ORDINAL_BITS = PIRADS_MAX - PIRADS_MIN  # 4


def encode_pirads_ordinal(ranks):
    """ranks: LongTensor (...,) with values in [1,5]. Returns (..., 4) float."""
    thresholds = torch.arange(PIRADS_MIN + 1, PIRADS_MAX + 1, device=ranks.device)  # [2,3,4,5]
    return (ranks.unsqueeze(-1) >= thresholds).float()


def decode_pirads_ordinal(bits, threshold=0.5):
    """bits: (..., 4) in [0,1]. Returns LongTensor (...,) rank in [1,5]."""
    return PIRADS_MIN + (bits > threshold).sum(dim=-1)
