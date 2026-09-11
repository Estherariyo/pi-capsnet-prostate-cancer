"""Loads the per-slice caches produced by preprocess.py. Augmentation (random
shift, scale, flip) applied identically to image and mask, matching the
paper's stated augmentation set."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import encode_pirads_ordinal


def sample_affine_params(max_shift=8, scale_range=(0.9, 1.1)):
    """Draws one set of spatial-augmentation parameters, so a whole slice
    SEQUENCE (Approach B v2 / CapsGRU dataset) can be transformed
    consistently instead of each neighbouring slice getting an independent
    random flip/shift/scale, which would destroy the very inter-slice
    spatial correspondence CapsGRU is meant to learn from."""
    return dict(
        flip_x=np.random.rand() < 0.5,
        flip_y=np.random.rand() < 0.5,
        dy=np.random.randint(-max_shift, max_shift + 1),
        dx=np.random.randint(-max_shift, max_shift + 1),
        scale=np.random.uniform(*scale_range),
    )


def apply_affine(image, mask, params):
    c, h, w = image.shape
    if params["flip_x"]:
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()
    if params["flip_y"]:
        image = image[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()

    image = np.roll(image, shift=(params["dy"], params["dx"]), axis=(1, 2))
    mask = np.roll(mask, shift=(params["dy"], params["dx"]), axis=(1, 2))

    scale = params["scale"]
    if abs(scale - 1.0) > 1e-3:
        import scipy.ndimage as ndi
        image = ndi.zoom(image, (1, scale, scale), order=1)
        mask = ndi.zoom(mask, (1, scale, scale), order=0)
        image = _center_crop_or_pad(image, (h, w))
        mask = _center_crop_or_pad(mask, (h, w))
    return image, mask


def random_affine(image, mask, max_shift=8, scale_range=(0.9, 1.1)):
    params = sample_affine_params(max_shift, scale_range)
    return apply_affine(image, mask, params)


def intensity_jitter(image, brightness_range=0.15, contrast_range=0.15):
    """Approach B v2, targeted-augmentation lever: random brightness/contrast
    perturbation applied ONLY to lesion-positive (rank>1) samples when
    positive_extra_aug=True, on top of the spatial augmentation every sample
    already gets. csPCa-positive patches are the minority class even after
    --balanced resampling re-weights how OFTEN they're drawn -- resampling
    alone still shows the model the exact same handful of positive images
    repeatedly, just more often, rather than genuinely diverse views of them.
    This adds that diversity. Only channels 0-1 (T2W, ADC) are jittered --
    channel 2 (zonal mask) and the mask target stay untouched since they're
    binary/geometric, not intensity signals."""
    image = image.copy()
    brightness = np.random.uniform(-brightness_range, brightness_range)
    contrast = np.random.uniform(1.0 - contrast_range, 1.0 + contrast_range)
    for ch in (0, 1):
        image[ch] = np.clip((image[ch] - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    return image


def _center_crop_or_pad(arr, target_hw):
    c, h, w = arr.shape
    th, tw = target_hw
    out = np.zeros((c, th, tw), dtype=arr.dtype)
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    ty0 = max(0, (th - h) // 2)
    tx0 = max(0, (tw - w) // 2)
    hh = min(h - y0, th - ty0)
    ww = min(w - x0, tw - tx0)
    out[:, ty0:ty0 + hh, tx0:tx0 + ww] = arr[:, y0:y0 + hh, x0:x0 + ww]
    return out


class SliceCacheDataset(Dataset):
    """Reads the per-slice .npz caches produced by preprocess_picai.py
    (dataset-agnostic: image/mask/rank/patient_id fields only).

    positive_extra_aug (Approach B v2): when True and augment is also True,
    lesion-positive (rank>1) samples additionally get intensity_jitter on
    top of the spatial augmentation every sample gets -- see that function's
    docstring. Off by default so every prior run's exact behavior stays
    reproducible."""
    def __init__(self, cache_root, split="train", augment=False, positive_extra_aug=False):
        self.files = sorted(glob.glob(os.path.join(cache_root, split, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no cached slices under {cache_root}/{split}; run preprocess_picai.py first")
        self.augment = augment
        self.positive_extra_aug = positive_extra_aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        image = d["image"]  # (3,100,100)
        mask = d["mask"]  # (1,100,100)
        rank = int(d["rank"])
        patient_id = str(d["patient_id"])

        if self.augment:
            image, mask = random_affine(image, mask)
            if self.positive_extra_aug and rank > 1:
                image = intensity_jitter(image)

        pirads_bits = encode_pirads_ordinal(torch.tensor(rank)).float()
        recon_mask = np.broadcast_to(image[2:3] > 0, image.shape).astype(np.float32).copy()
        recon_mask = np.clip(recon_mask + 0.1, 0, 1)  # small floor so background isn't fully zero-weighted

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float(),
            "pirads_bits": pirads_bits,
            "recon_mask": torch.from_numpy(recon_mask).float(),
            "patient_id": patient_id,
        }
