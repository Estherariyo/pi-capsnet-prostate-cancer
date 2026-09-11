"""Approach B v2, CapsGRU lever: groups the SAME per-slice .npz cache files
SliceCacheDataset reads into T-slice neighbouring sequences, so
MiniSegCaps.forward_sequence_seg_first has the (B,T,C,H,W) input it needs.
No new preprocessing required -- this only changes how the existing cached
slices are indexed and batched.

Sequence label/target convention: the CENTER slice's rank/mask is the
supervision target (that's the ground truth we actually have per-slice);
the GRU still sees and refines using the T-slice window's full capsule
context. At volume boundaries, the edge slice is repeated rather than
padding with zeros, so the GRU never sees a fabricated blank slice.
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import sample_affine_params, apply_affine, intensity_jitter
from utils import encode_pirads_ordinal

FNAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.npz$")


class SliceSequenceDataset(Dataset):
    def __init__(self, cache_root, split="train", seq_len=3, augment=False, positive_extra_aug=False):
        assert seq_len % 2 == 1, "seq_len must be odd so there's a well-defined center slice"
        self.seq_len = seq_len
        self.augment = augment
        self.positive_extra_aug = positive_extra_aug

        files = sorted(glob.glob(os.path.join(cache_root, split, "*.npz")))
        if not files:
            raise FileNotFoundError(f"no cached slices under {cache_root}/{split}; run preprocess_picai_cv.py first")

        groups = defaultdict(list)  # (pid, sid) -> list of (z, filepath)
        for fp in files:
            m = FNAME_RE.match(os.path.basename(fp))
            if not m:
                continue
            pid, sid, z = m.group(1), m.group(2), int(m.group(3))
            groups[(pid, sid)].append((z, fp))
        for key in groups:
            groups[key].sort(key=lambda t: t[0])

        # every retained slice can be a sequence center; store (group_key, index_within_group)
        self.groups = groups
        self.index = []
        for key, items in groups.items():
            for i in range(len(items)):
                self.index.append((key, i))

    def __len__(self):
        return len(self.index)

    def center_files(self):
        """Center-slice filepaths in index order -- same length/order as
        the dataset, so train.py's build_sample_weights (rank-per-file
        lookup) works unchanged for the balanced sampler."""
        return [self.groups[key][center_i][1] for key, center_i in self.index]

    def _neighbours(self, key, center_i):
        items = self.groups[key]
        n = len(items)
        half = self.seq_len // 2
        idxs = [min(max(center_i + off, 0), n - 1) for off in range(-half, half + 1)]
        return [items[i][1] for i in idxs]

    def __getitem__(self, idx):
        key, center_i = self.index[idx]
        paths = self._neighbours(key, center_i)

        images, masks = [], []
        params = sample_affine_params() if self.augment else None
        center = self.seq_len // 2
        center_rank = None
        center_image = None
        for t, fp in enumerate(paths):
            d = np.load(fp)
            image = d["image"]
            mask = d["mask"]
            if t == center:
                center_rank = int(d["rank"])
                patient_id = str(d["patient_id"])
            if self.augment:
                image, mask = apply_affine(image, mask, params)
                if self.positive_extra_aug and int(d["rank"]) > 1:
                    image = intensity_jitter(image)
            if t == center:
                center_image = image  # captured post-augmentation, for the recon target below
            images.append(image)
            masks.append(mask)

        image_seq = np.stack(images, axis=0)  # (T,3,100,100)
        mask_center = masks[center]  # (1,100,100) -- center slice is the segmentation supervision target

        pirads_bits = encode_pirads_ordinal(torch.tensor(center_rank)).float()
        # recon target/mask are center-slice-only (C,H,W), matching
        # CombinedLoss's existing single-slice shape expectations -- train.py's
        # sequence epoch loop indexes model.recon at the same center timestep.
        recon_mask = np.broadcast_to(center_image[2:3] > 0, center_image.shape).astype(np.float32).copy()
        recon_mask = np.clip(recon_mask + 0.1, 0, 1)

        return {
            "image_seq": torch.from_numpy(image_seq.copy()).float(),
            "mask_center": torch.from_numpy(mask_center.copy()).float(),
            "pirads_bits": pirads_bits,
            "recon_target": torch.from_numpy(center_image.copy()).float(),
            "recon_mask": torch.from_numpy(recon_mask).float(),
            "patient_id": patient_id,
        }
