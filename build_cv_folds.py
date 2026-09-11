"""Builds fold{k}/train and fold{k}/valid directories (hardlinks, falling
back to copies if hardlinking across the mount fails) for k in 0..4 from the
flat all_manifest.csv cache produced by preprocess_picai_cv.py.

EFFICIENCY FIX (post-hoc audit): also writes a {split}_manifest.csv into each
fold's directory (filename,rank), so train.py's balanced sampler can read it
directly instead of falling back to scanning every .npz file individually to
recover its rank -- that fallback scan cost several minutes to tens of
minutes per fold on the CV run's network-mounted storage (5 folds x 2 splits
= 10 avoidable scans), for data this script already knows the rank of."""
import argparse
import csv
import os
import shutil

import pandas as pd
from tqdm import tqdm


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", default="/workspace/minisegcaps/data/picai_cache_cv")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    manifest = pd.read_csv(os.path.join(args.cv_root, "all_manifest.csv"))
    all_dir = os.path.join(args.cv_root, "all")

    for k in range(args.n_folds):
        train_dir = os.path.join(args.cv_root, f"fold{k}", "train")
        valid_dir = os.path.join(args.cv_root, f"fold{k}", "valid")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        valid_rows = manifest[manifest["fold"] == k]
        train_rows = manifest[manifest["fold"] != k]

        for _, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc=f"fold{k} valid"):
            link_or_copy(os.path.join(all_dir, row["filename"]), os.path.join(valid_dir, row["filename"]))
        for _, row in tqdm(train_rows.iterrows(), total=len(train_rows), desc=f"fold{k} train"):
            link_or_copy(os.path.join(all_dir, row["filename"]), os.path.join(train_dir, row["filename"]))

        # per-fold rank manifests, so train.py never needs the slow fallback scan
        for split, rows, split_dir in (("train", train_rows, train_dir), ("valid", valid_rows, valid_dir)):
            manifest_path = os.path.join(split_dir.rsplit(os.sep, 1)[0], f"{split}_manifest.csv")
            with open(manifest_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filename", "rank"])
                for _, row in rows.iterrows():
                    w.writerow([row["filename"], row["rank"]])

        print(f"fold{k}: train={len(train_rows)} valid={len(valid_rows)} (manifests written)")


if __name__ == "__main__":
    main()
