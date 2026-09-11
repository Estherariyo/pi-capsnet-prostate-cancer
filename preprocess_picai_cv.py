"""Preprocess PI-CAI once into a flat cache annotated with a 5-fold,
patient-level, stratified-by-csPCa fold assignment, so 5-fold cross-
validation can reuse a single preprocessing pass (build_cv_folds.py then
hardlinks each fold's train/valid split from this cache; the fold assignment
here is the ground truth for that split).

Identical image/label handling to preprocess_picai.py (T2W+ADC+zonal-mask
input, ADC resampled onto the T2W grid, ISUP-to-ordinal-rank mapping);
see that file's docstring for the full rationale.

CORRECTNESS FIXES (post-hoc audit, applied here to match preprocess_picai.py):
1. Lesion annotations are split across csPCa_lesion_delineations/human_expert/
   resampled/ (1,295 cases) and .../human_expert/Pooch25/ (205 cases added by
   a July-2025 update). The previous version only checked `resampled/`,
   silently mislabeling csPCa-positive patients whose annotation lives only
   in Pooch25/ as benign on every slice. Both directories share the T2W-
   aligned resampled grid (verified directly), so no extra resampling is
   needed -- this version checks both, in order, and reports final coverage.
2. process_case used to return the bare int 0 on early-exit (missing patient
   directory), while its success path returns a list of (filename, rank)
   tuples -- callers do `for fname, rank in saved`, which would raise
   TypeError if that early-exit path were ever actually hit. It happened not
   to fire in the run this dataset went through, but it's a latent crash
   risk on any future run with incomplete image extraction. Fixed to always
   return a list.
"""
import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

PATCH = 100
ISUP_TO_RANK = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 5}


def load_img(path):
    return sitk.ReadImage(path)


def resample_to_reference(moving, reference):
    return sitk.Resample(moving, reference, sitk.Transform(), sitk.sitkLinear, 0.0, moving.GetPixelID())


def normalize(img, pmin=0.5, pmax=99.5):
    lo, hi = np.percentile(img, [pmin, pmax])
    img = np.clip(img, lo, hi)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img.astype(np.float32)


def crop_or_pad(arr2d, center_yx, patch=PATCH, pad_value=0.0):
    h, w = arr2d.shape
    cy, cx = center_yx
    y0, x0 = int(round(cy - patch / 2)), int(round(cx - patch / 2))
    out = np.full((patch, patch), pad_value, dtype=arr2d.dtype)
    sy0, sy1 = max(0, y0), min(h, y0 + patch)
    sx0, sx1 = max(0, x0), min(w, x0 + patch)
    dy0, dx0 = sy0 - y0, sx0 - x0
    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr2d[sy0:sy1, sx0:sx1]
    return out


def find_patient_dir(images_root, patient_id):
    d = os.path.join(images_root, str(patient_id))
    return d if os.path.isdir(d) else None


def find_lesion_path(lesion_roots, prefix):
    for root in lesion_roots:
        candidate = os.path.join(root, f"{prefix}.nii.gz")
        if os.path.exists(candidate):
            return candidate
    return None


def process_case(row, images_root, zonal_root, lesion_roots, out_dir):
    """Returns (saved_files, diagnostic); saved_files is always a list
    (possibly empty) so callers can safely `for fname, rank in saved`
    regardless of which early-exit path was taken."""
    pid, sid = int(row["patient_id"]), int(row["study_id"])
    case_isup = int(row["case_ISUP"])
    case_cspca = row["case_csPCa"] == "YES"
    rank_case = ISUP_TO_RANK[case_isup]

    diag = {"case_cspca": case_cspca, "lesion_file_found": False, "max_lesion_pixels": 0.0}

    pdir = find_patient_dir(images_root, pid)
    if pdir is None:
        return [], diag
    prefix = f"{pid}_{sid}"
    t2w_path = os.path.join(pdir, f"{prefix}_t2w.mha")
    adc_path = os.path.join(pdir, f"{prefix}_adc.mha")
    zonal_path = os.path.join(zonal_root, f"{prefix}.nii.gz")
    lesion_path = find_lesion_path(lesion_roots, prefix)
    diag["lesion_file_found"] = lesion_path is not None
    if not (os.path.exists(t2w_path) and os.path.exists(adc_path) and os.path.exists(zonal_path)):
        return [], diag

    t2w_im = load_img(t2w_path)
    adc_im = resample_to_reference(load_img(adc_path), t2w_im)
    zonal_im = load_img(zonal_path)

    t2 = sitk.GetArrayFromImage(t2w_im).astype(np.float32)
    adc = sitk.GetArrayFromImage(adc_im).astype(np.float32)
    zonal_bin = (sitk.GetArrayFromImage(zonal_im).astype(np.float32) > 0).astype(np.float32)

    if lesion_path is not None:
        lesion_im = load_img(lesion_path)
        tumor_bin = (sitk.GetArrayFromImage(lesion_im).astype(np.float32) > 0).astype(np.float32)
    else:
        tumor_bin = np.zeros_like(zonal_bin)

    t2n = normalize(t2)
    adcn = normalize(adc)

    saved_files = []
    for z in range(t2.shape[0]):
        zmask = zonal_bin[z]
        if zmask.sum() < 10:
            continue
        ys, xs = np.nonzero(zmask)
        center = (ys.mean(), xs.mean())

        img_ch0 = crop_or_pad(t2n[z], center)
        img_ch1 = crop_or_pad(adcn[z], center)
        img_ch2 = crop_or_pad(zmask, center)
        image = np.stack([img_ch0, img_ch1, img_ch2], axis=0)

        mask = crop_or_pad(tumor_bin[z], center)[None]
        mask_pixels = float(mask.sum())
        diag["max_lesion_pixels"] = max(diag["max_lesion_pixels"], mask_pixels)
        lesion_present = mask_pixels > 0
        rank = rank_case if lesion_present else 1

        fname = f"{pid}_{sid}_{z:03d}.npz"
        out_path = os.path.join(out_dir, fname)
        np.savez_compressed(out_path, image=image.astype(np.float32),
                             mask=mask.astype(np.float32), rank=np.int64(rank),
                             patient_id=str(pid))
        saved_files.append((fname, rank))
    return saved_files, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default="/workspace/minisegcaps/data/picai_images/extracted")
    ap.add_argument("--labels-root", default="/workspace/minisegcaps/data/picai_labels")
    ap.add_argument("--out-root", default="/workspace/minisegcaps/data/picai_cache_cv")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    marksheet = pd.read_csv(os.path.join(args.labels_root, "clinical_information", "marksheet.csv"))
    zonal_root = os.path.join(args.labels_root, "anatomical_delineations", "zonal_pz_tz", "AI", "Yuan23")
    lesion_roots = [
        os.path.join(args.labels_root, "csPCa_lesion_delineations", "human_expert", "resampled"),
        os.path.join(args.labels_root, "csPCa_lesion_delineations", "human_expert", "Pooch25"),
    ]
    for r in lesion_roots:
        if not os.path.isdir(r):
            print(f"WARNING: expected lesion-annotation directory not found: {r}")

    all_dir = os.path.join(args.out_root, "all")
    os.makedirs(all_dir, exist_ok=True)

    # patient-level stratified fold assignment by case_csPCa (YES/NO)
    patient_label = marksheet.groupby("patient_id")["case_csPCa"].agg(lambda s: "YES" if (s == "YES").any() else "NO")
    patient_ids = patient_label.index.to_numpy()
    labels = patient_label.to_numpy()

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_of_patient = {}
    for fold_idx, (_, val_idx) in enumerate(skf.split(patient_ids, labels)):
        for pid in patient_ids[val_idx]:
            fold_of_patient[int(pid)] = fold_idx

    manifest_rows = []
    total_slices = 0
    cspca_diag = {"total": 0, "no_file": 0, "file_but_empty": 0, "ok": 0}

    for _, row in tqdm(marksheet.iterrows(), total=len(marksheet), desc="PI-CAI cases"):
        pid = int(row["patient_id"])
        fold = fold_of_patient.get(pid)
        if fold is None:
            continue
        saved, diag = process_case(row, args.images_root, zonal_root, lesion_roots, all_dir)
        for fname, rank in saved:
            manifest_rows.append({"filename": fname, "rank": rank, "patient_id": pid, "fold": fold})
        total_slices += len(saved)

        if diag["case_cspca"]:
            cspca_diag["total"] += 1
            if not diag["lesion_file_found"]:
                cspca_diag["no_file"] += 1
            elif diag["max_lesion_pixels"] <= 0:
                cspca_diag["file_but_empty"] += 1
            else:
                cspca_diag["ok"] += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_root, "all_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"total: {len(manifest_rows)} slices from {manifest['patient_id'].nunique()} patients -> {all_dir}")
    print("fold sizes (patients):", {k: int((patient_label.index.map(fold_of_patient) == k).sum()) for k in range(args.n_folds)})
    print("manifest ->", manifest_path)

    d = cspca_diag
    print("\n=== csPCa (case_csPCa=YES) label-coverage diagnostic (whole dataset) ===")
    print(f"{d['total']} csPCa patients total | {d['ok']} have a usable non-empty lesion mask | "
          f"{d['no_file']} have NO lesion annotation file in any source | "
          f"{d['file_but_empty']} have a file but zero lesion pixels in every retained slice")
    gap = d["no_file"] + d["file_but_empty"]
    if gap > 0:
        pct = 100.0 * gap / max(1, d["total"])
        print(f"  -> {gap}/{d['total']} ({pct:.1f}%) csPCa patients will be trained/evaluated as if benign on every slice. "
              f"Investigate before training if this is larger than a handful of cases.")


if __name__ == "__main__":
    main()
