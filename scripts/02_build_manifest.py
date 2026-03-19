#!/usr/bin/env python3
"""
Build manifest: merge metadata with NIfTI paths.
Use --preprocessed to point nifti_path to MNI-registered (syn/) images for training.
"""
import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/adni431.csv", help="Metadata CSV with Subject, Group")
    ap.add_argument("--nifti_root", default="data/adni_nifti", help="Root of raw NIfTI dirs")
    ap.add_argument("--out_csv", default=None,
                    help="Output manifest (default: data/merged431_syn.csv if --preprocessed, else data/merged431.csv)")
    ap.add_argument("--preprocessed", action="store_true",
                    help="Point nifti_path to syn/ (MNI-registered). Run 01_MNI_preprocess first.")
    ap.add_argument("--data_root", default="data", help="Root for preprocessed dirs (syn/, stripped/)")
    args = ap.parse_args()

    if args.out_csv is None:
        args.out_csv = "data/merged431_syn.csv" if args.preprocessed else "data/merged431.csv"

    df = pd.read_csv(args.metadata)
    df["Subject"] = df["Subject"].astype(str)

    nifti_root = Path(args.nifti_root)
    records = []
    for subject_dir in nifti_root.iterdir():
        if subject_dir.is_dir():
            nii_files = list(subject_dir.glob("*.nii.gz"))
            if len(nii_files) == 1:
                records.append({
                    "Subject": subject_dir.name,
                    "nifti_path": nii_files[0].as_posix()
                })

    nii_df = pd.DataFrame(records)
    merged = df.merge(nii_df, on="Subject", how="inner")

    if args.preprocessed:
        syn_dir = Path(args.data_root) / "syn"
        if not syn_dir.exists():
            raise FileNotFoundError(f"Preprocessed dir not found: {syn_dir}. Run 01_MNI_preprocess first.")
        new_paths = []
        kept = []
        for idx, row in merged.iterrows():
            subject = str(row["Subject"]).strip()
            fname = f"syn_registered_{subject}_m12.nii.gz"
            full_path = syn_dir / fname
            rel_path = os.path.join(args.data_root, "syn", fname)
            if full_path.exists():
                new_paths.append(rel_path)
                kept.append(True)
            else:
                kept.append(False)
        merged = merged[kept].copy()
        merged["nifti_path"] = new_paths
        merged = merged.reset_index(drop=True)
        print(f"Preprocessed: {len(merged)} subjects (syn/)")
    else:
        print("Raw paths: data/adni_nifti/")

    print("metadata:", len(df))
    print("nii subjects:", len(nii_df))
    print("merged:", len(merged))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Done -> {args.out_csv}")


if __name__ == "__main__":
    main()
