#!/usr/bin/env python3
"""
Build manifest pointing to preprocessed (MNI-registered) NIfTI for training.

Use this AFTER 01_MNI_preprocess.py has completed. Creates a manifest with
nifti_path pointing to syn/ (or stripped/ if --stage stripped).

Usage:
  python scripts/04_build_manifest_preprocessed.py --in_csv data/merged431.csv --out_csv data/merged431_syn.csv
  python scripts/03_make_splits.py --in_csv data/merged431_syn.csv --out_dir manifests
"""
import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="data/merged431.csv",
                    help="Input manifest (from 02) with Subject, Group, nifti_path")
    ap.add_argument("--out_csv", type=str, default="data/merged431_syn.csv",
                    help="Output manifest with nifti_path -> preprocessed")
    ap.add_argument("--data_root", type=str, default="data",
                    help="Data root (syn/, stripped/ are under this)")
    ap.add_argument("--stage", type=str, default="syn", choices=["syn", "stripped", "affine"],
                    help="Preprocessed stage: syn (MNI SyN), stripped (skull-stripped), affine")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df.columns = df.columns.str.strip()

    subj_col = "Subject" if "Subject" in df.columns else "subject"
    if subj_col not in df.columns:
        raise ValueError(f"Missing Subject column. Found: {list(df.columns)}")

    if args.stage == "syn":
        subdir = "syn"
        prefix = "syn_registered_"
        suffix = ".nii.gz"
    elif args.stage == "stripped":
        subdir = "stripped"
        prefix = ""
        suffix = "_brain.nii.gz"
    else:
        subdir = "affine"
        prefix = "affine_registered_"
        suffix = ".nii.gz"

    preproc_dir = Path(args.data_root) / subdir
    if not preproc_dir.exists():
        raise FileNotFoundError(f"Preprocessed dir not found: {preproc_dir}")

    new_paths = []
    missing = []
    for idx, row in df.iterrows():
        subject = str(row[subj_col]).strip()
        base_name = subject + "_m12"
        if args.stage == "syn":
            fname = f"{prefix}{base_name}{suffix}"
        elif args.stage == "stripped":
            fname = f"{base_name}{suffix}"
        else:
            fname = f"{prefix}{base_name}{suffix}"
        full_path = preproc_dir / fname
        rel_path = os.path.join(args.data_root, subdir, fname)
        if full_path.exists():
            new_paths.append(rel_path)
        else:
            new_paths.append(None)
            missing.append(subject)

    df["nifti_path"] = new_paths
    df_out = df[df["nifti_path"].notna()].copy()
    df_out = df_out.reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    print(f"Input: {len(df)} rows")
    print(f"Output: {len(df_out)} rows (pointing to {args.stage}/)")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"Done -> {args.out_csv}")


if __name__ == "__main__":
    main()
