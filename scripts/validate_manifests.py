#!/usr/bin/env python3
"""
Check manifests for:
  - Binary labels AD vs CN (only keys in --label_map appear in Group)
  - One row per Subject in each split; Subjects disjoint across train/val/test
  - One nifti_path per Subject globally (no duplicate paths)
  - Optional: nifti files exist under --data_root

Usage:
  python scripts/validate_manifests.py --manifest_dir manifests --data_root .
  python scripts/validate_manifests.py --manifest_dir manifests --data_root /scratch/.../adni-dttvit --check_files
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _resolve_cols(df: pd.DataFrame):
    subj = "Subject" if "Subject" in df.columns else "subject"
    lab = "Group" if "Group" in df.columns else ("label" if "label" in df.columns else None)
    path = "nifti_path" if "nifti_path" in df.columns else "path"
    if subj not in df.columns:
        raise SystemExit(f"Missing Subject column; have {list(df.columns)}")
    if lab is None or lab not in df.columns:
        raise SystemExit(f"Missing Group/label column; have {list(df.columns)}")
    if path not in df.columns:
        raise SystemExit(f"Missing nifti_path/path column; have {list(df.columns)}")
    return subj, lab, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default="manifests")
    ap.add_argument("--data_root", default=".", help="Root for relative nifti_path")
    ap.add_argument(
        "--label_map",
        default="CN=0,AD=1",
        help='Allowed class names in CSV, e.g. "CN=0,AD=1" (only these keys may appear)',
    )
    ap.add_argument("--check_files", action="store_true")
    args = ap.parse_args()

    allowed_labels = {x.split("=", 1)[0].strip() for x in args.label_map.split(",")}

    tables = {}
    for split in ("train", "val", "test"):
        fp = os.path.join(args.manifest_dir, f"{split}.csv")
        if not os.path.isfile(fp):
            print(f"ERROR: missing {fp}", file=sys.stderr)
            sys.exit(1)
        tables[split] = pd.read_csv(fp)

    subj_col, lab_col, path_col = _resolve_cols(tables["train"])

    errors = []
    warnings = []

    all_subjects = []
    all_paths = []
    split_subjects = {}

    for split, df in tables.items():
        s_col, l_col, p_col = _resolve_cols(df)
        if {s_col, l_col, p_col} != {subj_col, lab_col, path_col}:
            errors.append(f"{split}: column names inconsistent with train.csv")

        subj = df[subj_col].astype(str).str.strip()
        if subj.duplicated().any():
            dup = subj[subj.duplicated(keep=False)].unique().tolist()
            errors.append(f"{split}: duplicate Subject rows: {dup[:8]}{'...' if len(dup) > 8 else ''}")

        labs = df[l_col].astype(str).str.strip()
        bad = sorted(set(labs.unique()) - allowed_labels)
        if bad:
            errors.append(
                f"{split}: labels not in {sorted(allowed_labels)}: {bad} "
                f"(binary AD/CN only — remove MCI etc. or extend --label_map)"
            )

        paths = df[p_col].astype(str).str.strip()
        if paths.isna().any() or (paths == "").any():
            errors.append(f"{split}: empty nifti_path")

        split_subjects[split] = set(subj)
        all_subjects.extend(subj.tolist())
        all_paths.extend(paths.tolist())

    # disjoint splits
    t, v, te = split_subjects["train"], split_subjects["val"], split_subjects["test"]
    if t & v:
        errors.append(f"train∩val non-empty: {list(t & v)[:5]}")
    if t & te:
        errors.append(f"train∩test non-empty: {list(t & te)[:5]}")
    if v & te:
        errors.append(f"val∩test non-empty: {list(v & te)[:5]}")

    # global one-subject-one-volume: same as row count == unique subjects
    n_total_rows = sum(len(tables[s]) for s in tables)
    u_sub = set(all_subjects)
    if n_total_rows != len(u_sub):
        errors.append(f"Total rows {n_total_rows} != unique subjects {len(u_sub)} (subject appears in multiple splits?)")

    u_path = {}
    for sp, df in tables.items():
        _, _, p_col = _resolve_cols(df)
        for _, row in df.iterrows():
            p = str(row[p_col]).strip()
            sid = str(row[subj_col]).strip()
            if p in u_path and u_path[p] != sid:
                errors.append(f"nifti_path reused by different subjects: {p}")
            u_path[p] = sid
    if len(all_paths) != len(set(all_paths)):
        errors.append("Duplicate nifti_path across manifest rows")

    # class balance info (both classes should appear somewhere for binary task)
    all_labs = pd.concat(
        [tables[s][lab_col].astype(str).str.strip() for s in ("train", "val", "test")],
        ignore_index=True,
    )
    present = set(all_labs.unique())
    if not present <= allowed_labels:
        pass  # already in errors
    elif len(present) < 2:
        warnings.append(f"Only one class label in all splits: {present} (should be CN and AD for binary)")

    if args.check_files:
        root = os.path.abspath(args.data_root)
        for split, df in tables.items():
            _, _, p_col = _resolve_cols(df)
            for _, row in df.iterrows():
                rel = str(row[p_col]).strip()
                full = rel if os.path.isabs(rel) else os.path.join(root, rel)
                if not os.path.isfile(full):
                    errors.append(f"missing file ({split}): {full}")

    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("OK: manifests pass checks.")
    print(f"  Subjects: train={len(t)} val={len(v)} test={len(te)} total={len(u_sub)}")
    lab_series = pd.concat(
        [tables[s][lab_col].astype(str).str.strip() for s in ("train", "val", "test")],
        ignore_index=True,
    )
    print("  Class counts (all splits):", lab_series.value_counts().sort_index().to_dict())
    if args.check_files:
        print(f"  All nifti_path files exist under data_root={root}")


if __name__ == "__main__":
    main()
