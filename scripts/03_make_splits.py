#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="data/merged431_syn.csv",
                    help="Input manifest (use data/merged431.csv for raw, data/merged431_syn.csv for preprocessed)")
    ap.add_argument("--out_dir", type=str, default="manifests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # subject column
    subj_col = "Subject" if "Subject" in df.columns else "subject"
    if subj_col not in df.columns:
        raise ValueError(f"Missing Subject/subject. Found: {list(df.columns)}")

    # label column (optional, for stratify)
    label_col = "Group" if "Group" in df.columns else ("label" if "label" in df.columns else None)

    subjects = df[subj_col].astype(str)

    # split subjects (stratify if label exists)
    if label_col is not None:
        y = df[label_col].astype(str)
        train_subj, temp_subj = train_test_split(subjects, test_size=0.2, random_state=args.seed, stratify=y)
        y_temp = df.loc[temp_subj.index, label_col].astype(str)
        val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=args.seed, stratify=y_temp)
    else:
        train_subj, temp_subj = train_test_split(subjects, test_size=0.2, random_state=args.seed)
        val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=args.seed)

    train_df = df[df[subj_col].astype(str).isin(set(train_subj))].copy()
    val_df   = df[df[subj_col].astype(str).isin(set(val_subj))].copy()
    test_df  = df[df[subj_col].astype(str).isin(set(test_subj))].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("OK rows:", len(train_df), len(val_df), len(test_df))
    if label_col is not None:
        print("train label counts:\n", train_df[label_col].value_counts())
        print("val label counts:\n", val_df[label_col].value_counts())
        print("test label counts:\n", test_df[label_col].value_counts())

if __name__ == "__main__":
    main()
