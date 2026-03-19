import pandas as pd
from pathlib import Path

df = pd.read_csv("data/adni431.csv")
df["Subject"] = df["Subject"].astype(str)

nifti_root = Path("data/adni_nifti")
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

print("metadata:", len(df))
print("nii subjects:", len(nii_df))
print("merged:", len(merged))

merged.to_csv("data/merged431.csv", index=False)

print("Done → data/merged431.csv")
