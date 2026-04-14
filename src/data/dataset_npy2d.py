import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


class ADNINPY2DDataset(Dataset):
    """
    Read 2D slices saved as .npy from a manifest CSV.
    Required columns: npy_path, label (string: CN or AD for binary)
    """
    def __init__(self, csv_path="", label_map=None, data_root=".", df=None):
        self.df = df.reset_index(drop=True) if df is not None else pd.read_csv(csv_path)
        self.data_root = data_root
        self.label_map = label_map

        if "npy_path" not in self.df.columns:
            raise ValueError(f"Manifest must contain 'npy_path'. Got columns: {list(self.df.columns)}")
        if "label" not in self.df.columns:
            raise ValueError(f"Manifest must contain 'label'. Got columns: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        p = str(r["npy_path"])
        if not os.path.isabs(p):
            p = os.path.join(self.data_root, p)

        x = np.load(p).astype(np.float32)          # (H, W), already normalized to [0,1] in your script
        x = torch.from_numpy(x)[None, ...]         # (1, H, W)
        # resize to 224x224 for ViT
        x = F.interpolate(
            x.unsqueeze(0),  # (1, 1, H, W)
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (1, 224, 224)

        y_raw = r["label"]
        if isinstance(y_raw, str):
            y = self.label_map[y_raw]
        else:
            y = int(y_raw)

        return x, torch.tensor(y, dtype=torch.long)
