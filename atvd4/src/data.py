from pathlib import Path

import pandas as pd

FEATURES = ["C", "S", "ST", "T", "IT", "I", "IN", "N", "SN"]


def load_dataset(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    mask_completo = df[FEATURES].notna().all(axis=1)
    mask_faixa = ((df[FEATURES] >= 30) & (df[FEATURES] <= 90)).all(axis=1)
    return df.loc[mask_completo & mask_faixa].copy()
