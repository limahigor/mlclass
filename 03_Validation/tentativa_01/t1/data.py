import pandas as pd

from .config import APP_PATH, TRAIN_PATH


def load_datasets():
    df = pd.read_csv(TRAIN_PATH)
    app_df = pd.read_csv(APP_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    app_df.columns = [c.strip().lower() for c in app_df.columns]
    return df, app_df


def get_columns(df):
    feature_cols = [c for c in df.columns if c != 'type']
    numeric_cols = [c for c in feature_cols if c != 'sex']
    categorical_cols = ['sex']
    return feature_cols, numeric_cols, categorical_cols
