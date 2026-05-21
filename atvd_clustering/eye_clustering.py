#!/usr/bin/env python3
"""Análise de clustering para perfis oculares usando a base XLSX."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["AL", "ACD", "WTW", "K1", "K2"]
TARGET_COLUMN = "Correto"
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "barrettII_eyes_clustering.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILES = {
    "clustered": "barrettII_eyes_clustering_clustered.xlsx",
    "profiles": "barrettII_cluster_profiles.xlsx",
    "k_selection": "barrettII_k_selection.csv",
}
K_RANGE = range(2, 7)
RANDOM_STATE = 42
N_INIT = 20


def load_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_excel(file_path)
    missing_columns = [column for column in FEATURES if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Colunas ausentes na base: {missing}")

    if TARGET_COLUMN not in data.columns:
        print(
            f"Aviso: coluna complementar '{TARGET_COLUMN}' não encontrada. "
            "A análise seguirá apenas com as variáveis do clustering."
        )
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned[FEATURES] = cleaned[FEATURES].apply(pd.to_numeric, errors="coerce")

    invalid_rows = cleaned[FEATURES].isna().any(axis=1)
    if invalid_rows.any():
        removed = int(invalid_rows.sum())
        print(
            "Removendo "
            f"{removed} registro(s) com valores ausentes ou inválidos "
            "nas variáveis do clustering."
        )
        cleaned = cleaned.loc[~invalid_rows].copy()

    duplicated_rows = cleaned.duplicated(subset=FEATURES)
    if duplicated_rows.any():
        duplicates = int(duplicated_rows.sum())
        print(
            f"Aviso: {duplicates} registro(s) duplicado(s) nas variáveis "
            "do clustering foram mantidos."
        )

    if cleaned.empty:
        raise ValueError("Não restaram registros válidos para executar o clustering.")

    return cleaned.reset_index(drop=True)


def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[FEATURES])
    return pd.DataFrame(x_scaled, columns=FEATURES, index=data.index)


def choose_k(x_scaled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_scaled, labels),
                "inertia": model.inertia_,
            }
        )
    return pd.DataFrame(rows)


def select_best_k(k_table: pd.DataFrame) -> int:
    ranked = k_table.sort_values(["silhouette", "k"], ascending=[False, True])
    return int(ranked.iloc[0]["k"])


def build_cluster_labels(clustered: pd.DataFrame) -> pd.Series:
    al_by_cluster = clustered.groupby("cluster_id")["AL"].mean().sort_values()
    cluster_name_map = {
        cluster_id: f"Grupo {index}"
        for index, cluster_id in enumerate(al_by_cluster.index, start=1)
    }
    return clustered["cluster_id"].map(cluster_name_map)


def summarize_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    stats = clustered.groupby("cluster")[FEATURES].agg(
        ["mean", "median", "std", "min", "max"]
    )
    quartiles = clustered.groupby("cluster")[FEATURES].quantile([0.25, 0.75]).unstack()
    quartiles.columns = pd.MultiIndex.from_tuples(
        [
            (feature, "q1" if quantile == 0.25 else "q3")
            for feature, quantile in quartiles.columns
        ]
    )

    profile = stats.round(3).join(quartiles.round(3))

    if TARGET_COLUMN in clustered.columns:
        target_distribution = (
            clustered.groupby("cluster")[TARGET_COLUMN]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percentual")
            .round(2)
            .reset_index()
            .pivot(index="cluster", columns=TARGET_COLUMN, values="percentual")
            .fillna(0.0)
        )
        target_distribution.columns = pd.MultiIndex.from_tuples(
            [(TARGET_COLUMN, f"{col}_percentual") for col in target_distribution.columns]
        )
        profile = profile.join(target_distribution)

    counts = clustered.groupby("cluster").size().rename("quantidade")
    counts_frame = counts.to_frame()
    counts_frame.columns = pd.MultiIndex.from_tuples([("geral", "quantidade")])
    return counts_frame.join(profile)


def fit_clusters(
    data: pd.DataFrame, x_scaled: pd.DataFrame, n_clusters: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = KMeans(n_clusters=n_clusters, n_init=N_INIT, random_state=RANDOM_STATE)
    cluster_ids = model.fit_predict(x_scaled)

    clustered = data.copy()
    clustered["cluster_id"] = cluster_ids
    clustered["cluster"] = build_cluster_labels(clustered)

    profile = summarize_profiles(clustered)
    return clustered, profile


def export_results(
    output_dir: Path,
    clustered: pd.DataFrame,
    profile: pd.DataFrame,
    k_table: pd.DataFrame,
) -> None:
    output_dir.mkdir(exist_ok=True)
    clustered.to_excel(output_dir / OUTPUT_FILES["clustered"], index=False)
    profile.to_excel(output_dir / OUTPUT_FILES["profiles"])
    k_table.to_csv(output_dir / OUTPUT_FILES["k_selection"], index=False)


def print_summary(
    data: pd.DataFrame,
    clustered: pd.DataFrame,
    k_table: pd.DataFrame,
    best_k: int,
) -> None:
    print("=" * 72)
    print("ANÁLISE DE CLUSTERING - PERFIS OCULARES")
    print("=" * 72)
    print(f"Base analisada : {DATA_FILE.name}")
    print(f"Registros lidos: {len(data)}")
    print(f"Variáveis      : {', '.join(FEATURES)}")
    print(f"Coluna auxiliar: {TARGET_COLUMN} (opcional, fora do clustering)")
    print()
    print("Seleção do número de grupos")
    print("-" * 72)
    print(k_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Melhor valor de k: {best_k}")
    print()
    print("Resumo dos grupos")
    print("-" * 72)
    summary = (
        clustered.groupby("cluster")[FEATURES]
        .mean()
        .round(3)
        .join(clustered.groupby("cluster").size().rename("quantidade"))
    )
    summary = summary[["quantidade", *FEATURES]]
    print(summary.to_string())
    print()
    print("Arquivos gerados")
    print("-" * 72)
    for key in ["clustered", "profiles", "k_selection"]:
        print(f"- {output_path(OUTPUT_FILES[key])}")


def output_path(file_name: str) -> str:
    return str(OUTPUT_DIR / file_name)


def main() -> None:
    data = clean_data(load_data(DATA_FILE))
    x_scaled = scale_features(data)
    k_table = choose_k(x_scaled)
    best_k = select_best_k(k_table)
    clustered, profile = fit_clusters(data, x_scaled, n_clusters=best_k)

    export_results(OUTPUT_DIR, clustered, profile, k_table)
    print_summary(data, clustered, k_table, best_k)


if __name__ == "__main__":
    main()
