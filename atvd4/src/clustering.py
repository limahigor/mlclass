import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES

PROFILE_ORDER = [
    "A - Fino difuso",
    "B - Intermediário baixo",
    "C - Assimétrico superior/nasal",
    "D - Intermediário alto",
    "E - Espesso difuso",
]


def scale_features(df: pd.DataFrame):
    X = df[FEATURES].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, scaler, X_scaled


def evaluate_clusters(X_scaled, k_values=range(2, 9)) -> pd.DataFrame:
    resultados = []
    for k in k_values:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = modelo.fit_predict(X_scaled)
        resultados.append(
            {
                "k": k,
                "inertia": modelo.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(resultados)


def fit_final_model(
    df_clean: pd.DataFrame, scaler: StandardScaler, X_scaled, k_final: int = 5
):
    modelo_final = KMeans(n_clusters=k_final, random_state=42, n_init=50)
    labels = modelo_final.fit_predict(X_scaled)
    df_clustered = df_clean.copy()
    df_clustered["cluster_raw"] = labels

    centros = pd.DataFrame(
        scaler.inverse_transform(modelo_final.cluster_centers_),
        columns=FEATURES,
    )
    centros["cluster_raw"] = range(k_final)
    centros["n"] = df_clustered["cluster_raw"].value_counts().sort_index().values
    centros["pct"] = centros["n"] / len(df_clustered) * 100
    centros["mean_epi"] = centros[FEATURES].mean(axis=1)
    centros["range_epi"] = centros[FEATURES].max(axis=1) - centros[FEATURES].min(axis=1)
    centros["S_minus_I"] = centros["S"] - centros["I"]
    centros["N_minus_T"] = centros["N"] - centros["T"]
    centros["IT_minus_SN"] = centros["IT"] - centros["SN"]
    centros["min_region"] = centros[FEATURES].idxmin(axis=1)
    centros["max_region"] = centros[FEATURES].idxmax(axis=1)

    centros_ordenados = centros.sort_values("mean_epi").copy()
    cluster_assimetrico = centros["range_epi"].idxmax()
    restantes = [
        c for c in centros_ordenados["cluster_raw"].tolist() if c != cluster_assimetrico
    ]

    name_map = {
        restantes[0]: "A - Fino difuso",
        restantes[1]: "B - Intermediário baixo",
        cluster_assimetrico: "C - Assimétrico superior/nasal",
        restantes[2]: "D - Intermediário alto",
        restantes[3]: "E - Espesso difuso",
    }

    df_clustered["Perfil"] = df_clustered["cluster_raw"].map(name_map)
    centros["Perfil"] = centros["cluster_raw"].map(name_map)

    tabela_perfis = centros.set_index("Perfil").reindex(PROFILE_ORDER)
    tabela_perfis = tabela_perfis[
        ["n", "pct"]
        + FEATURES
        + [
            "mean_epi",
            "range_epi",
            "S_minus_I",
            "N_minus_T",
            "IT_minus_SN",
            "min_region",
            "max_region",
        ]
    ]

    return df_clustered, tabela_perfis, modelo_final


def characterize_profiles(df_clustered: pd.DataFrame) -> pd.DataFrame:
    df_age = df_clustered[df_clustered["Age"].between(1, 100)].copy()

    return (
        df_age.groupby("Perfil")
        .agg(
            n=("Perfil", "size"),
            idade_media=("Age", "mean"),
            idade_mediana=("Age", "median"),
            idade_dp=("Age", "std"),
            idade_min=("Age", "min"),
            idade_max=("Age", "max"),
            pct_feminino=("Gender", lambda s: (s == "F").mean() * 100),
            pct_od=("Eye", lambda s: (s == "OD").mean() * 100),
        )
        .reindex(PROFILE_ORDER)
    )
