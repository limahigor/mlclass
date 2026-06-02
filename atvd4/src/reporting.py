from pathlib import Path

from src.clustering import PROFILE_ORDER
from src.data import FEATURES


def export_tables(
    df_clustered, tabela_perfis, caracterizacao, output_dir: Path
) -> None:
    df_clustered[
        ["Index", "pID", "Age", "Gender", "Eye"] + FEATURES + ["Perfil"]
    ].to_csv(
        output_dir / "base_mapa_epitelial_com_perfis.csv",
        index=False,
    )
    tabela_perfis.round(2).to_csv(output_dir / "resumo_perfis_mapa_epitelial.csv")
    caracterizacao.round(2).to_csv(
        output_dir / "caracterizacao_perfis_mapa_epitelial.csv"
    )


def build_summary_text(df_original, df_clustered, metricas, tabela_perfis, pca) -> str:
    linhas = []

    linhas.append(f"Registros originais: {len(df_original)}")
    linhas.append(f"Registros usados no clustering: {len(df_clustered)}")
    linhas.append(f"Registros removidos: {len(df_original) - len(df_clustered)}")
    linhas.append("")
    linhas.append("Métricas por k:")
    linhas.append(metricas.round(4).to_string(index=False))
    linhas.append("")
    linhas.append("Contagem por perfil:")
    linhas.append(
        df_clustered["Perfil"].value_counts().reindex(PROFILE_ORDER).to_string()
    )
    linhas.append("")
    linhas.append("Resumo dos perfis:")
    linhas.append(
        tabela_perfis[["n", "pct", "mean_epi", "range_epi", "min_region", "max_region"]]
        .round(2)
        .to_string()
    )
    linhas.append("")
    linhas.append(f"PCA explained variance: {pca.explained_variance_ratio_}")

    return "\n".join(linhas)
