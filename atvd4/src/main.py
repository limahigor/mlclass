from pathlib import Path

from src.clustering import (
    characterize_profiles,
    evaluate_clusters,
    fit_final_model,
    scale_features,
)
from src.data import load_dataset, clean_dataset
from src.reporting import build_summary_text, export_tables
from src.visualization import (
    plot_elbow,
    plot_pca,
    plot_profile_heatmap,
    plot_profile_lines,
    plot_silhouette,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "RTVue_20221110_MLClass.xlsx"
OUTPUT_DIR = BASE_DIR / "artefatos"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_FILE)
    df_clean = clean_dataset(df)

    _, scaler, X_scaled = scale_features(df_clean)
    metricas = evaluate_clusters(X_scaled)

    df_clustered, tabela_perfis, _ = fit_final_model(
        df_clean, scaler, X_scaled, k_final=5
    )
    caracterizacao = characterize_profiles(df_clustered)

    plot_elbow(metricas, OUTPUT_DIR)
    plot_silhouette(metricas, OUTPUT_DIR)
    plot_profile_lines(tabela_perfis, OUTPUT_DIR)
    plot_profile_heatmap(tabela_perfis, OUTPUT_DIR)

    pca = plot_pca(df_clustered, X_scaled, OUTPUT_DIR)

    export_tables(df_clustered, tabela_perfis, caracterizacao, OUTPUT_DIR)
    resumo = build_summary_text(df, df_clustered, metricas, tabela_perfis, pca)
    (OUTPUT_DIR / "resumo_execucao.txt").write_text(resumo)
    print(resumo)


if __name__ == "__main__":
    main()
