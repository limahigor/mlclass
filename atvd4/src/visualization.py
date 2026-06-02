from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.clustering import PROFILE_ORDER
from src.data import FEATURES


def plot_elbow(metricas, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(metricas["k"], metricas["inertia"], marker="o")
    plt.title("Método do cotovelo")

    plt.xlabel("Número de clusters")
    plt.ylabel("Inércia")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metodo_cotovelo.png", dpi=200)
    plt.close()


def plot_silhouette(metricas, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(metricas["k"], metricas["silhouette"], marker="o")
    plt.title("Silhouette por número de clusters")

    plt.xlabel("Número de clusters")
    plt.ylabel("Silhouette")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_por_k.png", dpi=200)
    plt.close()


def plot_profile_lines(tabela_perfis, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))

    for perfil in PROFILE_ORDER:
        plt.plot(
            FEATURES, tabela_perfis.loc[perfil, FEATURES], marker="o", label=perfil
        )

    plt.title("Perfis médios dos mapas epiteliais")
    plt.xlabel("Região do mapa epitelial")
    plt.ylabel("Espessura média")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "perfis_medios_mapa_epitelial.png", dpi=200)
    plt.close()


def plot_profile_heatmap(tabela_perfis, output_dir: Path) -> None:
    plt.figure(figsize=(11, 4.5))

    matriz = tabela_perfis[FEATURES].values

    plt.imshow(matriz, aspect="auto")
    plt.xticks(range(len(FEATURES)), FEATURES)
    plt.yticks(range(len(PROFILE_ORDER)), PROFILE_ORDER)

    plt.title("Mapa de calor dos centróides por perfil")
    plt.colorbar(label="Espessura média")

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            plt.text(j, i, f"{matriz[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_perfis_mapa_epitelial.png", dpi=200)
    plt.close()


def plot_pca(df_clustered, X_scaled, output_dir: Path):
    pca = PCA(n_components=2, random_state=42)

    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for perfil in PROFILE_ORDER:
        mask = df_clustered["Perfil"] == perfil
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, alpha=0.35, label=perfil)

    plt.title("Projeção PCA dos mapas epiteliais com os perfis encontrados")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% da variância)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% da variância)")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_perfis_mapa_epitelial.png", dpi=200)
    plt.close()

    return pca
