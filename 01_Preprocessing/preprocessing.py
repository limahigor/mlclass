#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funções de pré-processamento para o dataset de diabetes.

O pipeline padrão foi desenhado para o k-NN usado no projeto:
1. converter strings vazias para ausentes;
2. tratar zero como ausente apenas nas colunas biomédicas problemáticas;
3. imputar ausentes com a média calculada no conjunto de treino;
4. aplicar clipping por quantis para reduzir o impacto de extremos;
5. padronizar as features via z-score.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING_COLUMNS = {
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
}


def get_feature_columns() -> List[str]:
    """Retorna a lista oficial de colunas preditoras."""
    return FEATURE_COLUMNS.copy()


def get_zero_as_missing_columns() -> List[str]:
    """Retorna as colunas onde zero é tratado como dado ausente."""
    return [column for column in FEATURE_COLUMNS if column in ZERO_AS_MISSING_COLUMNS]


def _ensure_feature_columns(dataframe: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    missing_columns = [column for column in feature_cols if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(
            "As seguintes colunas não foram encontradas no DataFrame: "
            + ", ".join(missing_columns)
        )


def _bounds_to_dict(
    lower_bounds: pd.Series, upper_bounds: pd.Series, feature_cols: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    return {
        column: {
            "lower": float(lower_bounds[column]),
            "upper": float(upper_bounds[column]),
        }
        for column in feature_cols
    }


def _build_fill_values(
    dataframe: pd.DataFrame,
    feature_cols: Sequence[str],
    imputation_strategy: str,
) -> pd.Series:
    if imputation_strategy == "mean":
        return dataframe.loc[:, feature_cols].mean()
    if imputation_strategy == "median":
        return dataframe.loc[:, feature_cols].median()
    if imputation_strategy == "zero":
        return pd.Series(0.0, index=feature_cols, dtype="float64")

    raise ValueError(
        "Estratégia de imputação inválida. Use 'mean', 'median' ou 'zero'."
    )


def coerce_and_mark_missing(
    dataframe: pd.DataFrame, feature_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Garante colunas numéricas e marca ausentes.

    Strings vazias passam a ser NaN via ``to_numeric(errors="coerce")``.
    Nas colunas biomédicas selecionadas, o valor 0 também passa a ser tratado
    como ausente.
    """
    _ensure_feature_columns(dataframe, feature_cols)

    transformed = dataframe.copy()

    for column in feature_cols:
        # Mantemos as features em float para evitar conflitos de dtype nas etapas
        # de imputação, clipping e padronização em versões recentes do pandas.
        transformed[column] = pd.to_numeric(
            transformed[column], errors="coerce"
        ).astype("float64")
        if column in ZERO_AS_MISSING_COLUMNS:
            transformed[column] = transformed[column].mask(transformed[column] == 0)

    return transformed


def fit_preprocessing(
    dataframe: pd.DataFrame,
    feature_cols: Sequence[str],
    imputation_strategy: str = "mean",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    apply_clipping: bool = True,
    apply_scaling: bool = True,
) -> Dict[str, object]:
    """
    Ajusta os artefatos do pré-processamento usando apenas o conjunto de treino.
    """
    if not 0 <= lower_q <= upper_q <= 1:
        raise ValueError("Os quantis devem respeitar 0 <= lower_q <= upper_q <= 1.")

    prepared = coerce_and_mark_missing(dataframe, feature_cols)

    fill_values = _build_fill_values(prepared, feature_cols, imputation_strategy)
    filled = prepared.loc[:, feature_cols].fillna(fill_values)

    if apply_clipping:
        lower_bounds = filled.quantile(lower_q)
        upper_bounds = filled.quantile(upper_q)
        clipped = filled.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    else:
        lower_bounds = filled.min()
        upper_bounds = filled.max()
        clipped = filled

    if apply_scaling:
        scale_mean = clipped.mean()
        scale_std = clipped.std(ddof=0).replace(0, 1.0)
    else:
        scale_mean = pd.Series(0.0, index=feature_cols)
        scale_std = pd.Series(1.0, index=feature_cols)

    return {
        "feature_cols": list(feature_cols),
        "zero_as_missing_cols": get_zero_as_missing_columns(),
        "imputation_strategy": imputation_strategy,
        "fill_values_by_col": {column: float(fill_values[column]) for column in feature_cols},
        "clip_bounds_by_col": _bounds_to_dict(lower_bounds, upper_bounds, feature_cols),
        "scale_mean_by_col": {column: float(scale_mean[column]) for column in feature_cols},
        "scale_std_by_col": {column: float(scale_std[column]) for column in feature_cols},
        "lower_q": lower_q,
        "upper_q": upper_q,
        "apply_clipping": apply_clipping,
        "apply_scaling": apply_scaling,
    }


def transform_with_artifacts(
    dataframe: pd.DataFrame,
    artifacts: Dict[str, object],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Aplica imputação, clipping e padronização com artefatos do treino."""
    prepared = coerce_and_mark_missing(dataframe, feature_cols)
    transformed = prepared.copy()
    transformed = transformed.astype({column: "float64" for column in feature_cols})

    fill_values_by_col = pd.Series(artifacts["fill_values_by_col"])
    scale_mean_by_col = pd.Series(artifacts["scale_mean_by_col"])
    scale_std_by_col = pd.Series(artifacts["scale_std_by_col"])

    transformed.loc[:, feature_cols] = transformed.loc[:, feature_cols].fillna(
        fill_values_by_col
    )

    if artifacts.get("apply_clipping", True):
        lower_bounds = pd.Series(
            {
                column: artifacts["clip_bounds_by_col"][column]["lower"]
                for column in feature_cols
            }
        )
        upper_bounds = pd.Series(
            {
                column: artifacts["clip_bounds_by_col"][column]["upper"]
                for column in feature_cols
            }
        )
        transformed.loc[:, feature_cols] = transformed.loc[:, feature_cols].clip(
            lower=lower_bounds,
            upper=upper_bounds,
            axis=1,
        )

    if artifacts.get("apply_scaling", True):
        transformed.loc[:, feature_cols] = (
            transformed.loc[:, feature_cols] - scale_mean_by_col
        ) / scale_std_by_col

    return transformed


def preprocess_train_and_app(
    train_df: pd.DataFrame,
    app_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    imputation_strategy: str = "mean",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Pré-processa treino e aplicação em memória usando artefatos do treino.
    """
    feature_cols = list(feature_cols or get_feature_columns())

    train_prepared = coerce_and_mark_missing(train_df, feature_cols)
    app_prepared = coerce_and_mark_missing(app_df, feature_cols)

    artifacts = fit_preprocessing(
        train_prepared,
        feature_cols,
        imputation_strategy=imputation_strategy,
        lower_q=lower_q,
        upper_q=upper_q,
        apply_clipping=True,
        apply_scaling=True,
    )

    train_transformed = transform_with_artifacts(train_prepared, artifacts, feature_cols)
    app_transformed = transform_with_artifacts(app_prepared, artifacts, feature_cols)

    return train_transformed, app_transformed, artifacts


def fill_missing_with_zero(
    dataframe: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Estratégia experimental: converte ausentes para zero.
    """
    feature_cols = list(feature_cols or get_feature_columns())
    prepared = coerce_and_mark_missing(dataframe, feature_cols)
    transformed = prepared.copy()
    transformed.loc[:, feature_cols] = transformed.loc[:, feature_cols].fillna(0)
    return transformed


def clip_by_quantiles(
    dataframe: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    bounds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Estratégia experimental: aplica clipping por quantis configuráveis.
    """
    feature_cols = list(feature_cols or get_feature_columns())
    prepared = coerce_and_mark_missing(dataframe, feature_cols)
    clipped = prepared.copy()

    working = clipped.loc[:, feature_cols].fillna(clipped.loc[:, feature_cols].median())

    if bounds is None:
        lower_bounds = working.quantile(lower_q)
        upper_bounds = working.quantile(upper_q)
        bounds = _bounds_to_dict(lower_bounds, upper_bounds, feature_cols)
    else:
        lower_bounds = pd.Series(
            {column: bounds[column]["lower"] for column in feature_cols}
        )
        upper_bounds = pd.Series(
            {column: bounds[column]["upper"] for column in feature_cols}
        )

    clipped.loc[:, feature_cols] = working.clip(
        lower=lower_bounds,
        upper=upper_bounds,
        axis=1,
    )

    return clipped, bounds


def filter_rows_by_quantiles(
    dataframe: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    bounds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Estratégia experimental: remove linhas fora da faixa de quantis.
    """
    feature_cols = list(feature_cols or get_feature_columns())
    prepared = coerce_and_mark_missing(dataframe, feature_cols)
    filtered = prepared.copy()

    working = filtered.loc[:, feature_cols].fillna(filtered.loc[:, feature_cols].median())

    if bounds is None:
        lower_bounds = working.quantile(lower_q)
        upper_bounds = working.quantile(upper_q)
        bounds = _bounds_to_dict(lower_bounds, upper_bounds, feature_cols)
    else:
        lower_bounds = pd.Series(
            {column: bounds[column]["lower"] for column in feature_cols}
        )
        upper_bounds = pd.Series(
            {column: bounds[column]["upper"] for column in feature_cols}
        )

    mask = pd.Series(True, index=working.index)
    for column in feature_cols:
        mask &= working[column].between(lower_bounds[column], upper_bounds[column])

    filtered = filtered.loc[mask].copy()
    filtered.loc[:, feature_cols] = working.loc[mask]

    return filtered, bounds


def count_missing_by_column(
    dataframe: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> Dict[str, int]:
    """Conta ausentes após coerção e marcação de zeros suspeitos."""
    selected_columns = list(columns or get_feature_columns())
    prepared = coerce_and_mark_missing(dataframe, selected_columns)
    return {
        column: int(prepared[column].isna().sum())
        for column in selected_columns
    }
