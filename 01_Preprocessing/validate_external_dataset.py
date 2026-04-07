#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação opcional usando uma cópia pública equivalente do dataset Pima.

Se não houver internet ou a fonte externa estiver indisponível, o script
encerra de forma amigável sem impactar o fluxo principal do projeto.
"""

import argparse
from typing import Optional

import pandas as pd

from preprocessing import get_feature_columns
from validate_local_model import (
    evaluate_strategy,
    get_strategy_definitions,
    print_strategy_results,
)


DEFAULT_EXTERNAL_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

CANONICAL_COLUMNS = get_feature_columns() + ["Outcome"]
COLUMN_ALIASES = {
    "pregnancies": "Pregnancies",
    "preg": "Pregnancies",
    "glucose": "Glucose",
    "plas": "Glucose",
    "bloodpressure": "BloodPressure",
    "pres": "BloodPressure",
    "skinthickness": "SkinThickness",
    "skin": "SkinThickness",
    "insulin": "Insulin",
    "test": "Insulin",
    "bmi": "BMI",
    "mass": "BMI",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
    "outcome": "Outcome",
    "class": "Outcome",
}


def _column_name_looks_numeric(column_name: object) -> bool:
    try:
        float(str(column_name).strip())
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_headerless_dataset(dataframe: pd.DataFrame) -> bool:
    return len(dataframe.columns) == len(CANONICAL_COLUMNS) and all(
        _column_name_looks_numeric(column) for column in dataframe.columns
    )


def normalize_external_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    renamed_columns = {}

    for column in normalized.columns:
        key = str(column).strip().lower()
        key = key.replace(" ", "").replace("_", "").replace("-", "")
        renamed_columns[column] = COLUMN_ALIASES.get(key, column)

    normalized = normalized.rename(columns=renamed_columns)

    if not set(CANONICAL_COLUMNS).issubset(normalized.columns) and len(normalized.columns) == 9:
        normalized = normalized.copy()
        normalized.columns = CANONICAL_COLUMNS

    missing_columns = [column for column in CANONICAL_COLUMNS if column not in normalized.columns]
    if missing_columns:
        raise ValueError(
            "Não foi possível mapear o schema externo para o schema esperado. "
            f"Colunas ausentes: {', '.join(missing_columns)}"
        )

    normalized = normalized.loc[:, CANONICAL_COLUMNS].copy()
    for column in CANONICAL_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["Outcome"]).reset_index(drop=True)
    return normalized


def download_external_dataset(url: str) -> Optional[pd.DataFrame]:
    print(f"\n - Tentando baixar dataset externo de: {url}")

    try:
        dataframe = pd.read_csv(url)
        if _looks_like_headerless_dataset(dataframe):
            raise ValueError(
                "O arquivo externo parece não ter cabeçalho; repetindo leitura com header=None."
            )
        dataframe = normalize_external_columns(dataframe)
        print(" - Download concluído usando leitura com cabeçalho.")
        return dataframe
    except Exception as exc:
        print(f" - Leitura com cabeçalho não serviu para esta fonte: {exc}")
        pass

    try:
        dataframe = pd.read_csv(url, header=None)
        dataframe = normalize_external_columns(dataframe)
        print(" - Download concluído usando leitura sem cabeçalho.")
        return dataframe
    except Exception as exc:
        print(" - Não foi possível baixar ou normalizar o dataset externo.")
        print(f" - Motivo: {exc}")
        print(" - O fluxo principal continua funcionando normalmente sem essa etapa.")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida o pipeline usando uma base pública equivalente."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_EXTERNAL_URL,
        help="URL pública do CSV externo a ser avaliado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_data = download_external_dataset(args.url)

    if external_data is None:
        return

    feature_cols = get_feature_columns()
    print(
        f" - Base externa pronta para validação: {len(external_data)} linhas e "
        f"{len(external_data.columns)} colunas."
    )

    results = [
        evaluate_strategy(external_data, feature_cols, strategy)
        for strategy in get_strategy_definitions()
    ]
    results = sorted(results, key=lambda item: item["accuracy"], reverse=True)

    print("\n=== Comparação na base externa equivalente ===")
    print_strategy_results(results)


if __name__ == "__main__":
    main()
