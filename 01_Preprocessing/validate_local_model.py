#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação offline do pipeline de pré-processamento e do modelo k-NN.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import (
    count_missing_by_column,
    coerce_and_mark_missing,
    fill_missing_with_zero,
    fit_preprocessing,
    get_feature_columns,
    get_zero_as_missing_columns,
    preprocess_train_and_app,
    transform_with_artifacts,
)


DATASET_PATH = "diabetes_dataset.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_SPLITS = 5
STABILITY_RANDOM_STATES = [11, 22, 33, 42, 55]


def get_strategy_definitions() -> List[Dict[str, object]]:
    return [
        {
            "name": "zero_fill",
            "label": "Zero fill",
            "mode": "zero_fill",
        },
        {
            "name": "current_baseline_median_standard_clip_1_99",
            "label": "Baseline atual: mediana + padronização + clipping 1%-99%",
            "mode": "artifacts",
            "imputation_strategy": "median",
            "apply_clipping": True,
            "apply_scaling": True,
            "lower_q": 0.01,
            "upper_q": 0.99,
        },
        {
            "name": "candidate_mean_standard_clip_1_99",
            "label": "Candidata: média + padronização + clipping 1%-99%",
            "mode": "artifacts",
            "imputation_strategy": "mean",
            "apply_clipping": True,
            "apply_scaling": True,
            "lower_q": 0.01,
            "upper_q": 0.99,
        },
        {
            "name": "candidate_mean_standard_no_clip",
            "label": "Candidata: média + padronização sem clipping",
            "mode": "artifacts",
            "imputation_strategy": "mean",
            "apply_clipping": False,
            "apply_scaling": True,
        },
        {
            "name": "median_standard_no_clip",
            "label": "Referência: mediana + padronização sem clipping",
            "mode": "artifacts",
            "imputation_strategy": "median",
            "apply_clipping": False,
            "apply_scaling": True,
        },
    ]


def prepare_fold_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    strategy: Dict[str, object],
) -> List[pd.DataFrame]:
    if strategy["mode"] == "zero_fill":
        train_processed = fill_missing_with_zero(train_df, feature_cols)
        valid_processed = fill_missing_with_zero(valid_df, feature_cols)
        return [train_processed, valid_processed]

    train_prepared = coerce_and_mark_missing(train_df, feature_cols)
    valid_prepared = coerce_and_mark_missing(valid_df, feature_cols)

    artifacts = fit_preprocessing(
        train_prepared,
        feature_cols,
        imputation_strategy=str(strategy.get("imputation_strategy", "mean")),
        lower_q=float(strategy.get("lower_q", 0.01)),
        upper_q=float(strategy.get("upper_q", 0.99)),
        apply_clipping=bool(strategy.get("apply_clipping", True)),
        apply_scaling=bool(strategy.get("apply_scaling", True)),
    )

    train_processed = transform_with_artifacts(train_prepared, artifacts, feature_cols)
    valid_processed = transform_with_artifacts(valid_prepared, artifacts, feature_cols)

    return [train_processed, valid_processed]


def evaluate_strategy_once(
    dataframe: pd.DataFrame,
    feature_cols: List[str],
    strategy: Dict[str, object],
    n_splits: int = DEFAULT_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    metric_history = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    aggregated_confusion = np.zeros((2, 2), dtype=int)

    for train_index, valid_index in splitter.split(
        dataframe[feature_cols], dataframe["Outcome"]
    ):
        train_df = dataframe.iloc[train_index].copy()
        valid_df = dataframe.iloc[valid_index].copy()

        train_processed, valid_processed = prepare_fold_data(
            train_df, valid_df, feature_cols, strategy
        )

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(train_processed[feature_cols], train_df["Outcome"])

        predictions = model.predict(valid_processed[feature_cols])
        expected = valid_df["Outcome"]

        metric_history["accuracy"].append(accuracy_score(expected, predictions))
        metric_history["precision"].append(
            precision_score(expected, predictions, zero_division=0)
        )
        metric_history["recall"].append(
            recall_score(expected, predictions, zero_division=0)
        )
        metric_history["f1"].append(f1_score(expected, predictions, zero_division=0))
        aggregated_confusion += confusion_matrix(expected, predictions, labels=[0, 1])

    return {
        "accuracy": float(np.mean(metric_history["accuracy"])),
        "precision": float(np.mean(metric_history["precision"])),
        "recall": float(np.mean(metric_history["recall"])),
        "f1": float(np.mean(metric_history["f1"])),
        "confusion_matrix": aggregated_confusion,
    }


def evaluate_strategy(
    dataframe: pd.DataFrame,
    feature_cols: List[str],
    strategy: Dict[str, object],
    n_splits: int = DEFAULT_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    single_run = evaluate_strategy_once(
        dataframe,
        feature_cols,
        strategy,
        n_splits=n_splits,
        random_state=random_state,
    )
    stability_scores = [
        evaluate_strategy_once(
            dataframe,
            feature_cols,
            strategy,
            n_splits=n_splits,
            random_state=seed,
        )["accuracy"]
        for seed in STABILITY_RANDOM_STATES
    ]

    return {
        "strategy": strategy["label"],
        "accuracy": single_run["accuracy"],
        "precision": single_run["precision"],
        "recall": single_run["recall"],
        "f1": single_run["f1"],
        "confusion_matrix": single_run["confusion_matrix"],
        "stability_mean_accuracy": float(np.mean(stability_scores)),
        "stability_std_accuracy": float(np.std(stability_scores)),
        "stability_min_accuracy": float(np.min(stability_scores)),
        "stability_max_accuracy": float(np.max(stability_scores)),
    }


def print_quality_report(dataframe: pd.DataFrame, feature_cols: List[str]) -> None:
    print("\n=== Diagnóstico da base bruta ===")
    print(f"Linhas: {len(dataframe)} | Colunas: {len(dataframe.columns)}")
    print("Ausentes após leitura do CSV:")
    print(dataframe[feature_cols].isna().sum().to_string())

    zero_columns = get_zero_as_missing_columns()
    print("\nZeros suspeitos nas colunas biomédicas:")
    print((dataframe[zero_columns] == 0).sum().to_string())

    print("\nAusentes após coerção + marcação de zeros como ausentes:")
    missing_after_rules = count_missing_by_column(dataframe, feature_cols)
    print(pd.Series(missing_after_rules).to_string())

    processed_train, processed_app, _ = preprocess_train_and_app(
        dataframe,
        dataframe[feature_cols].copy(),
        feature_cols=feature_cols,
        imputation_strategy="mean",
    )

    print("\nChecagem do pipeline padrão:")
    print("NaNs restantes no treino:")
    print(processed_train[feature_cols].isna().sum().to_string())
    print("NaNs restantes na aplicação simulada:")
    print(processed_app[feature_cols].isna().sum().to_string())


def print_strategy_results(results: List[Dict[str, object]]) -> None:
    print("\n=== Comparação entre estratégias ===")
    for result in results:
        print(f"\nEstratégia: {result['strategy']}")
        print(f"Acurácia média : {result['accuracy']:.4f}")
        print(
            "Estabilidade   : "
            f"média={result['stability_mean_accuracy']:.4f} | "
            f"desvio={result['stability_std_accuracy']:.4f} | "
            f"mín={result['stability_min_accuracy']:.4f} | "
            f"máx={result['stability_max_accuracy']:.4f}"
        )
        print(f"Precisão média : {result['precision']:.4f}")
        print(f"Recall médio   : {result['recall']:.4f}")
        print(f"F1 médio       : {result['f1']:.4f}")
        print("Matriz de confusão agregada:")
        print(result["confusion_matrix"])


def main() -> None:
    feature_cols = get_feature_columns()
    data = pd.read_csv(DATASET_PATH)

    print_quality_report(data, feature_cols)

    results = [
        evaluate_strategy(data, feature_cols, strategy)
        for strategy in get_strategy_definitions()
    ]

    results = sorted(results, key=lambda item: item["accuracy"], reverse=True)
    print_strategy_results(results)


if __name__ == "__main__":
    main()
