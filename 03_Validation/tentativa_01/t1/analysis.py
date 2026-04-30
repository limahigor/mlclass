import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import OUTPUT_DIR, RANDOM_STATE


def run_analysis(df, app_df, feature_cols, numeric_cols, models):
    X = df[feature_cols]
    y = df['type']

    summary = {
        'train_shape': list(df.shape),
        'app_shape': list(app_df.shape),
        'columns': df.columns.tolist(),
        'missing_values': df.isna().sum().to_dict(),
        'app_missing_values': app_df.isna().sum().to_dict(),
        'class_distribution': y.value_counts().sort_index().to_dict(),
        'numeric_summary': df[numeric_cols].describe().round(4).to_dict(),
    }

    corr = df[numeric_cols + ['type']].corr(numeric_only=True)
    summary['correlation_with_type'] = corr['type'].drop('type').sort_values(ascending=False).round(4).to_dict()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=['accuracy', 'f1_macro'],
            n_jobs=-1,
            return_train_score=False,
        )
        cv_results.append({
            'model': name,
            'accuracy_mean': float(scores['test_accuracy'].mean()),
            'accuracy_std': float(scores['test_accuracy'].std()),
            'f1_macro_mean': float(scores['test_f1_macro'].mean()),
            'f1_macro_std': float(scores['test_f1_macro'].std()),
        })

    cv_df = pd.DataFrame(cv_results).sort_values(by='accuracy_mean', ascending=False)
    summary['cv_results'] = cv_df.round(6).to_dict(orient='records')

    best_model_name = cv_df.iloc[0]['model']
    best_model = models[best_model_name]
    summary['selected_model'] = best_model_name

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_app_pred = best_model.predict(app_df[feature_cols])

    summary['holdout_accuracy'] = float(accuracy_score(y_test, y_pred))
    summary['holdout_classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    summary['holdout_confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

    if best_model_name == 'random_forest':
        prep = best_model.named_steps['prep']
        rf = best_model.named_steps['model']
        feature_names = prep.get_feature_names_out()
        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        summary['random_forest_feature_importances'] = importances.head(15).round(6).to_dict()

    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    cv_df_round = cv_df.copy()
    cv_df_round[['accuracy_mean', 'accuracy_std', 'f1_macro_mean', 'f1_macro_std']] = cv_df_round[
        ['accuracy_mean', 'accuracy_std', 'f1_macro_mean', 'f1_macro_std']
    ].round(4)
    cv_df_round.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)

    return {
        'summary': summary,
        'cv_df': cv_df_round,
        'corr': corr,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_app_pred': y_app_pred,
        'best_model_name': best_model_name,
    }
