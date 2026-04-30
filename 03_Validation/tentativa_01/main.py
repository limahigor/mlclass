import argparse
from pathlib import Path

from t1.analysis import run_analysis
from t1.config import OUTPUT_DIR
from t1.data import get_columns, load_datasets
from t1.models import build_models
from t1.plots import generate_plots
from t1.server import send_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Tentativa 01 - análise e execução modularizada')
    parser.add_argument('--analyze', action='store_true', help='Executa análise e salva summary/cv_results')
    parser.add_argument('--graph', action='store_true', help='Gera os gráficos da tentativa 01')
    parser.add_argument('--server', action='store_true', help='Envia as predições ao servidor')
    parser.add_argument('--all', action='store_true', help='Executa análise e gráficos')
    return parser.parse_args()


def main():
    args = parse_args()
    if not any([args.analyze, args.graph, args.server, args.all]):
        args.all = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, app_df = load_datasets()
    feature_cols, numeric_cols, categorical_cols = get_columns(df)
    models = build_models(numeric_cols, categorical_cols)
    results = run_analysis(df, app_df, feature_cols, numeric_cols, models)

    if args.all or args.graph:
        generate_plots(
            df=df,
            numeric_cols=numeric_cols,
            corr=results['corr'],
            cv_df_round=results['cv_df'],
            y_test=results['y_test'],
            y_pred=results['y_pred'],
            y_values=df['type'],
            best_model_name=results['best_model_name'],
            summary=results['summary'],
        )

    if args.server:
        response_text = send_predictions(results['y_app_pred'])
        (Path(OUTPUT_DIR) / 'server_run.log').write_text(
            f' - Resposta do servidor:\n {response_text} \n', encoding='utf-8'
        )
        print(response_text)


if __name__ == '__main__':
    main()
