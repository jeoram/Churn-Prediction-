from pathlib import Path

from src.data_loader import ensure_data_file, load_data
from src.eda import run_eda
from src.preprocessing import prepare_training_data
from src.modeling import train_models
from src.evaluation import evaluate_models


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    data_path = ensure_data_file(data_dir)
    df = load_data(data_path)

    eda_dir = outputs_dir / "eda"
    eda_dir.mkdir(exist_ok=True)
    run_eda(df, eda_dir)

    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(df)
    models = train_models(X_train, y_train)
    evaluation = evaluate_models(models, X_test, y_test, feature_names, outputs_dir)

    print("Projet exécuté avec succès.")
    print(f"Dataset : {data_path}")
    print("Résultats disponibles dans :", outputs_dir)
    print("\nRésumé des performances :")
    for name, metrics in evaluation.items():
        print(f"- {name}: AUC={metrics['auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}, Accuracy={metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
