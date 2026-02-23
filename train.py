import json
from pathlib import Path
import joblib

from src.config import load_config
from src.data import load_training_data, split_data
from src.model import build_model
from src.evaluate import regression_metrics


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("config/config.json")

    X, y = load_training_data()

    X_train, X_test, y_train, y_test = split_data(
        X, y,
        cfg["split"]["test_size"],
        cfg["split"]["random_state"]
    )

    model = build_model(cfg["model"]["params"])
    model.fit(X_train, y_train)

    metrics = regression_metrics(model, X_test, y_test)

    model_path = Path(cfg["output"]["model_dir"]) / "housing_model.joblib"
    metrics_path = Path(cfg["output"]["metrics_dir"]) / "metrics.json"

    ensure_dir(model_path.parent)
    ensure_dir(metrics_path.parent)

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Training finished")
    print(metrics)


if __name__ == "__main__":
    main()
