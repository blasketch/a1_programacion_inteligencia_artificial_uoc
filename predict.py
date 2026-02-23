import joblib
import pandas as pd
from pathlib import Path


def main():
    model = joblib.load("outputs/models/housing_model.joblib")

    new_data = pd.read_csv("data/new_houses.csv")

    predictions = model.predict(new_data)

    new_data["PredictedPrice"] = predictions

    out_path = Path("outputs/predictions/predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_data.to_csv(out_path, index=False)

    print("Predictions saved to:", out_path)


if __name__ == "__main__":
    main()
