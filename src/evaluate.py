from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def regression_metrics(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse)
    }
