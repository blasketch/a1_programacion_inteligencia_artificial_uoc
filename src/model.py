from sklearn.ensemble import RandomForestRegressor


def build_model(params: dict):
    return RandomForestRegressor(**params)
