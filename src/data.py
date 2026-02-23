import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_training_data():
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y


def split_data(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_new_data(path: str):
    return pd.read_csv(path)
