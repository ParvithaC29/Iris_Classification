# data.py
from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names

def get_dataframe():
    X, y, feature_names, target_names = load_data()
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    return df
