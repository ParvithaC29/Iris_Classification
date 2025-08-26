# predict.py
import joblib
import numpy as np
from data import load_data

MODEL_PATH = "models/iris_model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_species(features):
    model = load_model()
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    _, _, _, target_names = load_data()
    return target_names[pred], dict(zip(target_names, probs))
