# train.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data import load_data

MODEL_PATH = "models/iris_model.joblib"

def train_and_save_model():
    X, y, _, _ = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipe.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
