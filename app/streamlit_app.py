import streamlit as st
import numpy as np
import sys
import os

# ✅ Add project root and src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# ✅ Import functions from src
from src.predict import predict_species
from src.train import MODEL_PATH, train_and_save_model

# ✅ Train model if it does not exist
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# ✅ Streamlit App Layout
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

st.title("🌸 Iris Species Predictor")
st.markdown("This app predicts the **species of an Iris flower** based on its measurements.")

# ✅ Sliders for input
st.subheader("Enter Flower Measurements:")
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.35)
petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

# ✅ Prediction Button
if st.button("🔍 Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    species, probs = predict_species(features)

    # ✅ Show Prediction Result
    st.success(f"🌟 Predicted Species: **{species.capitalize()}**")

    # ✅ Display Probabilities
    st.subheader("Prediction Probabilities:")
    for sp, p in probs.items():
        st.write(f"{sp.capitalize()}: {p:.2%}")

    # ✅ Optional: Probability Bar Chart
    st.bar_chart(list(probs.values()), height=250)

