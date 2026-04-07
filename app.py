import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("iris_random_forest_model.joblib")

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered",
)

# ----------------------------
# Header Section
# ----------------------------
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='color:#E8474C;'>🌸 Iris Species Predictor</h1>
        <p style='font-size:18px; color:#555;'>
            Enter your flower measurements to identify the Iris species using a trained Machine Learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Input Section
# ----------------------------
st.markdown("### 🌼 Enter Flower Measurements")

with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1
        )
        petal_length = st.number_input(
            "Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1
        )

    with col2:
        sepal_width = st.number_input(
            "Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1
        )
        petal_width = st.number_input(
            "Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1
        )

# ----------------------------
# Prediction Button
# ----------------------------
predict_btn = st.button("🔍 Identify Species", use_container_width=True)

# ----------------------------
# Species Mapping
# ----------------------------
species_map = {
    0: "Iris Setosa 🌱",
    1: "Iris Versicolor 🌿",
    2: "Iris Virginica 🌺",
}

# ----------------------------
# Perform Prediction
# ----------------------------
if predict_btn:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <h2 style='color:#4CAF50;'>Prediction Result</h2>
            <p style='font-size:24px; font-weight:bold;'>
                {species_map[prediction]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
