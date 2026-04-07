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
# Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
        .main > div { padding-top: 2rem; }

        .header-box {
            background: linear-gradient(135deg, #fff5f5 0%, #fff0f6 100%);
            border: 1px solid #fcd5d6;
            border-radius: 16px;
            padding: 28px 24px 20px;
            text-align: center;
            margin-bottom: 28px;
        }
        .header-box h1 {
            color: #d63031;
            font-size: 2rem;
            margin-bottom: 8px;
        }
        .header-box p {
            color: #636e72;
            font-size: 16px;
            margin: 0;
        }

        .result-box {
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 100%);
            border: 1px solid #b2dfdb;
            border-radius: 16px;
            padding: 28px;
            text-align: center;
            margin-top: 8px;
        }
        .result-label {
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #81c784;
            margin-bottom: 8px;
        }
        .result-species {
            font-size: 26px;
            font-weight: 700;
            color: #2e7d32;
            margin: 0;
        }

        div[data-testid="stNumberInput"] label {
            font-weight: 500;
            color: #444;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header Section
# ----------------------------
st.markdown(
    """
    <div class="header-box">
        <h1>🌸 Iris Species Predictor</h1>
        <p>Enter your flower measurements to identify the Iris species<br>using a trained Machine Learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Input Section
# ----------------------------
st.markdown("#### 🌼 Flower Measurements")

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

st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

# ----------------------------
# Prediction Button
# ----------------------------
predict_btn = st.button("🔍 Identify Species", use_container_width=True, type="primary")

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

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Predicted Species</div>
            <p class="result-species">{species_map[prediction]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )