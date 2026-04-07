import streamlit as st
import numpy as np
import joblib
import base64
from pathlib import Path
# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("iris_random_forest_model.joblib")

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Iris Species Identifier",
    page_icon="iris-icon.jpg",
    layout="centered",
)

BG_PATH = Path("iris-bg2.jpg")

if BG_PATH.exists():
    bg_data = base64.b64encode(BG_PATH.read_bytes()).decode()
    bg_css = f"url('data:image/jpeg;base64,{bg_data}')"
else:
    bg_css = "linear-gradient(160deg, #2c2a35 0%, #1a2a1e 40%, #0d1a2e 100%)"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Jost:wght@300;400;500;600&display=swap');

        /* ── Full-page background ── */
        .stApp {{
            background: {bg_css};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Jost', sans-serif;
        }}

        /* Dim overlay so text stays readable over a photo */
        .stApp::before {{
            content: '';
            position: fixed;
            inset: 0;
            background: rgba(6, 10, 16, 0.74);
            pointer-events: none;
            z-index: 0;
        }}

        /* Pull Streamlit content above overlay */
        .main .block-container {{
            position: relative;
            z-index: 1;
            padding-top: 3rem;
            max-width: 700px;
        }}

        /* ── Header ── */
        .header-wrap {{
            text-align: center;
            margin-bottom: 36px;
        }}
        .header-eyebrow {{
            font-family: 'Jost', sans-serif;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            color: #c8b89a;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .header-eyebrow::before,
        .header-eyebrow::after {{
            content: '';
            display: block;
            width: 40px;
            height: 1px;
            background: #c8b89a;
            opacity: 0.6;
        }}
        .header-title {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 3rem;
            font-weight: 400;
            color: #f0ece4;
            line-height: 1.15;
            margin: 0 0 10px;
        }}
        .header-title em {{
            font-style: italic;
            color: #c8b89a;
        }}
        .header-subtitle {{
            font-size: 13px;
            font-weight: 300;
            color: rgba(240,236,228,0.5);
            letter-spacing: 0.05em;
        }}
        .header-subtitle span {{
            margin: 0 6px;
            opacity: 0.4;
        }}

        /* ── Input card ── */
        .card {{
            background: rgba(18, 22, 30, 0.82);
            border: 1px solid rgba(200, 184, 154, 0.18);
            border-radius: 12px;
            overflow: hidden;
            backdrop-filter: blur(12px);
            margin-bottom: 14px;
        }}

        div[data-testid="stNumberInput"] label {{
            font-family: 'Jost', sans-serif !important;
            font-size: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            color: #c8b89a !important;
        }}
        div[data-testid="stNumberInput"] input {{
            background: transparent !important;
            border: none !important;
            color: #f0ece4 !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-size: 2.2rem !important;
            font-weight: 400 !important;
            text-align: center !important;
            padding: 4px 0 !important;
            box-shadow: none !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background: rgba(200,184,154,0.08) !important;
            border: 1px solid rgba(200,184,154,0.2) !important;
            border-radius: 6px !important;
            color: #c8b89a !important;
        }}
        div[data-testid="stNumberInput"] button:hover {{
            background: rgba(200,184,154,0.18) !important;
        }}

        div[data-testid="column"] {{
            border-right: 1px solid rgba(200,184,154,0.1);
            padding: 20px 24px !important;
        }}
        div[data-testid="column"]:last-child {{
            border-right: none;
        }}

        .row-divider {{
            border: none;
            border-top: 1px solid rgba(200,184,154,0.1);
            margin: 0;
        }}

        /* ── Identify button ── */
        div[data-testid="stButton"] > button {{
            background: transparent !important;
            border: 1px solid rgba(200,184,154,0.35) !important;
            border-radius: 8px !important;
            color: #f0ece4 !important;
            font-family: 'Jost', sans-serif !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            letter-spacing: 0.22em !important;
            text-transform: uppercase !important;
            padding: 16px !important;
            transition: background 0.2s, border-color 0.2s;
        }}
        div[data-testid="stButton"] > button:hover {{
            background: rgba(200,184,154,0.1) !important;
            border-color: rgba(200,184,154,0.65) !important;
        }}

        /* ── Result box ── */
        .result-box {{
            background: rgba(18, 22, 30, 0.82);
            border: 1px solid rgba(200, 184, 154, 0.25);
            border-radius: 12px;
            padding: 32px;
            text-align: center;
            margin-top: 14px;
            backdrop-filter: blur(12px);
        }}
        .result-label {{
            font-family: 'Jost', sans-serif;
            font-size: 20px;
            font-weight: 600;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #c8b89a;
            margin-bottom: 10px;
        }}
        .result-box .result-species {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 64px !important;
            font-weight: 600 !important;
            font-style: italic;
            color: #f0ece4 !important;
            line-height: 1.05 !important;
            margin: 10px 0 16px !important;
            display: block;
        }}
        .result-confidence {{
            font-family: 'Jost', sans-serif;
            font-size: 13px;
            font-weight: 400;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(240,236,228,0.72);
            margin: 0;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="header-wrap">
        <div class="header-eyebrow">Iris Species Predictor</div>
        <h1 class="header-title">Identify the <em>Iris</em></h1>
        <p class="header-subtitle">
            Random Forest
            <span>·</span>
            100 Trees
            <span>·</span>
            70/30 Split
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Input Card
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input(
        "• Sepal Length", min_value=0.0, max_value=10.0, value=5.8,
        step=0.1, format="%.1f", key="sl"
    )
with col2:
    sepal_width = st.number_input(
        "• Sepal Width", min_value=0.0, max_value=10.0, value=3.0,
        step=0.1, format="%.1f", key="sw"
    )

st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    petal_length = st.number_input(
        "• Petal Length", min_value=0.0, max_value=10.0, value=3.8,
        step=0.1, format="%.1f", key="pl"
    )
with col4:
    petal_width = st.number_input(
        "• Petal Width", min_value=0.0, max_value=10.0, value=1.2,
        step=0.1, format="%.1f", key="pw"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Predict Button
# ----------------------------
predict_btn = st.button("Identify Species", use_container_width=True)

# ----------------------------
# Species Mapping
# ----------------------------
species_map = {
    0: "Iris setosa",
    1: "Iris versicolor",
    2: "Iris virginica",
}

# ----------------------------
# Perform Prediction
# ----------------------------
if predict_btn:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Predicted Species</div>
            <p class="result-species" style="font-size:64px; line-height:1.05; margin:10px 0 16px;">{species_map[prediction]}</p>
            <p class="result-confidence">Confidence: {confidence:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
