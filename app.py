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
            padding-top: 4.5rem;
            max-width: 980px;
        }}

        .page-shell {{
            max-width: 880px;
            margin: 0 auto;
        }}

        /* ── Header ── */
        .header-wrap {{
            margin-bottom: 26px;
        }}
        .header-title-row {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 18px;
        }}
        .header-icon {{
            width: 54px;
            height: 54px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            background: rgba(200,184,154,0.08);
            border: 1px solid rgba(200,184,154,0.22);
            box-shadow: 0 12px 24px rgba(0,0,0,0.18);
        }}
        .header-title {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 3.35rem;
            font-weight: 600;
            color: #f0ece4;
            line-height: 1.05;
            margin: 0;
        }}
        .header-title em {{
            font-style: italic;
            color: #c8b89a;
        }}
        .header-subtitle {{
            font-size: 15px;
            font-weight: 400;
            color: rgba(240,236,228,0.72);
            letter-spacing: 0.01em;
            max-width: 760px;
            margin: 0 0 26px;
        }}
        .header-meta {{
            font-size: 13px;
            font-weight: 300;
            color: rgba(240,236,228,0.52);
            letter-spacing: 0.08em;
            margin-bottom: 18px;
        }}

        .field-grid {{
            margin-bottom: 14px;
        }}

        .field-group {{
            margin-bottom: 18px;
        }}

        .field-label {{
            font-family: 'Jost', sans-serif;
            font-size: 11px;
            font-weight: 500;
            color: #c8b89a;
            margin-bottom: 8px;
            letter-spacing: 0.04em;
        }}

        div[data-testid="stNumberInput"] label {{
            display: none !important;
        }}
        div[data-testid="stNumberInput"] {{
            background: rgba(18, 22, 30, 0.88);
            border: 1px solid rgba(200, 184, 154, 0.14);
            border-radius: 14px;
            padding: 2px 6px;
            backdrop-filter: blur(10px);
        }}
        div[data-testid="stNumberInput"] > div {{
            background: transparent !important;
            border: none !important;
        }}
        div[data-testid="stNumberInput"] input {{
            background: transparent !important;
            border: none !important;
            color: #f0ece4 !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-size: 2rem !important;
            font-weight: 400 !important;
            text-align: left !important;
            padding: 8px 12px !important;
            box-shadow: none !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background: rgba(200,184,154,0.04) !important;
            border: 1px solid rgba(200,184,154,0.16) !important;
            border-radius: 8px !important;
            color: #c8b89a !important;
            width: 40px !important;
            height: 40px !important;
        }}
        div[data-testid="stNumberInput"] button:hover {{
            background: rgba(200,184,154,0.18) !important;
        }}

        div[data-testid="column"] {{
            padding: 0 10px !important;
        }}

        /* ── Identify button ── */
        div[data-testid="stButton"] > button {{
            background: transparent !important;
            border: 1px solid rgba(200,184,154,0.42) !important;
            border-radius: 12px !important;
            color: #f0ece4 !important;
            font-family: 'Jost', sans-serif !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.22em !important;
            text-transform: uppercase !important;
            padding: 15px 24px !important;
            transition: background 0.2s, border-color 0.2s;
            width: auto !important;
            min-width: 270px;
        }}
        div[data-testid="stButton"] > button:hover {{
            background: rgba(200,184,154,0.1) !important;
            border-color: rgba(200,184,154,0.65) !important;
        }}

        /* ── Result box ── */
        .result-box {{
            background: rgba(18, 22, 30, 0.82);
            border: 1px solid rgba(200, 184, 154, 0.25);
            border-radius: 18px;
            padding: 32px;
            text-align: center;
            margin-top: 24px;
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
            font-size: 14px;
            font-weight: 400;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(240,236,228,0.72);
            margin: 0;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        @media (max-width: 768px) {{
            .main .block-container {{
                padding-top: 2.4rem;
            }}
            .header-title {{
                font-size: 2.5rem;
            }}
            .header-title-row {{
                align-items: flex-start;
            }}
            .header-subtitle {{
                font-size: 14px;
            }}
            div[data-testid="stButton"] > button {{
                width: 100% !important;
                min-width: 0;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="page-shell">
        <div class="header-wrap">
            <div class="header-title-row">
                <div class="header-icon">✿</div>
                <h1 class="header-title">Identify the <em>Iris</em></h1>
            </div>
            <p class="header-subtitle">Enter your measurements below to identify the iris species using a trained machine learning model.</p>
            <div class="header-meta">Random Forest · 100 Trees · 70/30 Split</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Input Grid
# ----------------------------
st.markdown('<div class="page-shell"><div class="field-grid">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="field-group"><div class="field-label">Sepal Length (cm)</div>', unsafe_allow_html=True)
    sepal_length = st.number_input(
        "Sepal Length", min_value=0.0, max_value=10.0, value=5.8,
        step=0.1, format="%.1f", key="sl"
    )
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="field-group"><div class="field-label">Sepal Width (cm)</div>', unsafe_allow_html=True)
    sepal_width = st.number_input(
        "Sepal Width", min_value=0.0, max_value=10.0, value=3.0,
        step=0.1, format="%.1f", key="sw"
    )
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="field-group"><div class="field-label">Petal Length (cm)</div>', unsafe_allow_html=True)
    petal_length = st.number_input(
        "Petal Length", min_value=0.0, max_value=10.0, value=3.8,
        step=0.1, format="%.1f", key="pl"
    )
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="field-group"><div class="field-label">Petal Width (cm)</div>', unsafe_allow_html=True)
    petal_width = st.number_input(
        "Petal Width", min_value=0.0, max_value=10.0, value=1.2,
        step=0.1, format="%.1f", key="pw"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------------------
# Predict Button
# ----------------------------
st.markdown('<div class="page-shell">', unsafe_allow_html=True)
predict_btn = st.button("Identify Species", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

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
        <div class="page-shell">
            <div class="result-box">
                <div class="result-label">Predicted Species</div>
                <p class="result-species" style="font-size:64px; line-height:1.05; margin:10px 0 16px;">{species_map[prediction]}</p>
                <p class="result-confidence">Confidence: {confidence:.1f}%</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
