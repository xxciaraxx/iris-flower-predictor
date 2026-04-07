import base64
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Iris Species Identifier", page_icon="iris-icon.jpg", layout="centered")

model = joblib.load("iris_random_forest_model.joblib")
bg_path = Path("iris-bg2.jpg")
icon_path = Path("iris-icon.jpg")
bg_css = (
    f"url('data:image/jpeg;base64,{base64.b64encode(bg_path.read_bytes()).decode()}')"
    if bg_path.exists()
    else "linear-gradient(160deg,#2c2a35 0%,#1a2a1e 40%,#0d1a2e 100%)"
)
icon_src = f"data:image/jpeg;base64,{base64.b64encode(icon_path.read_bytes()).decode()}" if icon_path.exists() else ""

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Jost:wght@300;400;500;600&display=swap');
        .stApp{{background:{bg_css} center/cover fixed;font-family:'Jost',sans-serif;}}
        .stApp::before{{content:'';position:fixed;inset:0;background:rgba(6,10,16,.74);pointer-events:none;z-index:0;}}
        .main .block-container{{position:relative;z-index:1;max-width:900px;padding-top:4rem;}}
        .shell{{max-width:820px;margin:0 auto;}}
        .hero{{margin-bottom:1.6rem;color:#f0ece4;}}
        .hero-top{{display:flex;align-items:center;gap:14px;margin-bottom:14px;}}
        .hero-icon{{width:48px;height:48px;border-radius:14px;object-fit:cover;background:rgba(200,184,154,.08);border:1px solid rgba(200,184,154,.18);}}
        .hero h1{{margin:0;font:600 3rem/1.05 'Cormorant Garamond',serif;color:#f0ece4;}}
        .hero em{{color:#c8b89a;font-style:italic;}}
        .hero p{{margin:0 0 12px;color:rgba(240,236,228,.72);font-size:15px;}}
        .meta{{color:rgba(240,236,228,.52);font-size:12px;letter-spacing:.08em;}}
        .label{{margin:0 0 6px;color:#c8b89a;font-size:11px;letter-spacing:.04em;}}
        div[data-testid="stNumberInput"] label{{display:none!important;}}
        div[data-testid="stNumberInput"]{{background:rgba(18,22,30,.88);border:1px solid rgba(200,184,154,.14);border-radius:12px;padding:2px 6px;}}
        div[data-testid="stNumberInput"]>div{{background:transparent!important;border:none!important;}}
        div[data-testid="stNumberInput"] input{{background:transparent!important;border:none!important;box-shadow:none!important;color:#f0ece4!important;font:400 1.8rem 'Cormorant Garamond',serif!important;padding:8px 12px!important;}}
        div[data-testid="stNumberInput"] button{{background:rgba(200,184,154,.04)!important;border:1px solid rgba(200,184,154,.16)!important;border-radius:8px!important;color:#c8b89a!important;}}
        div[data-testid="column"]{{padding:0 8px!important;}}
        div[data-testid="stButton"]>button{{background:transparent!important;border:1px solid rgba(200,184,154,.42)!important;border-radius:10px!important;color:#f0ece4!important;font-size:12px!important;font-weight:600!important;letter-spacing:.22em!important;text-transform:uppercase!important;padding:14px 22px!important;min-width:250px;}}
        .result{{margin-top:20px;padding:28px;text-align:center;background:rgba(18,22,30,.82);border:1px solid rgba(200,184,154,.25);border-radius:14px;}}
        .result small{{display:block;margin-bottom:10px;color:#c8b89a;font-size:16px;letter-spacing:.22em;text-transform:uppercase;}}
        .species{{display:block;margin:10px 0 16px;color:#f0ece4;font:600 52px/1.05 'Cormorant Garamond',serif;font-style:italic;}}
        .confidence{{margin:0;color:rgba(240,236,228,.72);font-size:13px;letter-spacing:.08em;text-transform:uppercase;}}
        #MainMenu,footer,header{{visibility:hidden;}}
        @media (max-width:768px){{
            .main .block-container{{padding-top:2.4rem;}}
            .hero h1{{font-size:2.3rem;}}
            .hero-top{{align-items:flex-start;}}
            div[data-testid="stButton"]>button{{width:100%!important;min-width:0;}}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="shell hero">
        <div class="hero-top">
            <img src="{icon_src}" class="hero-icon" alt="Iris icon">
            <h1>Identify the <em>Iris</em> Species</h1>
        </div>
        <p>Enter the measurements below.</p>
        <div class="meta">Random Forest · 100 Trees · 70/30 Split</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="shell">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="label">Sepal Length (cm)</div>', unsafe_allow_html=True)
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.8, step=0.1, format="%.1f")
with col2:
    st.markdown('<div class="label">Sepal Width (cm)</div>', unsafe_allow_html=True)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1, format="%.1f")

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="label">Petal Length (cm)</div>', unsafe_allow_html=True)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=3.8, step=0.1, format="%.1f")
with col4:
    st.markdown('<div class="label">Petal Width (cm)</div>', unsafe_allow_html=True)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.1f")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="shell">', unsafe_allow_html=True)
predict_btn = st.button("Identify Species", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

species_map = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}

if predict_btn:
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(x)[0]
    confidence = max(model.predict_proba(x)[0]) * 100
    st.markdown(
        f"""
        <div class="shell">
            <div class="result">
                <small>Predicted Species</small>
                <span class="species">{species_map[prediction]}</span>
                <p class="confidence">Confidence: {confidence:.1f}%</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
