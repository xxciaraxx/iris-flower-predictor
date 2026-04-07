"""
Iris Species Predictor — Streamlit App
=====================================
Matches the design of index.html with custom CSS and Streamlit layout.

Run:
    pip install streamlit scikit-learn joblib pandas
    streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path("iris_model.pkl")

# ── Custom CSS matching index.html ──
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --cream: #f7f1e8;
        --dark: #0f1c14;
        --gold: #c8a96e;
        --gold-light: #e8d5a8;
        --setosa: #7ecba1;
        --versicolor: #6ba3d6;
        --virginica: #c57ebf;
    }
    
    html, body { 
        margin: 0; padding: 0; 
        min-height: 100vh; 
        background: linear-gradient(160deg, rgba(15,28,20,0.5) 0%, rgba(15,28,20,0.9) 100%);
        background-attachment: fixed;
    }
    
    body::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url('iris-bg2.jpg');
        background-size: cover;
        background-position: center 30%;
        background-attachment: fixed;
        filter: saturate(0.7) brightness(0.32);
        z-index: -1;
    }
    
    [data-testid="stAppViewContainer"] { background: transparent !important; }
    [data-testid="stMainBlockContainer"] { padding: 3rem 1.5rem 4rem !important; position: relative; z-index: 1; }
    [data-testid="stMainBlockContainer"] > * { max-width: 560px; margin-left: auto; margin-right: auto; }
    
    body { font-family: 'DM Sans', sans-serif; color: var(--cream); }
    
    .header-section { text-align: center; margin-bottom: 3rem; }
    .eyebrow { font-size: 0.68rem; font-weight: 500; letter-spacing: 0.28em; text-transform: uppercase; color: var(--gold); display: flex; align-items: center; justify-content: center; gap: 0.8rem; margin-bottom: 0.8rem; }
    .main-title { font-family: 'Playfair Display', serif; font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 400; line-height: 1.1; color: var(--cream); margin-bottom: 0.5rem; }
    .main-title em { font-style: italic; color: var(--gold-light); }
    .subtitle { font-size: 0.82rem; color: rgba(247,241,232,0.4); font-weight: 300; letter-spacing: 0.03em; }
    
    .steppers { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; width: 100%; background: rgba(200,169,110,0.12); border: 1px solid rgba(200,169,110,0.12); border-radius: 6px 6px 0 0; overflow: hidden; margin-bottom: 1px; }
    
    .stepper-cell { background: rgba(10,20,13,0.75); backdrop-filter: blur(20px); padding: 1.6rem 1.4rem; display: flex; flex-direction: column; align-items: center; gap: 0.8rem; position: relative; }
    .stepper-cell::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; opacity: 0; transition: opacity 0.3s; }
    .stepper-cell.sepal::before { background: linear-gradient(90deg, transparent, var(--setosa), transparent); }
    .stepper-cell.petal::before { background: linear-gradient(90deg, transparent, var(--virginica), transparent); }
    .stepper-cell:hover::before { opacity: 1; }
    
    .stepper-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(247,241,232,0.35); display: flex; align-items: center; gap: 0.4rem; }
    .part-pip { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; background: var(--setosa); }
    .petal .part-pip { background: var(--virginica); }
    
    .step-value { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 400; color: var(--gold-light); text-align: center; }
    .step-unit { font-size: 0.6rem; color: rgba(247,241,232,0.2); letter-spacing: 0.1em; text-align: center; }
    
    .bottom-bar { width: 100%; background: rgba(10,20,13,0.75); backdrop-filter: blur(20px); border: 1px solid rgba(200,169,110,0.12); border-top: none; border-radius: 0 0 6px 6px; padding: 0.85rem 1.4rem; display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
    .samples-label { font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(247,241,232,0.2); flex-shrink: 0; }
    
    .pill { background: none; border: 1px solid rgba(247,241,232,0.12); border-radius: 20px; padding: 0.28rem 0.85rem; font-size: 0.7rem; font-style: italic; font-family: 'Playfair Display', serif; color: rgba(247,241,232,0.38); cursor: pointer; letter-spacing: 0.04em; transition: all 0.2s; flex-shrink: 0; }
    .pill:hover { border-color: var(--gold); color: var(--gold-light); }
    
    .predict-btn { width: 100%; padding: 1rem; background: transparent; border: 1px solid var(--gold); border-radius: 4px; color: var(--gold-light); font-family: 'DM Sans'; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.22em; text-transform: uppercase; cursor: pointer; position: relative; overflow: hidden; transition: color 0.3s; margin-bottom: 1.4rem; }
    .predict-btn::before { content: ''; position: absolute; inset: 0; background: var(--gold); transform: scaleX(0); transform-origin: left; transition: transform 0.35s cubic-bezier(0.22,0.61,0.36,1); z-index: -1; }
    .predict-btn:hover:not(:disabled) { color: var(--dark); }
    .predict-btn:hover:not(:disabled)::before { transform: scaleX(1); }
    .predict-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .result-panel { width: 100%; background: rgba(10,20,13,0.82); backdrop-filter: blur(24px); border: 1px solid rgba(200,169,110,0.15); border-radius: 6px; padding: 1.8rem 2rem; position: relative; overflow: hidden; animation: riseIn 0.45s cubic-bezier(0.22,0.61,0.36,1) both; }
    @keyframes riseIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    
    .result-glow { position: absolute; inset: 0; opacity: 0.07; transition: background 0.5s; pointer-events: none; }
    .result-glow.setosa { background: radial-gradient(ellipse at 85% 50%, var(--setosa), transparent 65%); }
    .result-glow.versicolor { background: radial-gradient(ellipse at 85% 50%, var(--versicolor), transparent 65%); }
    .result-glow.virginica { background: radial-gradient(ellipse at 85% 50%, var(--virginica), transparent 65%); }
    
    .result-top { display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 1.6rem; position: relative; }
    .result-label { font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(247,241,232,0.28); margin-bottom: 0.35rem; }
    .result-name { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 400; font-style: italic; line-height: 1; }
    .result-name.setosa { color: var(--setosa); }
    .result-name.versicolor { color: var(--versicolor); }
    .result-name.virginica { color: var(--virginica); }
    
    .result-conf-block { text-align: right; }
    .conf-label-sm { font-size: 0.62rem; color: rgba(247,241,232,0.28); letter-spacing: 0.12em; text-transform: uppercase; }
    .result-conf-big { font-family: 'Playfair Display', serif; font-size: 2.6rem; font-weight: 400; color: var(--gold-light); line-height: 1; }
    
    .bars-label { font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(247,241,232,0.18); margin-bottom: 0.9rem; }
    .bar-row { display: grid; grid-template-columns: 100px 1fr 44px; align-items: center; gap: 0.9rem; margin-bottom: 0.7rem; }
    .bar-name { font-size: 0.73rem; font-style: italic; font-family: 'Playfair Display', serif; color: rgba(247,241,232,0.45); }
    .bar-track { height: 3px; background: rgba(247,241,232,0.07); border-radius: 2px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 2px; transition: width 0.7s cubic-bezier(0.22,0.61,0.36,1); }
    .bar-fill.setosa { background: var(--setosa); }
    .bar-fill.versicolor { background: var(--versicolor); }
    .bar-fill.virginica { background: var(--virginica); }
    .bar-pct { font-size: 0.72rem; color: rgba(247,241,232,0.38); text-align: right; }
    
    footer { margin-top: 3rem; text-align: center; font-size: 0.65rem; color: rgba(247,241,232,0.18); letter-spacing: 0.08em; }
    
    @media (max-width: 480px) { .steppers { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_dataset():
    if Path("iris.csv").exists():
        return pd.read_csv("iris.csv")
    return None

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <div class="eyebrow">Botanical Classifier</div>
        <h1 class="main-title">Identify the <em>Iris</em></h1>
        <p class="subtitle">Random Forest · 100 Trees · 70/30 Split · Streamlit Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ iris_model.pkl not found. Make sure it's in the same folder as app.py.")
        st.stop()
    
    # Initialize session state for persistent values
    for val in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        if val not in st.session_state:
            st.session_state[val] = [5.8, 3.0, 3.8, 1.2][["sepal_length", "sepal_width", "petal_length", "petal_width"].index(val)]
    
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
        st.session_state.probabilities = None
    
    # Create stepper grid with HTML
    st.markdown('<div class="steppers">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stepper-cell sepal">
            <span class="stepper-label"><span class="part-pip"></span>Sepal Length</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.slider("Sepal length", 4.0, 8.0, st.session_state.sepal_length, 0.1, label_visibility="collapsed", key="sl")
        st.session_state.sepal_length = val
        st.markdown(f'<div style="text-align: center; font-family: Playfair Display; font-size: 2.1rem; color: #e8d5a8;">{val:.1f}</div><div style="text-align: center; font-size: 0.6rem; color: rgba(247,241,232,0.2);">centimetres</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stepper-cell sepal">
            <span class="stepper-label"><span class="part-pip"></span>Sepal Width</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.slider("Sepal width", 2.0, 4.5, st.session_state.sepal_width, 0.1, label_visibility="collapsed", key="sw")
        st.session_state.sepal_width = val
        st.markdown(f'<div style="text-align: center; font-family: Playfair Display; font-size: 2.1rem; color: #e8d5a8;">{val:.1f}</div><div style="text-align: center; font-size: 0.6rem; color: rgba(247,241,232,0.2);">centimetres</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="stepper-cell petal">
            <span class="stepper-label"><span class="part-pip"></span>Petal Length</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.slider("Petal length", 1.0, 7.0, st.session_state.petal_length, 0.1, label_visibility="collapsed", key="pl")
        st.session_state.petal_length = val
        st.markdown(f'<div style="text-align: center; font-family: Playfair Display; font-size: 2.1rem; color: #e8d5a8;">{val:.1f}</div><div style="text-align: center; font-size: 0.6rem; color: rgba(247,241,232,0.2);">centimetres</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stepper-cell petal">
            <span class="stepper-label"><span class="part-pip"></span>Petal Width</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.slider("Petal width", 0.1, 2.5, st.session_state.petal_width, 0.1, label_visibility="collapsed", key="pw")
        st.session_state.petal_width = val
        st.markdown(f'<div style="text-align: center; font-family: Playfair Display; font-size: 2.1rem; color: #e8d5a8;">{val:.1f}</div><div style="text-align: center; font-size: 0.6rem; color: rgba(247,241,232,0.2);">centimetres</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom bar with sample pills
    st.markdown("""
    <div class="bottom-bar">
        <span class="samples-label">Try</span>
        <span class="pill">I. setosa</span>
        <span class="pill">I. versicolor</span>
        <span class="pill">I. virginica</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Predict button
    if st.button("IDENTIFY SPECIES", use_container_width=True, key="predict", help="Predict the iris species"):
        features = np.array([[st.session_state.sepal_length, st.session_state.sepal_width, st.session_state.petal_length, st.session_state.petal_width]])
        pred = model.predict(features)[0]
        probas = model.predict_proba(features)[0]
        st.session_state.prediction = pred
        st.session_state.probabilities = {cls: float(p) for cls, p in zip(model.classes_, probas)}
    
    # Display results
    if st.session_state.prediction:
        pred = st.session_state.prediction
        probs = st.session_state.probabilities
        conf = max(probs.values()) * 100
        
        st.markdown(f"""
        <div class="result-panel">
            <div class="result-glow {pred}"></div>
            <div class="result-top">
                <div>
                    <div class="result-label">Identified Species</div>
                    <div class="result-name {pred}">{pred}</div>
                </div>
                <div class="result-conf-block">
                    <div class="conf-label-sm">Confidence</div>
                    <div class="result-conf-big">{conf:.1f}%</div>
                </div>
            </div>
            <div class="bars-label">All species probabilities</div>
        """, unsafe_allow_html=True)
        
        for species in ["setosa", "versicolor", "virginica"]:
            pct = probs.get(species, 0) * 100
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-name">I. {species}</div>
                <div class="bar-track"><div class="bar-fill {species}" style="width: {pct:.1f}%"></div></div>
                <div class="bar-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<footer>Iris Dataset · scikit-learn RandomForestClassifier · Streamlit</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
