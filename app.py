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
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    :root {
        --cream: #f7f1e8;
        --dark: #0f1c14;
        --gold: #c8a96e;
        --gold-light: #e8d5a8;
        --setosa: #7ecba1;
        --versicolor: #6ba3d6;
        --virginica: #c57ebf;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, rgba(15,28,20,0.7) 0%, rgba(15,28,20,1) 100%);
    }
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url('https://images.unsplash.com/photo-1490750967868-88df5691cc06?w=1920&q=80');
        background-size: cover;
        background-position: center 30%;
        filter: saturate(0.7) brightness(0.32);
        z-index: 0;
    }
    
    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
        padding: 2rem 1rem !important;
    }
    
    body { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; color: var(--cream); }
    
    .header-section { text-align: center; margin-bottom: 2rem; }
    .eyebrow { font-size: 0.68rem; font-weight: 500; letter-spacing: 0.28em; text-transform: uppercase; color: var(--gold); margin-bottom: 0.8rem; }
    .main-title { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 400; color: var(--cream); margin-bottom: 0.5rem; }
    .main-title em { font-style: italic; color: var(--gold-light); }
    .subtitle { font-size: 0.82rem; color: rgba(247,241,232,0.4); font-weight: 300; }
    
    .steppers { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; width: 100%; max-width: 560px; margin: 0 auto 1rem; background: rgba(200,169,110,0.12); border: 1px solid rgba(200,169,110,0.12); border-radius: 6px 6px 0 0; overflow: hidden; }
    
    .stepper-cell { background: rgba(10,20,13,0.75); backdrop-filter: blur(20px); padding: 1.6rem 1.4rem; display: flex; flex-direction: column; align-items: center; gap: 0.8rem; position: relative; }
    .stepper-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(247,241,232,0.35); }
    
    .input-wrapper { width: 100%; }
    .input-row { display: flex; align-items: center; gap: 0.8rem; justify-content: center; }
    .step-btn { width: 38px; height: 38px; background: rgba(200,169,110,0.08); border: 1px solid rgba(200,169,110,0.2); border-radius: 4px; color: var(--gold); font-size: 1.2rem; cursor: pointer; transition: all 0.15s; }
    .step-btn:hover { background: rgba(200,169,110,0.18); border-color: rgba(200,169,110,0.5); }
    
    .step-value { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 400; color: var(--gold-light); text-align: center; min-width: 80px; }
    .step-unit { font-size: 0.6rem; color: rgba(247,241,232,0.2); letter-spacing: 0.1em; text-align: center; }
    
    .bottom-bar { width: 100%; max-width: 560px; margin: 0 auto 1rem; background: rgba(10,20,13,0.75); border: 1px solid rgba(200,169,110,0.12); border-top: none; border-radius: 0 0 6px 6px; padding: 0.85rem 1.4rem; display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
    .samples-label { font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(247,241,232,0.2); }
    
    .pill-btn { background: none; border: 1px solid rgba(247,241,232,0.12); border-radius: 20px; padding: 0.4rem 0.85rem; font-size: 0.7rem; font-style: italic; color: rgba(247,241,232,0.38); cursor: pointer; transition: all 0.2s; }
    .pill-btn:hover { border-color: var(--gold); color: var(--gold-light); }
    
    .predict-btn-wrapper { width: 100%; max-width: 560px; margin: 0 auto 1.2rem; }
    .predict-btn { width: 100%; padding: 1rem; background: transparent; border: 1px solid var(--gold); border-radius: 4px; color: var(--gold-light); font-family: 'DM Sans'; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.22em; text-transform: uppercase; cursor: pointer; transition: all 0.3s; }
    .predict-btn:hover { background: var(--gold); color: var(--dark); }
    .predict-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .result-panel { width: 100%; max-width: 560px; margin: 1.4rem auto 0; background: rgba(10,20,13,0.82); border: 1px solid rgba(200,169,110,0.15); border-radius: 6px; padding: 1.8rem 2rem; position: relative; animation: riseIn 0.45s cubic-bezier(0.22,0.61,0.36,1) both; }
    @keyframes riseIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    
    .result-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 1.6rem; }
    .result-name { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 400; font-style: italic; line-height: 1; }
    .result-name.setosa { color: var(--setosa); }
    .result-name.versicolor { color: var(--versicolor); }
    .result-name.virginica { color: var(--virginica); }
    
    .conf-block { text-align: right; }
    .conf-label { font-size: 0.62rem; color: rgba(247,241,232,0.28); letter-spacing: 0.12em; text-transform: uppercase; }
    .result-conf { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 400; color: var(--gold-light); }
    
    .bars-section { margin-top: 1rem; }
    .bars-label { font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(247,241,232,0.18); margin-bottom: 0.9rem; }
    .bar-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.7rem; }
    .bar-name { font-size: 0.73rem; font-style: italic; font-family: 'Playfair Display', serif; color: rgba(247,241,232,0.45); min-width: 100px; }
    .bar-track { flex: 1; height: 3px; background: rgba(247,241,232,0.07); border-radius: 2px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 2px; transition: width 0.7s cubic-bezier(0.22,0.61,0.36,1); }
    .bar-fill.setosa { background: var(--setosa); }
    .bar-fill.versicolor { background: var(--versicolor); }
    .bar-fill.virginica { background: var(--virginica); }
    .bar-pct { font-size: 0.72rem; color: rgba(247,241,232,0.38); min-width: 40px; text-align: right; }
    
    .info-section { width: 100%; max-width: 560px; margin: 1.4rem auto 0; font-size: 0.75rem; color: rgba(247,241,232,0.4); }
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
    <div class="header-section">
        <div class="eyebrow">🌿 Botanical Classifier</div>
        <h1 class="main-title">Identify the <em>Iris</em></h1>
        <p class="subtitle">Random Forest · Streamlit Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("iris_model.pkl not found. Please ensure it's in the same folder.")
        st.stop()
    
    # Initialize session state
    if "sepal_length" not in st.session_state:
        st.session_state.sepal_length = 5.1
    if "sepal_width" not in st.session_state:
        st.session_state.sepal_width = 3.5
    if "petal_length" not in st.session_state:
        st.session_state.petal_length = 1.4
    if "petal_width" not in st.session_state:
        st.session_state.petal_width = 0.2
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "probabilities" not in st.session_state:
        st.session_state.probabilities = None
    
    # Stepper controls
    st.markdown('<div class="steppers">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stepper-cell sepal">
            <div class="stepper-label">🌸 Sepal Length</div>
            <div class="input-wrapper">
                <div class="input-row">
                    <button class="step-btn">−</button>
                    <div style="text-align: center; flex: 1;">
                        <div class="step-value">"""
                            + f"{st.session_state.sepal_length:.1f}"
                        + """</div>
                        <div class="step-unit">centimetres</div>
                    </div>
                    <button class="step-btn">+</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("Sepal length", min_value=4.0, max_value=8.0, value=st.session_state.sepal_length, step=0.1, label_visibility="collapsed")
        st.session_state.sepal_length = val
    
    with col2:
        st.markdown("""
        <div class="stepper-cell sepal">
            <div class="stepper-label">🌸 Sepal Width</div>
            <div class="input-wrapper">
                <div class="input-row">
                    <button class="step-btn">−</button>
                    <div style="text-align: center; flex: 1;">
                        <div class="step-value">"""
                            + f"{st.session_state.sepal_width:.1f}"
                        + """</div>
                        <div class="step-unit">centimetres</div>
                    </div>
                    <button class="step-btn">+</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("Sepal width", min_value=2.0, max_value=4.5, value=st.session_state.sepal_width, step=0.1, label_visibility="collapsed")
        st.session_state.sepal_width = val
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="stepper-cell petal">
            <div class="stepper-label">🌺 Petal Length</div>
            <div class="input-wrapper">
                <div class="input-row">
                    <button class="step-btn">−</button>
                    <div style="text-align: center; flex: 1;">
                        <div class="step-value">"""
                            + f"{st.session_state.petal_length:.1f}"
                        + """</div>
                        <div class="step-unit">centimetres</div>
                    </div>
                    <button class="step-btn">+</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("Petal length", min_value=1.0, max_value=7.0, value=st.session_state.petal_length, step=0.1, label_visibility="collapsed")
        st.session_state.petal_length = val
    
    with col4:
        st.markdown("""
        <div class="stepper-cell petal">
            <div class="stepper-label">🌺 Petal Width</div>
            <div class="input-wrapper">
                <div class="input-row">
                    <button class="step-btn">−</button>
                    <div style="text-align: center; flex: 1;">
                        <div class="step-value">"""
                            + f"{st.session_state.petal_width:.1f}"
                        + """</div>
                        <div class="step-unit">centimetres</div>
                    </div>
                    <button class="step-btn">+</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("Petal width", min_value=0.1, max_value=2.5, value=st.session_state.petal_width, step=0.1, label_visibility="collapsed")
        st.session_state.petal_width = val
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom bar with samples
    st.markdown(f"""
    <div class="bottom-bar">
        <span class="samples-label">Try samples:</span>
        <button class="pill-btn" onclick="window.location.reload()">I. setosa</button>
        <button class="pill-btn" onclick="window.location.reload()">I. versicolor</button>
        <button class="pill-btn" onclick="window.location.reload()">I. virginica</button>
    </div>
    """, unsafe_allow_html=True)
    
    # Predict button
    st.markdown('<div class="predict-btn-wrapper">', unsafe_allow_html=True)
    if st.button("Identify Species", use_container_width=True, key="predict_btn"):
        features = np.array([[
            st.session_state.sepal_length,
            st.session_state.sepal_width,
            st.session_state.petal_length,
            st.session_state.petal_width
        ]])
        
        prediction = model.predict(features)[0]
        probas = model.predict_proba(features)[0]
        st.session_state.prediction = prediction
        st.session_state.probabilities = {
            cls: float(p) for cls, p in zip(model.classes_, probas)
        }
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results
    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        probs = st.session_state.probabilities
        conf = max(probs.values()) * 100
        
        st.markdown(f"""
        <div class="result-panel">
            <div class="result-header">
                <div>
                    <div style="font-size: 0.62rem; color: rgba(247,241,232,0.28); letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.35rem;">Identified Species</div>
                    <div class="result-name {pred}">{pred.capitalize()}</div>
                </div>
                <div class="conf-block">
                    <div class="conf-label">Confidence</div>
                    <div class="result-conf">{conf:.1f}%</div>
                </div>
            </div>
            <div class="bars-section">
                <div class="bars-label">All species probabilities</div>
        """, unsafe_allow_html=True)
        
        for species in ["setosa", "versicolor", "virginica"]:
            prob = probs.get(species, 0) * 100
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-name">I. {species}</div>
                <div class="bar-track"><div class="bar-fill {species}" style="width: {prob:.1f}%"></div></div>
                <div class="bar-pct">{prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer/Info
    st.markdown("""
    <div class="info-section">
        <p>📊 This Streamlit app uses a pre-trained Random Forest model. Iris dataset · scikit-learn · joblib</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
