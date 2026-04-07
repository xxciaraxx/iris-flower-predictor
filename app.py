import base64
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="iris-icon.jpg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("iris_model.pkl")
BG_PATH = Path("iris-bg2.jpg")

DEFAULT_VALUES = {
    "sl": 5.8,
    "sw": 3.0,
    "pl": 3.8,
    "pw": 1.2,
}

SAMPLES = {
    "setosa": {"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2},
    "versicolor": {"sl": 6.0, "sw": 2.9, "pl": 4.5, "pw": 1.5},
    "virginica": {"sl": 6.3, "sw": 3.3, "pl": 6.0, "pw": 2.5},
}


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def set_sample(sample_name: str) -> None:
    sample = SAMPLES[sample_name]
    st.session_state["sl"] = sample["sl"]
    st.session_state["sw"] = sample["sw"]
    st.session_state["pl"] = sample["pl"]
    st.session_state["pw"] = sample["pw"]


def run_prediction(sl: float, sw: float, pl: float, pw: float):
    model = load_model()
    if model is None:
        return None

    features = np.array([[sl, sw, pl, pw]], dtype=float)
    pred = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    probs = {cls: float(p) for cls, p in zip(model.classes_, probas)}
    confidence = max(probs.values()) * 100

    return {
        "species": str(pred),
        "confidence": round(confidence, 1),
        "probabilities": probs,
    }


for key, value in DEFAULT_VALUES.items():
    st.session_state.setdefault(key, value)

bg_uri = img_to_base64(BG_PATH)

st.markdown(
    f"""
    <style>
        #MainMenu, header, footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {{
            display: none !important;
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background: #0f1c14 !important;
        }}

        .stApp {{
            background:
                linear-gradient(160deg, rgba(15, 28, 20, 0.45), rgba(15, 28, 20, 0.92)),
                url("{bg_uri}") center/cover fixed no-repeat;
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 960px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }}

        .hero {{
            text-align: center;
            color: #f7f1e8;
            margin-bottom: 2.5rem;
        }}

        .eyebrow {{
            font-size: 0.72rem;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: #c8a96e;
            margin-bottom: 0.8rem;
        }}

        .hero h1 {{
            font-size: clamp(2.2rem, 5vw, 3.8rem);
            line-height: 1.05;
            margin: 0;
            font-weight: 700;
            color: #f7f1e8;
        }}

        .hero p {{
            margin-top: 0.8rem;
            color: rgba(247, 241, 232, 0.74);
        }}

        .glass {{
            background: rgba(10, 20, 13, 0.78);
            border: 1px solid rgba(200, 169, 110, 0.18);
            border-radius: 20px;
            padding: 1.4rem;
            backdrop-filter: blur(18px);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        }}

        .sample-title {{
            color: rgba(247, 241, 232, 0.7);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            margin-bottom: 0.8rem;
        }}

        div[data-testid="column"] {{
            width: 100%;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid #c8a96e;
            background: transparent;
            color: #f1dfb3;
            padding: 0.85rem 1rem;
            font-weight: 600;
            letter-spacing: 0.08em;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            border-color: #f1dfb3;
            color: #0f1c14;
            background: #c8a96e;
        }}

        div[data-testid="stNumberInput"] {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(200, 169, 110, 0.12);
            border-radius: 16px;
            padding: 0.35rem 0.6rem;
        }}

        div[data-testid="stNumberInput"] label {{
            color: rgba(247, 241, 232, 0.82) !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        div[data-testid="stNumberInput"] input {{
            color: #f7f1e8 !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            background: transparent !important;
        }}

        .result-card {{
            background: rgba(10, 20, 13, 0.82);
            border: 1px solid rgba(200, 169, 110, 0.18);
            border-radius: 22px;
            padding: 1.5rem;
            margin-top: 1rem;
            color: #f7f1e8;
            backdrop-filter: blur(20px);
        }}

        .result-head {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: end;
            margin-bottom: 1rem;
        }}

        .result-kicker {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: rgba(247, 241, 232, 0.58);
        }}

        .result-name {{
            font-size: 2.1rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }}

        .result-confidence {{
            text-align: right;
        }}

        .prob-row {{
            display: grid;
            grid-template-columns: 110px 1fr 54px;
            gap: 0.8rem;
            align-items: center;
            margin-top: 0.8rem;
        }}

        .prob-label {{
            color: rgba(247, 241, 232, 0.8);
            font-size: 0.95rem;
        }}

        .prob-track {{
            background: rgba(247, 241, 232, 0.08);
            border-radius: 999px;
            height: 8px;
            overflow: hidden;
        }}

        .prob-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #c8a96e, #e8d5a8);
        }}

        .prob-value {{
            text-align: right;
            color: rgba(247, 241, 232, 0.82);
        }}

        .model-note {{
            margin-top: 0.9rem;
            color: rgba(247, 241, 232, 0.65);
            font-size: 0.88rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Iris Species Predictor</div>
        <h1>Predict from the trained <em>iris_model.pkl</em></h1>
        <p>Enter flower measurements, then Streamlit runs the saved model in Python and returns the prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
if model is None:
    st.error("`iris_model.pkl` was not found, so prediction cannot run.")
    st.stop()

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sample-title">Quick Samples</div>', unsafe_allow_html=True)
sample_cols = st.columns(3)
with sample_cols[0]:
    if st.button("Setosa", use_container_width=True):
        set_sample("setosa")
        st.rerun()
with sample_cols[1]:
    if st.button("Versicolor", use_container_width=True):
        set_sample("versicolor")
        st.rerun()
with sample_cols[2]:
    if st.button("Virginica", use_container_width=True):
        set_sample("virginica")
        st.rerun()

with st.form("iris_predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Sepal Length",
            min_value=4.3,
            max_value=7.9,
            step=0.1,
            key="sl",
            format="%.1f",
        )
        st.number_input(
            "Petal Length",
            min_value=1.0,
            max_value=6.9,
            step=0.1,
            key="pl",
            format="%.1f",
        )
    with col2:
        st.number_input(
            "Sepal Width",
            min_value=2.0,
            max_value=4.4,
            step=0.1,
            key="sw",
            format="%.1f",
        )
        st.number_input(
            "Petal Width",
            min_value=0.1,
            max_value=2.5,
            step=0.1,
            key="pw",
            format="%.1f",
        )

    submitted = st.form_submit_button("Identify Species", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    st.session_state["pred_result"] = run_prediction(
        st.session_state["sl"],
        st.session_state["sw"],
        st.session_state["pl"],
        st.session_state["pw"],
    )

pred_result = st.session_state.get("pred_result")

if pred_result:
    species = pred_result["species"]
    confidence = pred_result["confidence"]
    probabilities = pred_result["probabilities"]

    rows = []
    for cls in ["setosa", "versicolor", "virginica"]:
        pct = (probabilities.get(cls, 0.0) * 100)
        rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-label">{cls.title()}</div>
                <div class="prob-track"><div class="prob-fill" style="width: {pct:.1f}%;"></div></div>
                <div class="prob-value">{pct:.1f}%</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-head">
                <div>
                    <div class="result-kicker">Predicted Species</div>
                    <div class="result-name">{species.title()}</div>
                </div>
                <div class="result-confidence">
                    <div class="result-kicker">Confidence</div>
                    <div class="result-name">{confidence:.1f}%</div>
                </div>
            </div>
            {''.join(rows)}
            <div class="model-note">Prediction source: <code>iris_model.pkl</code> loaded with <code>joblib</code> and evaluated in Python.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
