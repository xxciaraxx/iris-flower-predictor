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


def step_value(key: str, direction: int, min_value: float, max_value: float) -> None:
    current = float(st.session_state.get(key, DEFAULT_VALUES[key]))
    updated = round(current + (0.1 * direction), 1)
    st.session_state[key] = min(max(updated, min_value), max_value)


def render_measure_input(
    label: str,
    key: str,
    min_value: float,
    max_value: float,
    accent: str,
) -> None:
    st.markdown(
        f"""
        <div class="measure-card {accent}">
            <div class="measure-label"><span class="measure-dot"></span>{label}</div>
        """,
        unsafe_allow_html=True,
    )
    control_cols = st.columns([0.9, 2.2, 0.9])
    with control_cols[0]:
        if st.form_submit_button("−", use_container_width=True):
            step_value(key, -1, min_value, max_value)
            st.rerun()
    with control_cols[1]:
        st.markdown(
            f"""
            <div class="measure-value-wrap">
                <div class="measure-value">{float(st.session_state[key]):.1f}</div>
                <div class="measure-unit">centimetres</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with control_cols[2]:
        if st.form_submit_button("+", use_container_width=True):
            step_value(key, 1, min_value, max_value)
            st.rerun()
    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            max-width: 1180px;
            padding-top: 2.2rem;
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
            display: inline-flex;
            align-items: center;
            gap: 1rem;
        }}

        .eyebrow::before,
        .eyebrow::after {{
            content: "";
            width: 44px;
            height: 1px;
            background: rgba(200, 169, 110, 0.45);
        }}

        .hero h1 {{
            font-size: clamp(3.4rem, 7vw, 5.4rem);
            line-height: 0.96;
            margin: 0 0 0.6rem 0;
            font-weight: 500;
            color: #f7f1e8;
            letter-spacing: -0.03em;
        }}

        .hero p {{
            margin-top: 0.2rem;
            color: rgba(247, 241, 232, 0.45);
            font-size: 0.88rem;
        }}

        .panel-shell {{
            max-width: 700px;
            margin: 0 auto;
        }}

        .glass {{
            background: rgba(10, 20, 13, 0.76);
            border: 1px solid rgba(200, 169, 110, 0.16);
            border-radius: 10px;
            padding: 0;
            backdrop-filter: blur(16px);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
            overflow: hidden;
        }}

        .sample-title {{
            color: rgba(247, 241, 232, 0.28);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.68rem;
            margin-bottom: 0;
        }}

        .sample-row {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 1rem 1.2rem 1.1rem;
            border-top: 1px solid rgba(200, 169, 110, 0.14);
            background: rgba(4, 15, 10, 0.24);
        }}

        div[data-testid="column"] {{
            width: 100%;
        }}

        div[data-testid="stHorizontalBlock"] {{
            gap: 0;
        }}

        .metric-grid [data-testid="column"] {{
            border-right: 1px solid rgba(200, 169, 110, 0.14);
            border-bottom: 1px solid rgba(200, 169, 110, 0.14);
            padding: 1.65rem 1.7rem 1.8rem;
            background: rgba(12, 22, 15, 0.72);
        }}

        .metric-grid [data-testid="column"]:nth-child(2),
        .metric-grid [data-testid="column"]:nth-child(4) {{
            border-right: none;
        }}

        .metric-grid-bottom [data-testid="column"] {{
            border-bottom: none;
        }}

        .measure-card {{
            min-height: 168px;
        }}

        .measure-label {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: rgba(247, 241, 232, 0.42);
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.66rem;
            margin-bottom: 1.4rem;
            font-weight: 600;
        }}

        .measure-dot {{
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #7ecba1;
            display: inline-block;
        }}

        .measure-card.petal .measure-dot {{
            background: #c57ebf;
        }}

        .measure-value-wrap {{
            text-align: center;
            padding-top: 0.15rem;
        }}

        .measure-value {{
            color: #f5e7c5;
            font-size: 2.2rem;
            line-height: 1;
            font-weight: 500;
            letter-spacing: -0.03em;
        }}

        .measure-unit {{
            margin-top: 0.45rem;
            color: rgba(247, 241, 232, 0.2);
            letter-spacing: 0.14em;
            text-transform: lowercase;
            font-size: 0.66rem;
            font-weight: 600;
        }}

        .sample-row .stFormSubmitButton > button {{
            border-radius: 999px;
            border: 1px solid rgba(247, 241, 232, 0.14);
            background: transparent;
            color: rgba(247, 241, 232, 0.58);
            padding: 0.45rem 0.8rem;
            font-weight: 500;
            letter-spacing: 0;
            font-style: italic;
            box-shadow: none;
            min-height: 0;
            margin-top: 0;
        }}

        .sample-row .stFormSubmitButton > button:hover {{
            border-color: rgba(200, 169, 110, 0.5);
            color: #f1dfb3;
            background: rgba(200, 169, 110, 0.08);
        }}

        .stFormSubmitButton > button {{
            width: 100%;
            border-radius: 4px;
            border: 1px solid #c8a96e;
            background: transparent;
            color: #f1dfb3;
            padding: 1rem 1rem;
            font-weight: 600;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            min-height: 0;
            margin-top: 1.2rem;
        }}

        .stFormSubmitButton > button:hover {{
            border-color: #f1dfb3;
            color: #f7f1e8;
            background: rgba(200, 169, 110, 0.08);
        }}

        .measure-card .stFormSubmitButton > button {{
            width: 48px;
            height: 48px;
            min-height: 48px;
            padding: 0;
            border-radius: 6px;
            margin-top: 0.3rem;
            color: #d7b977;
            background: rgba(200, 169, 110, 0.08);
            border: 1px solid rgba(200, 169, 110, 0.16);
            font-size: 1.7rem;
            line-height: 1;
            letter-spacing: 0;
            text-transform: none;
            font-weight: 400;
        }}

        .measure-card .stFormSubmitButton > button:hover {{
            background: rgba(200, 169, 110, 0.16);
            color: #f1dfb3;
            border-color: rgba(200, 169, 110, 0.28);
        }}

        .measure-card [data-testid="stHorizontalBlock"] {{
            align-items: center;
            gap: 1rem;
        }}

        .result-card {{
            background: rgba(8, 18, 12, 0.8);
            border: 1px solid rgba(200, 169, 110, 0.14);
            border-radius: 10px;
            padding: 1.8rem 2rem;
            margin: 2rem auto 0;
            color: #f7f1e8;
            backdrop-filter: blur(20px);
            max-width: 980px;
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
            font-size: 2.2rem;
            font-weight: 500;
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

        .reference-card {{
            max-width: 980px;
            margin: 2rem auto 0;
            background: rgba(8, 18, 12, 0.74);
            border: 1px solid rgba(200, 169, 110, 0.14);
            border-radius: 10px;
            padding: 1.8rem 2rem;
            color: #f7f1e8;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            backdrop-filter: blur(20px);
        }}

        .reference-title {{
            font-size: 1.05rem;
            margin-bottom: 0.45rem;
            color: #f2e5c7;
        }}

        .reference-copy {{
            color: rgba(247, 241, 232, 0.68);
            max-width: 430px;
            line-height: 1.45;
        }}

        .reference-stats {{
            color: rgba(247, 241, 232, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-size: 0.78rem;
            white-space: nowrap;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Iris Species Predictor</div>
        <h1>Identify the <em style="color:#e8d5a8;">Iris</em></h1>
        <p>Random Forest · 100 Trees · 70/30 Split · Live model prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
if model is None:
    st.error("`iris_model.pkl` was not found, so prediction cannot run.")
    st.stop()

st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
with st.form("iris_predict_form"):
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    top_row = st.columns(2)
    with top_row[0]:
        render_measure_input("Sepal Length", "sl", 4.3, 7.9, "sepal")
    with top_row[1]:
        render_measure_input("Sepal Width", "sw", 2.0, 4.4, "sepal")
    st.markdown('</div><div class="metric-grid metric-grid-bottom">', unsafe_allow_html=True)
    bottom_row = st.columns(2)
    with bottom_row[0]:
        render_measure_input("Petal Length", "pl", 1.0, 6.9, "petal")
    with bottom_row[1]:
        render_measure_input("Petal Width", "pw", 0.1, 2.5, "petal")
    st.markdown('</div><div class="sample-row">', unsafe_allow_html=True)
    sample_cols = st.columns([0.8, 1, 1, 1])
    with sample_cols[0]:
        st.markdown('<div class="sample-title">Try</div>', unsafe_allow_html=True)
    with sample_cols[1]:
        if st.form_submit_button("I. setosa", use_container_width=True):
            set_sample("setosa")
            st.rerun()
    with sample_cols[2]:
        if st.form_submit_button("I. versicolor", use_container_width=True):
            set_sample("versicolor")
            st.rerun()
    with sample_cols[3]:
        if st.form_submit_button("I. virginica", use_container_width=True):
            set_sample("virginica")
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)
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

st.markdown(
    """
    <div class="reference-card">
        <div>
            <div class="reference-title">Iris Dataset Reference</div>
            <div class="reference-copy">Use this panel as a visual anchor for the original design. Predictions above still come directly from the trained model.</div>
        </div>
        <div class="reference-stats">150 Samples · 3 Species · 4 Features</div>
    </div>
    """,
    unsafe_allow_html=True,
)
