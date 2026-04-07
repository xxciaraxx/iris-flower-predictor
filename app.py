import base64
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="iris-icon.jpg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("iris_model.pkl")
BG_PATH    = Path("iris-bg2.jpg")
ICON_PATH  = Path("iris-icon.jpg")


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def img_to_base64(path: Path) -> str:
    """Return a data-URI for a local image so Streamlit can embed it."""
    if path.exists():
        ext = path.suffix.lstrip(".").lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{data}"
    return ""


st.markdown(
    """
    <style>
        /* Hide every default Streamlit element */
        #MainMenu, header, footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] { display: none !important; }

        /* Zero-out body / app container padding */
        html, body { margin: 0 !important; padding: 0 !important; width: 100vw !important; min-height: 100vh !important; background: #0f1c14 !important; overflow: hidden !important; }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .block-container,
        section.main > div { padding: 0 !important; margin: 0 !important; background: transparent !important; }
        .css-18e3th9 { background: transparent !important; }

        /* Remove default iframe border that Streamlit sometimes adds */
        iframe { border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def run_prediction(sl, sw, pl, pw):
    model = load_model()
    if model is None:
        return None
    features = np.array([[sl, sw, pl, pw]])
    pred     = model.predict(features)[0]
    probas   = model.predict_proba(features)[0]
    probs    = {cls: float(p) for cls, p in zip(model.classes_, probas)}
    conf     = max(probs.values()) * 100
    return {"species": pred, "confidence": round(conf, 1), "probabilities": probs}


params = st.query_params
pred_result = st.session_state.get("pred_result")

default_values = {
    "sl": 5.8,
    "sw": 3.0,
    "pl": 3.8,
    "pw": 1.2,
}


def get_param_value(name: str) -> float:
    try:
        return float(params.get(name, default_values[name]))
    except (TypeError, ValueError):
        return default_values[name]


current_sl = get_param_value("sl")
current_sw = get_param_value("sw")
current_pl = get_param_value("pl")
current_pw = get_param_value("pw")

if "current_sl" in st.session_state:
    current_sl = st.session_state["current_sl"]
    current_sw = st.session_state["current_sw"]
    current_pl = st.session_state["current_pl"]
    current_pw = st.session_state["current_pw"]

if "predict" in params:
    try:
        sl = current_sl
        sw = current_sw
        pl = current_pl
        pw = current_pw
        pred_result = run_prediction(sl, sw, pl, pw)
        st.session_state["pred_result"] = pred_result
    except Exception:
        pred_result = None

pred_json = json.dumps(pred_result) if pred_result else "null"

bg_uri   = img_to_base64(BG_PATH)
icon_uri = img_to_base64(ICON_PATH)

html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}

  :root {{
    --cream: #f7f1e8;
    --dark: #0f1c14;
    --gold: #c8a96e;
    --gold-light: #e8d5a8;
    --setosa: #7ecba1;
    --versicolor: #6ba3d6;
    --virginica: #c57ebf;
  }}

  body {{
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    background: var(--dark);
    color: var(--cream);
    overflow-x: hidden;
  }}

  .hero-bg {{
    position: fixed; inset: 0; z-index: 0;
    background-image: url('{bg_uri}');
    background-size: cover;
    background-position: center 30%;
    filter: saturate(0.7) brightness(1);
  }}
  .hero-bg::after {{
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(160deg, rgba(15,28,20,0.5) 0%, rgba(15,28,20,0.9) 100%);
  }}

  .page {{
    position: relative; z-index: 1;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 3rem 1.5rem 4rem;
  }}

  /* HEADER */
  header {{ text-align: center; margin-bottom: 3rem; }}

  .eyebrow {{
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.28em; text-transform: uppercase;
    color: var(--gold);
    display: flex; align-items: center; justify-content: center;
    gap: 0.8rem; margin-bottom: 0.8rem;
  }}
  .eyebrow::before, .eyebrow::after {{
    content: ''; width: 36px; height: 1px;
    background: var(--gold); opacity: 0.45;
  }}

  h1 {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 400; line-height: 1.1; margin-bottom: 0.5rem;
  }}
  h1 em {{ font-style: italic; color: var(--gold-light); }}

  .subtitle {{
    font-size: 0.82rem; color: rgba(247,241,232,0.4);
    font-weight: 300; letter-spacing: 0.03em;
  }}

  /* STEPPER GRID */
  .steppers {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    width: 100%; max-width: 560px;
    background: rgba(200,169,110,0.12);
    border: 1px solid rgba(200,169,110,0.12);
    border-radius: 6px 6px 0 0;
    overflow: hidden;
  }}

  .stepper-cell {{
    background: rgba(10,20,13,0.75);
    backdrop-filter: blur(20px);
    padding: 1.6rem 1.4rem;
    display: flex; flex-direction: column; align-items: center;
    gap: 0.8rem; position: relative;
  }}
  .stepper-cell::before {{
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    opacity: 0; transition: opacity 0.3s;
  }}
  .stepper-cell.sepal::before {{ background: linear-gradient(90deg, transparent, var(--setosa), transparent); }}
  .stepper-cell.petal::before {{ background: linear-gradient(90deg, transparent, var(--virginica), transparent); }}
  .stepper-cell:hover::before, .stepper-cell:focus-within::before {{ opacity: 1; }}

  .stepper-label {{
    font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: rgba(247,241,232,0.35);
    display: flex; align-items: center; gap: 0.4rem;
  }}
  .part-pip {{ width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }}
  .sepal .part-pip {{ background: var(--setosa); }}
  .petal .part-pip {{ background: var(--virginica); }}

  .stepper-control {{ display: flex; align-items: center; width: 100%; }}

  .step-btn {{
    flex-shrink: 0; width: 38px; height: 38px;
    background: rgba(200,169,110,0.08);
    border: 1px solid rgba(200,169,110,0.2);
    border-radius: 4px; color: var(--gold);
    font-size: 1.35rem; font-weight: 300;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background 0.15s, border-color 0.15s, transform 0.1s;
    user-select: none; -webkit-user-select: none; line-height: 1;
  }}
  .step-btn:hover {{ background: rgba(200,169,110,0.18); border-color: rgba(200,169,110,0.5); }}
  .step-btn:active, .step-btn.held {{ transform: scale(0.91); background: rgba(200,169,110,0.28); }}

  .step-display {{ flex: 1; text-align: center; }}
  .step-value {{
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem; font-weight: 400; color: var(--gold-light);
    line-height: 1; display: block; border: none; background: transparent;
    width: 100%; text-align: center; outline: none; caret-color: var(--gold);
  }}
  .step-value:hover,
  .step-value:focus,
  .step-value:active,
  .step-value:focus-visible {{
    background: transparent !important;
    outline: none !important;
    box-shadow: none !important;
  }}
  input[type=number] {{
    background: transparent !important;
    border: none !important;
    color: inherit !important;
    -webkit-appearance: none;
    -moz-appearance: textfield;
    appearance: none;
  }}
  input[type=number]::-webkit-outer-spin-button,
  input[type=number]::-webkit-inner-spin-button {{
    -webkit-appearance: none;
    margin: 0;
  }}
  input[type=number]:focus {{
    background: transparent !important;
    outline: none !important;
    box-shadow: none !important;
  }}
  input[type=number]:hover {{
    background: transparent !important;
  }}
  .step-unit {{ font-size: 0.6rem; color: rgba(247,241,232,0.2); letter-spacing: 0.1em; margin-top: 0.15rem; display: block; }}

  /* BOTTOM BAR */
  .bottom-bar {{
    width: 100%; max-width: 560px;
    background: rgba(10,20,13,0.75); backdrop-filter: blur(20px);
    border: 1px solid rgba(200,169,110,0.12); border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 0.85rem 1.4rem;
    display: flex; align-items: center; gap: 0.6rem;
  }}
  .samples-label {{ font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(247,241,232,0.2); flex-shrink: 0; }}
  .pill {{
    background: none; border: 1px solid rgba(247,241,232,0.12);
    border-radius: 20px; padding: 0.28rem 0.85rem;
    font-size: 0.7rem; font-style: italic;
    font-family: 'Playfair Display', serif;
    color: rgba(247,241,232,0.38); cursor: pointer;
    letter-spacing: 0.04em; transition: all 0.2s; flex-shrink: 0;
  }}
  .pill:hover {{ border-color: var(--gold); color: var(--gold-light); }}

  /* PREDICT BUTTON */
  .predict-btn {{
    width: 100%; max-width: 560px; margin-top: 1.2rem; padding: 1rem;
    background: transparent; border: 1px solid var(--gold); border-radius: 4px;
    color: var(--gold-light); font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem; font-weight: 500; letter-spacing: 0.22em;
    text-transform: uppercase; cursor: pointer;
    position: relative; overflow: hidden; transition: color 0.3s;
  }}
  .predict-btn::before {{
    content: ''; position: absolute; inset: 0;
    background: var(--gold); transform: scaleX(0);
    transform-origin: left; transition: transform 0.35s cubic-bezier(0.22,0.61,0.36,1); z-index: -1;
  }}
  .predict-btn:hover:not(:disabled) {{ color: var(--dark); }}
  .predict-btn:hover:not(:disabled)::before {{ transform: scaleX(1); }}
  .predict-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

  /* RESULT */
  .result-panel {{
    width: 100%; max-width: 560px; margin-top: 1.4rem;
    animation: riseIn 0.45s cubic-bezier(0.22,0.61,0.36,1) both;
  }}
  @keyframes riseIn {{
    from {{ opacity:0; transform: translateY(14px); }}
    to   {{ opacity:1; transform: translateY(0); }}
  }}
  .result-inner {{
    background: rgba(10,20,13,0.82); backdrop-filter: blur(24px);
    border: 1px solid rgba(200,169,110,0.15);
    border-radius: 6px; padding: 1.8rem 2rem;
    position: relative; overflow: hidden;
  }}
  .result-glow {{
    position: absolute; inset: 0; opacity: 0.07;
    transition: background 0.5s; pointer-events: none;
  }}
  .result-glow.setosa    {{ background: radial-gradient(ellipse at 85% 50%, var(--setosa),     transparent 65%); }}
  .result-glow.versicolor{{ background: radial-gradient(ellipse at 85% 50%, var(--versicolor), transparent 65%); }}
  .result-glow.virginica {{ background: radial-gradient(ellipse at 85% 50%, var(--virginica),  transparent 65%); }}

  .result-top {{
    display: flex; align-items: flex-end;
    justify-content: space-between; margin-bottom: 1.6rem; position: relative;
  }}
  .result-label {{ font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(247,241,232,0.28); margin-bottom: 0.35rem; }}
  .result-name {{ font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 400; font-style: italic; line-height: 1; }}
  .result-name.setosa     {{ color: var(--setosa); }}
  .result-name.versicolor {{ color: var(--versicolor); }}
  .result-name.virginica  {{ color: var(--virginica); }}

  .result-conf-block {{ text-align: right; }}
  .conf-label-sm {{ font-size: 0.62rem; color: rgba(247,241,232,0.28); letter-spacing: 0.12em; text-transform: uppercase; }}
  .result-conf-big {{ font-family: 'Playfair Display', serif; font-size: 2.6rem; font-weight: 400; color: var(--gold-light); line-height: 1; }}

  .bars-label {{ font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(247,241,232,0.18); margin-bottom: 0.9rem; }}
  .bar-row {{ display: grid; grid-template-columns: 100px 1fr 44px; align-items: center; gap: 0.9rem; margin-bottom: 0.7rem; }}
  .bar-name {{ font-size: 0.73rem; font-style: italic; font-family: 'Playfair Display', serif; color: rgba(247,241,232,0.45); }}
  .bar-track {{ height: 3px; background: rgba(247,241,232,0.07); border-radius: 2px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 2px; width: 0; transition: width 0.7s cubic-bezier(0.22,0.61,0.36,1); }}
  .bar-fill.setosa     {{ background: var(--setosa); }}
  .bar-fill.versicolor {{ background: var(--versicolor); }}
  .bar-fill.virginica  {{ background: var(--virginica); }}
  .bar-pct {{ font-size: 0.72rem; color: rgba(247,241,232,0.38); text-align: right; }}

  /* Spinner */
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .spinner {{
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid rgba(200,169,110,0.3);
    border-top-color: var(--gold); border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle; margin-right: 0.4rem;
  }}

  footer {{ margin-top: 3rem; text-align: center; font-size: 0.65rem; color: rgba(247,241,232,0.18); letter-spacing: 0.08em; }}

  @media (max-width: 480px) {{ .steppers {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="hero-bg"></div>

<div class="page">

  <header>
    <div class="eyebrow">Iris Species Predictor</div>
    <h1>Identify the <em>Species</em></h1>
    <p class="subtitle">Random Forest · 100 Trees · 70/30 Split</p>
  </header>

  <div class="steppers">

    <div class="stepper-cell sepal">
      <span class="stepper-label"><span class="part-pip"></span>Sepal Length</span>
      <div class="stepper-control">
        <button class="step-btn" data-field="sepal_length" data-dir="-1">−</button>
        <div class="step-display">
          <input class="step-value" id="val-sepal_length" name="sl" type="number" value="{current_sl:.1f}" min="4.3" max="7.9" step="0.1">
          <span class="step-unit">centimetres</span>
        </div>
        <button class="step-btn" data-field="sepal_length" data-dir="1">+</button>
      </div>
    </div>

    <div class="stepper-cell sepal">
      <span class="stepper-label"><span class="part-pip"></span>Sepal Width</span>
      <div class="stepper-control">
        <button class="step-btn" data-field="sepal_width" data-dir="-1">−</button>
        <div class="step-display">
          <input class="step-value" id="val-sepal_width" name="sw" type="number" value="{current_sw:.1f}" min="2.0" max="4.4" step="0.1">
          <span class="step-unit">centimetres</span>
        </div>
        <button class="step-btn" data-field="sepal_width" data-dir="1">+</button>
      </div>
    </div>

    <div class="stepper-cell petal">
      <span class="stepper-label"><span class="part-pip"></span>Petal Length</span>
      <div class="stepper-control">
        <button class="step-btn" data-field="petal_length" data-dir="-1">−</button>
        <div class="step-display">
          <input class="step-value" id="val-petal_length" name="pl" type="number" value="{current_pl:.1f}" min="1.0" max="6.9" step="0.1">
          <span class="step-unit">centimetres</span>
        </div>
        <button class="step-btn" data-field="petal_length" data-dir="1">+</button>
      </div>
    </div>

    <div class="stepper-cell petal">
      <span class="stepper-label"><span class="part-pip"></span>Petal Width</span>
      <div class="stepper-control">
        <button class="step-btn" data-field="petal_width" data-dir="-1">−</button>
        <div class="step-display">
          <input class="step-value" id="val-petal_width" name="pw" type="number" value="{current_pw:.1f}" min="0.1" max="2.5" step="0.1">
          <span class="step-unit">centimetres</span>
        </div>
        <button class="step-btn" data-field="petal_width" data-dir="1">+</button>
      </div>
    </div>

  </div>

  <div class="bottom-bar">
    <span class="samples-label">Try</span>
    <button class="pill" onclick="fillSample(5.1,3.5,1.4,0.2)">setosa</button>
    <button class="pill" onclick="fillSample(6.0,2.9,4.5,1.5)">versicolor</button>
    <button class="pill" onclick="fillSample(6.3,3.3,6.0,2.5)">virginica</button>
  </div>

  <button type="button" class="predict-btn" id="predict-btn" onclick="predict()">Identify Species</button>

  <!-- Result panel (pre-populated if Python already ran a prediction) -->
  <div id="result-panel" class="result-panel" style="display:none;">
    <div class="result-inner">
      <div class="result-glow" id="result-glow"></div>
      <div class="result-top">
        <div>
          <div class="result-label">Identified Species</div>
          <div class="result-name" id="result-name">—</div>
        </div>
        <div class="result-conf-block">
          <div class="conf-label-sm">Confidence</div>
          <div class="result-conf-big" id="result-conf">—</div>
        </div>
      </div>
      <div class="bars-label">All species probabilities</div>
      <div class="bar-row">
        <div class="bar-name">I. setosa</div>
        <div class="bar-track"><div class="bar-fill setosa" id="bar-setosa"></div></div>
        <div class="bar-pct" id="pct-setosa">—</div>
      </div>
      <div class="bar-row">
        <div class="bar-name">I. versicolor</div>
        <div class="bar-track"><div class="bar-fill versicolor" id="bar-versicolor"></div></div>
        <div class="bar-pct" id="pct-versicolor">—</div>
      </div>
      <div class="bar-row">
        <div class="bar-name">I. virginica</div>
        <div class="bar-track"><div class="bar-fill virginica" id="bar-virginica"></div></div>
        <div class="bar-pct" id="pct-virginica">—</div>
      </div>
    </div>
  </div>

  <footer>
    Iris Dataset &nbsp;·&nbsp; scikit-learn RandomForestClassifier &nbsp;·&nbsp;
    &nbsp;·&nbsp; Served by Streamlit
  </footer>
</div>

<script>
// ── Pre-filled result from Python (if any) ──────────────────────────────────
const PRE_RESULT = {pred_json};
if (PRE_RESULT) showResult(PRE_RESULT);
document.querySelectorAll('.step-btn, .pill').forEach(btn => btn.type = 'button');

// ── Stepper buttons with hold-to-repeat ─────────────────────────────────────
const FEATURES = ['sepal_length','sepal_width','petal_length','petal_width'];
let holdTimer = null, holdInterval = null;

function step(field, dir) {{
  const input = document.getElementById('val-' + field);
  let v = Math.round((parseFloat(input.value) + dir * 0.1) * 10) / 10;
  v = Math.min(parseFloat(input.max), Math.max(parseFloat(input.min), v));
  input.value = v.toFixed(1);
}}

document.querySelectorAll('.step-btn').forEach(btn => {{
  const field = btn.dataset.field;
  const dir   = parseInt(btn.dataset.dir);

  const startHold = () => {{
    btn.classList.add('held');
    holdTimer = setTimeout(() => {{
      holdInterval = setInterval(() => step(field, dir), 80);
    }}, 350);
  }};
  const stopHold = () => {{
    btn.classList.remove('held');
    clearTimeout(holdTimer);
    clearInterval(holdInterval);
  }};

  btn.addEventListener('mousedown',  startHold);
  btn.addEventListener('touchstart', startHold, {{ passive: true }});
  btn.addEventListener('mouseup',    stopHold);
  btn.addEventListener('mouseleave', stopHold);
  btn.addEventListener('touchend',   stopHold);
  btn.addEventListener('click', () => step(field, dir));
}});

// ── Predict: send values to Streamlit via Streamlit's postMessage ────────────
function predict() {{
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Identifying…';

  const sl = parseFloat(document.getElementById('val-sepal_length').value);
  const sw = parseFloat(document.getElementById('val-sepal_width').value);
  const pl = parseFloat(document.getElementById('val-petal_length').value);
  const pw = parseFloat(document.getElementById('val-petal_width').value);
  const requestId = Date.now().toString() + '-' + Math.random().toString(16).slice(2);

  window.parent.postMessage({{
    isStreamlitMessage: true,
    type: 'streamlit:setComponentValue',
    value: {{ action: 'predict', sl, sw, pl, pw, request_id: requestId }}
  }}, '*');
}}

function fillSample(sl, sw, pl, pw) {{
  document.getElementById('val-sepal_length').value = sl.toFixed(1);
  document.getElementById('val-sepal_width').value  = sw.toFixed(1);
  document.getElementById('val-petal_length').value = pl.toFixed(1);
  document.getElementById('val-petal_width').value  = pw.toFixed(1);
}}

function showResult(data) {{
  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';
  panel.style.animation = 'none';
  void panel.offsetWidth;
  panel.style.animation = '';

  const nameEl = document.getElementById('result-name');
  nameEl.className = 'result-name ' + data.species;
  nameEl.textContent = data.species.charAt(0).toUpperCase() + data.species.slice(1);

  document.getElementById('result-conf').textContent = data.confidence + '%';
  document.getElementById('result-glow').className   = 'result-glow ' + data.species;

  const probs = data.probabilities;
  ['setosa','versicolor','virginica'].forEach(cls => {{
    const pct = ((probs[cls] || 0) * 100).toFixed(1);
    const bar = document.getElementById('bar-' + cls);
    if (bar) {{
      // Animate bar from 0 → value
      bar.style.width = '0%';
      requestAnimationFrame(() => {{ bar.style.width = pct + '%'; }});
    }}
    const pctEl = document.getElementById('pct-' + cls);
    if (pctEl) pctEl.textContent = pct + '%';
  }});

  // Restore button
  const btn = document.getElementById('predict-btn');
  if (btn) {{ btn.disabled = false; btn.textContent = 'Identify Species'; }}

  // Restore input values from query params if available
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('sl')) document.getElementById('val-sepal_length').value = parseFloat(urlParams.get('sl')).toFixed(1);
  if (urlParams.get('sw')) document.getElementById('val-sepal_width').value  = parseFloat(urlParams.get('sw')).toFixed(1);
  if (urlParams.get('pl')) document.getElementById('val-petal_length').value = parseFloat(urlParams.get('pl')).toFixed(1);
  if (urlParams.get('pw')) document.getElementById('val-petal_width').value  = parseFloat(urlParams.get('pw')).toFixed(1);
}}
</script>
</body>
</html>
"""

# Render at full viewport height (scroll internally if needed)
component_value = components.html(html_page, height=900, scrolling=True)

if (
    isinstance(component_value, dict)
    and component_value.get("action") == "predict"
):
    request_id = component_value.get("request_id")
    if request_id != st.session_state.get("last_request_id"):
        try:
            sl = float(component_value.get("sl", default_values["sl"]))
            sw = float(component_value.get("sw", default_values["sw"]))
            pl = float(component_value.get("pl", default_values["pl"]))
            pw = float(component_value.get("pw", default_values["pw"]))
        except (TypeError, ValueError):
            sl = current_sl
            sw = current_sw
            pl = current_pl
            pw = current_pw

        st.session_state["current_sl"] = sl
        st.session_state["current_sw"] = sw
        st.session_state["current_pl"] = pl
        st.session_state["current_pw"] = pw
        st.session_state["pred_result"] = run_prediction(sl, sw, pl, pw)
        st.session_state["last_request_id"] = request_id
        st.rerun()
