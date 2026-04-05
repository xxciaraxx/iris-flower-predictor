import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components

st.set_page_config(page_title='Iris Species Identifier', page_icon='🌸', layout='wide')

st.markdown(
    """
    <style>
      .css-18e3th9, .css-1d391kg, .main, .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
      }
      .stApp, .reportview-container, .main {
        background: #0f1c14 !important;
      }
      iframe {
        width: 100% !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_model():
    with open('rf_model.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv('iris.csv')

model = load_model()
dataset = load_dataset()

rows = []
for index, row in dataset.iterrows():
    rows.append(
        f'<tr class="dataset-row" data-species="{row["species"]}">'
        f'<td>{index + 1}</td>'
        f'<td>{row["sepal_length"]}</td>'
        f'<td>{row["sepal_width"]}</td>'
        f'<td>{row["petal_length"]}</td>'
        f'<td>{row["petal_width"]}</td>'
        f'<td>{row["species"]}</td>'
        '</tr>'
    )

dataset_rows = ''.join(rows)

html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iris Species Identifier</title>
<link rel="icon" href="iris-icon.jpg">
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
  html, body {{ min-height:100%; }}
  body {{ font-family:'DM Sans', sans-serif; min-height:100vh; background:var(--dark); color:var(--cream); overflow-x:hidden; }}
  .hero-bg {{ position:fixed; inset:0; z-index:0; background:linear-gradient(160deg, rgba(15,28,20,0.95) 0%, rgba(15,28,20,0.62) 100%); }}
  .page {{ position:relative; z-index:1; min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:3rem 1.5rem 4rem; }}
  header {{ text-align:center; margin-bottom:3rem; }}
  .eyebrow {{ font-size:0.75rem; letter-spacing:0.26em; text-transform:uppercase; color:var(--gold); display:inline-flex; align-items:center; gap:0.8rem; margin-bottom:0.9rem; }}
  .eyebrow::before, .eyebrow::after {{ content:''; width:34px; height:1px; background:var(--gold); opacity:0.45; }}
  h1 {{ font-family:'Playfair Display', serif; font-size:clamp(2.2rem, 5vw, 3.6rem); font-weight:400; line-height:1.05; margin-bottom:0.4rem; }}
  h1 em {{ font-style:italic; color:var(--gold-light); }}
  .subtitle {{ font-size:0.86rem; color:rgba(247,241,232,0.42); font-weight:300; letter-spacing:0.03em; }}
  .steppers {{ display:grid; grid-template-columns:repeat(2,1fr); gap:1px; width:100%; max-width:560px; background:rgba(200,169,110,0.12); border:1px solid rgba(200,169,110,0.12); border-radius:6px 6px 0 0; overflow:hidden; }}
  .stepper-cell {{ background:rgba(10,20,13,0.75); backdrop-filter:blur(20px); padding:1.6rem 1.4rem; display:flex; flex-direction:column; align-items:center; gap:0.8rem; position:relative; }}
  .stepper-cell::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; opacity:0; transition:opacity 0.3s; }}
  .stepper-cell.sepal::before {{ background:linear-gradient(90deg, transparent, var(--setosa), transparent); }}
  .stepper-cell.petal::before {{ background:linear-gradient(90deg, transparent, var(--virginica), transparent); }}
  .stepper-cell:hover::before, .stepper-cell:focus-within::before {{ opacity:1; }}
  .stepper-label {{ font-size:0.65rem; font-weight:500; letter-spacing:0.2em; text-transform:uppercase; color:rgba(247,241,232,0.35); display:flex; align-items:center; gap:0.4rem; }}
  .part-pip {{ width:5px; height:5px; border-radius:50%; flex-shrink:0; }}
  .sepal .part-pip {{ background:var(--setosa); }}
  .petal .part-pip {{ background:var(--virginica); }}
  .stepper-control {{ display:flex; align-items:center; width:100%; gap:0; }}
  .step-btn {{ flex-shrink:0; width:38px; height:38px; background:rgba(200,169,110,0.08); border:1px solid rgba(200,169,110,0.2); border-radius:4px; color:var(--gold); font-size:1.35rem; font-weight:300; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:background 0.15s, border-color 0.15s, transform 0.1s; user-select:none; }}
  .step-btn:hover {{ background:rgba(200,169,110,0.18); border-color:rgba(200,169,110,0.5); }}
  .step-btn:active, .step-btn.held {{ transform:scale(0.91); background:rgba(200,169,110,0.28); }}
  .step-display {{ flex:1; text-align:center; }}
  .step-value {{ font-family:'Playfair Display', serif; font-size:2.1rem; font-weight:400; color:var(--gold-light); line-height:1; display:block; width:100%; text-align:center; border:none; background:transparent; outline:none; caret-color:var(--gold); }}
  .step-unit {{ font-size:0.6rem; color:rgba(247,241,232,0.2); letter-spacing:0.1em; margin-top:0.15rem; display:block; }}
  .bottom-bar {{ width:100%; max-width:560px; background:rgba(10,20,13,0.75); backdrop-filter:blur(20px); border:1px solid rgba(200,169,110,0.12); border-top:none; border-radius:0 0 6px 6px; padding:0.85rem 1.4rem; display:flex; align-items:center; gap:0.6rem; }}
  .samples-label {{ font-size:0.6rem; letter-spacing:0.18em; text-transform:uppercase; color:rgba(247,241,232,0.2); flex-shrink:0; }}
  .pill {{ background:none; border:1px solid rgba(247,241,232,0.12); border-radius:20px; padding:0.28rem 0.85rem; font-size:0.7rem; font-style:italic; color:rgba(247,241,232,0.38); cursor:pointer; letter-spacing:0.04em; transition:all 0.2s; font-family:'Playfair Display', serif; }}
  .pill:hover {{ border-color:var(--gold); color:var(--gold-light); }}
  .predict-btn {{ width:100%; max-width:560px; margin-top:1.2rem; padding:1rem; background:transparent; border:1px solid var(--gold); border-radius:4px; color:var(--gold-light); font-family:'DM Sans', sans-serif; font-size:0.78rem; font-weight:500; letter-spacing:0.22em; text-transform:uppercase; cursor:pointer; }}
  .predict-btn:hover {{ color:var(--dark); background:rgba(200,169,110,0.08); }}
  .result-panel {{ width:100%; max-width:560px; margin-top:1.4rem; display:none; animation:riseIn 0.45s cubic-bezier(0.22,0.61,0.36,1) both; }}
  @keyframes riseIn {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:translateY(0); }} }}
  .result-inner {{ background:rgba(10,20,13,0.82); backdrop-filter:blur(24px); border:1px solid rgba(200,169,110,0.15); border-radius:6px; padding:1.8rem 2rem; position:relative; overflow:hidden; }}
  .result-glow {{ position:absolute; inset:0; opacity:0.07; transition:background 0.5s; pointer-events:none; }}
  .result-glow.setosa {{ background:radial-gradient(ellipse at 85% 50%, var(--setosa), transparent 65%); }}
  .result-glow.versicolor {{ background:radial-gradient(ellipse at 85% 50%, var(--versicolor), transparent 65%); }}
  .result-glow.virginica {{ background:radial-gradient(ellipse at 85% 50%, var(--virginica), transparent 65%); }}
  .result-top {{ display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:1.6rem; }}
  .result-label {{ font-size:0.62rem; letter-spacing:0.2em; text-transform:uppercase; color:rgba(247,241,232,0.28); margin-bottom:0.35rem; }}
  .result-name {{ font-family:'Playfair Display', serif; font-size:2.1rem; font-weight:400; font-style:italic; line-height:1; }}
  .result-name.setosa {{ color:var(--setosa); }}
  .result-name.versicolor {{ color:var(--versicolor); }}
  .result-name.virginica {{ color:var(--virginica); }}
  .result-conf-block {{ text-align:right; }}
  .result-conf-big {{ font-family:'Playfair Display', serif; font-size:2.6rem; font-weight:400; color:var(--gold-light); margin:0; }}
  .bars-label {{ font-size:0.62rem; letter-spacing:0.18em; text-transform:uppercase; color:rgba(247,241,232,0.18); margin-bottom:0.9rem; }}
  .bar-row {{ display:grid; grid-template-columns:100px 1fr 44px; align-items:center; gap:0.9rem; margin-bottom:0.7rem; }}
  .bar-name {{ font-size:0.73rem; font-style:italic; font-family:'Playfair Display', serif; color:rgba(247,241,232,0.45); }}
  .bar-track {{ height:3px; background:rgba(247,241,232,0.07); border-radius:2px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:2px; width:0; transition:width 0.7s cubic-bezier(0.22,0.61,0.36,1); }}
  .bar-fill.setosa {{ background:var(--setosa); }}
  .bar-fill.versicolor {{ background:var(--versicolor); }}
  .bar-fill.virginica {{ background:var(--virginica); }}
  .bar-pct {{ font-size:0.72rem; color:rgba(247,241,232,0.38); text-align:right; }}
  .dataset-card {{ background:rgba(10,20,13,0.82); border:1px solid rgba(200,169,110,0.15); border-radius:16px; padding:1.8rem; width:100%; max-width:1000px; }}
  .dataset-header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin-bottom:1rem; }}
  .dataset-eyebrow {{ font-family:'Playfair Display', serif; font-size:0.95rem; margin-bottom:0.3rem; color:#e8d5a8; }}
  .dataset-copy {{ font-size:0.82rem; color:rgba(247,241,232,0.5); margin:0; }}
  .dataset-meta {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.14em; color:rgba(247,241,232,0.22); white-space:nowrap; }}
  .dataset-tabs {{ display:flex; flex-wrap:wrap; gap:0.65rem; margin-bottom:1.2rem; }}
  .dataset-tab {{ background:none; border:1px solid rgba(247,241,232,0.12); border-radius:999px; padding:0.65rem 0.95rem; color:rgba(247,241,232,0.7); cursor:pointer; font-size:0.72rem; }}
  .dataset-tab.active, .dataset-tab:hover {{ border-color:#c8a96e; color:#e8d5a8; background:rgba(200,169,110,0.12); }}
  .dataset-table-wrap {{ overflow-x:auto; max-height:360px; }}
  .dataset-table {{ width:100%; border-collapse:collapse; min-width:680px; background:rgba(247,241,232,0.03); }}
  .dataset-table th, .dataset-table td {{ padding:0.95rem 0.9rem; text-align:left; border-bottom:1px solid rgba(247,241,232,0.08); color:rgba(247,241,232,0.72); }}
  .dataset-table th {{ color:rgba(247,241,232,0.36); font-weight:500; letter-spacing:0.12em; text-transform:uppercase; font-size:0.72rem; }}
  .dataset-table tbody tr:hover {{ background:rgba(247,241,232,0.05); }}
  footer {{ margin-top:3rem; text-align:center; font-size:0.65rem; color:rgba(247,241,232,0.18); letter-spacing:0.08em; }}
  @media (max-width: 480px) {{ .steppers {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="hero-bg"></div>
<div class="page">
  <header>
    <div class="eyebrow">Botanical Classifier</div>
    <h1>Identify the <em>Iris</em></h1>
    <p class="subtitle">Random Forest · 100 Trees · 70/30 Split · 100% Test Accuracy</p>
  </header>
  <div class="steppers">
    <div class="stepper-cell sepal">
      <span class="stepper-label"><span class="part-pip"></span>Sepal Length</span>
      <div class="stepper-control">
        <button class="step-btn" data-field="sepal_length" data-dir="-1">−</button>
        <div class="step-display">
          <input class="step-value" id="val-sepal_length" type="number" value="5.8" min="4.3" max="7.9" step="0.1">
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
          <input class="step-value" id="val-sepal_width" type="number" value="3.0" min="2.0" max="4.4" step="0.1">
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
          <input class="step-value" id="val-petal_length" type="number" value="3.8" min="1.0" max="6.9" step="0.1">
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
          <input class="step-value" id="val-petal_width" type="number" value="1.2" min="0.1" max="2.5" step="0.1">
          <span class="step-unit">centimetres</span>
        </div>
        <button class="step-btn" data-field="petal_width" data-dir="1">+</button>
      </div>
    </div>
  </div>
  <div class="bottom-bar">
    <span class="samples-label">Try</span>
    <button class="pill" onclick="fillSample(5.1,3.5,1.4,0.2)">I. setosa</button>
    <button class="pill" onclick="fillSample(6.0,2.9,4.5,1.5)">I. versicolor</button>
    <button class="pill" onclick="fillSample(6.3,3.3,6.0,2.5)">I. virginica</button>
  </div>
  <button class="predict-btn" onclick="identify()">Identify Species</button>
  <div class="result-panel" id="result-panel">
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
  <section class="dataset-card">
    <div class="dataset-header">
      <div>
        <div class="dataset-eyebrow">Iris Dataset Reference</div>
        <p class="dataset-copy">Use this table to compare your inputs against real measurements in the dataset.</p>
      </div>
      <div class="dataset-meta">150 samples · 3 species · 4 features</div>
    </div>
    <div class="dataset-tabs">
      <button class="dataset-tab active" data-filter="all">🌸 All Species</button>
      <button class="dataset-tab" data-filter="setosa">🌿 Setosa</button>
      <button class="dataset-tab" data-filter="versicolor">🍃 Versicolor</button>
      <button class="dataset-tab" data-filter="virginica">🌺 Virginica</button>
    </div>
    <div class="dataset-table-wrap">
      <table class="dataset-table">
        <thead>
          <tr>
            <th>#</th>
            <th>sepal length (cm)</th>
            <th>sepal width (cm)</th>
            <th>petal length (cm)</th>
            <th>petal width (cm)</th>
            <th>species</th>
          </tr>
        </thead>
        <tbody id="dataset-table-body">
          {dataset_rows}
        </tbody>
      </table>
    </div>
  </section>
  <footer>
    Iris Dataset · scikit-learn RandomForestClassifier · Model: rf_model.json · Photo: Unsplash
  </footer>
</div>
<script>
const RF_MODEL = {model_json};
const FEATURES = ['sepal_length','sepal_width','petal_length','petal_width'];
let identified = false;

function traverseTree(node, s) {{
  if (node.leaf) return node.class;
  return s[node.feature] <= node.threshold ? traverseTree(node.left, s) : traverseTree(node.right, s);
}}

function predictRF(s) {{
  const votes = {{ setosa:0, versicolor:0, virginica:0 }};
  RF_MODEL.trees.forEach(t => {{ votes[traverseTree(t, s)]++; }});
  const n = RF_MODEL.trees.length;
  const probs = {{}};
  Object.keys(votes).forEach(k => probs[k] = votes[k] / n);
  const predicted = Object.keys(votes).reduce((a,b) => votes[a] > votes[b] ? a : b);
  return {{ predicted, probs }};
}}

function setValues(values) {{
  FEATURES.forEach(f => {{
    document.getElementById('val-' + f).value = values[f].toFixed(1);
  }});
}}

function evaluatePrediction() {{
  const s = {{}};
  FEATURES.forEach(f => {{
    const el = document.getElementById('val-' + f);
    let v = parseFloat(el.value);
    if (isNaN(v)) v = parseFloat(el.min);
    s[f] = Math.min(parseFloat(el.max), Math.max(parseFloat(el.min), v));
  }});
  return predictRF(s);
}}

function renderPrediction({{ predicted, probs }}) {{
  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';
  panel.style.animation = 'none';
  void panel.offsetWidth;
  panel.style.animation = '';
  const nameEl = document.getElementById('result-name');
  nameEl.className = 'result-name ' + predicted;
  nameEl.textContent = predicted.charAt(0).toUpperCase() + predicted.slice(1);
  document.getElementById('result-conf').textContent = (probs[predicted] * 100).toFixed(1) + '%';
  document.getElementById('result-glow').className = 'result-glow ' + predicted;
  RF_MODEL.classes.forEach(cls => {{
    const pct = (probs[cls] * 100).toFixed(1);
    document.getElementById('bar-' + cls).style.width = pct + '%';
    document.getElementById('pct-' + cls).textContent = pct + '%';
  }});
}}

function computePrediction() {{
  if (identified) {{
    renderPrediction(evaluatePrediction());
  }}
}}

function identify() {{
  identified = true;
  renderPrediction(evaluatePrediction());
}}

function fillSample(sl, sw, pl, pw) {{
  const values = {{ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw }};
  setValues(values);
  identify();
}}

let holdTimer = null, holdInterval = null;

document.querySelectorAll('.step-btn').forEach(btn => {{
  const field = btn.dataset.field;
  const dir = parseInt(btn.dataset.dir);
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
  btn.addEventListener('mousedown', startHold);
  btn.addEventListener('touchstart', startHold, {{ passive: true }});
  btn.addEventListener('mouseup', stopHold);
  btn.addEventListener('mouseleave', stopHold);
  btn.addEventListener('touchend', stopHold);
  btn.addEventListener('click', () => step(field, dir));
}});

function step(field, dir) {{
  const input = document.getElementById('val-' + field);
  let v = Math.round((parseFloat(input.value) + dir * 0.1) * 10) / 10;
  v = Math.min(parseFloat(input.max), Math.max(parseFloat(input.min), v));
  input.value = v.toFixed(1);
  computePrediction();
}}

FEATURES.forEach(f => {{
  document.getElementById('val-' + f).addEventListener('change', computePrediction);
}});

const tabButtons = document.querySelectorAll('.dataset-tab');
tabButtons.forEach(btn => {{
  btn.addEventListener('click', () => {{
    tabButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    document.querySelectorAll('.dataset-row').forEach(row => {{
      row.style.display = filter === 'all' || row.dataset.species === filter ? '' : 'none';
    }});
  }});
}});
</script>
</body>
</html>'''

html = html_template.format(model_json=json.dumps(model), dataset_rows=dataset_rows)

components.html(html, height=1600, scrolling=True)
