import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Iris Species Identifier", page_icon="🌸", layout="wide")

@st.cache_data
def load_model():
    with open('rf_model.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv('iris.csv')

def traverse_tree(node, features):
    if node['leaf']:
        return node['class']
    if features[node['feature']] <= node['threshold']:
        return traverse_tree(node['left'], features)
    return traverse_tree(node['right'], features)

def predict_rf(model, features):
    votes = {'setosa': 0, 'versicolor': 0, 'virginica': 0}
    for tree in model['trees']:
        votes[traverse_tree(tree, features)] += 1
    total = len(model['trees'])
    probs = {k: v / total for k, v in votes.items()}
    return max(votes, key=votes.get), probs

def initialize_state():
    defaults = {
        'sepal_length': 5.8,
        'sepal_width': 3.0,
        'petal_length': 3.8,
        'petal_width': 1.2,
        'identified': False,
        'predicted': None,
        'probs': {'setosa': 0.0, 'versicolor': 0.0, 'virginica': 0.0},
        'dataset_filter': 'All Species'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_sample(sepal_length, sepal_width, petal_length, petal_width):
    st.session_state.sepal_length = sepal_length
    st.session_state.sepal_width = sepal_width
    st.session_state.petal_length = petal_length
    st.session_state.petal_width = petal_width
    st.session_state.identified = False

def identify():
    features = {
        'sepal_length': st.session_state.sepal_length,
        'sepal_width': st.session_state.sepal_width,
        'petal_length': st.session_state.petal_length,
        'petal_width': st.session_state.petal_width
    }
    predicted, probs = predict_rf(model, features)
    st.session_state.predicted = predicted
    st.session_state.probs = probs
    st.session_state.identified = True

initialize_state()

CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');
html, body, .stApp, .main, .block-container {background: #0f1c14; color: #f7f1e8;}
body {font-family: 'DM Sans', sans-serif;}
.stButton>button {background:#c8a96e; color:#0f1c14; border:none; border-radius:12px; padding:0.8rem 1.1rem; font-weight:600;}
.stButton>button:hover {background:#e8d5a8;}
.css-1q8dd3e p, .css-1q8dd3e span, .css-1q8dd3e label {color:#dcd7c9;}
.hero {text-align:center; padding:2rem 0 1rem;}
.eyebrow {font-size:0.75rem; letter-spacing:0.2em; text-transform:uppercase; color:#c8a96e; margin-bottom:0.9rem;}
.hero-title {font-family:'Playfair Display', serif; font-size:3.4rem; margin-bottom:0.3rem;}
.hero-title em {font-style:italic; color:#e8d5a8;}
.subtitle {color:rgba(247,241,232,0.45); margin-bottom:0.25rem;}
.card {background:rgba(10,20,13,0.85); border:1px solid rgba(200,169,110,0.14); border-radius:22px; padding:1.8rem; margin-bottom:1.5rem;}
.card h2 {font-family:'Playfair Display', serif; margin-bottom:1rem;}
.row-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1.4rem;}
@media(max-width:900px){.row-grid{grid-template-columns:1fr;}}
.sample-buttons .stButton>button {width:100%; margin-bottom:0.6rem;}
.result-card {position:relative; overflow:hidden;}
.result-glow {position:absolute; inset:0; opacity:0.08; pointer-events:none;}
.result-top {display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.5rem;}
.result-label {color:rgba(247,241,232,0.3); letter-spacing:0.18em; text-transform:uppercase; font-size:0.72rem; margin-bottom:0.35rem;}
.result-name {font-family:'Playfair Display', serif; font-size:2.5rem; font-style:italic; margin:0;}
.result-name.setosa {color:#7ecba1;}
.result-name.versicolor {color:#6ba3d6;}
.result-name.virginica {color:#c57ebf;}
.result-conf-big {font-family:'Playfair Display', serif; font-size:2.6rem; color:#e8d5a8; margin:0;}
.bars-label {color:rgba(247,241,232,0.22); letter-spacing:0.18em; text-transform:uppercase; font-size:0.72rem; margin-bottom:1rem;}
.bar-row {display:grid; grid-template-columns:110px 1fr 56px; gap:0.9rem; align-items:center; margin-bottom:0.7rem;}
.bar-name {font-size:0.78rem; font-style:italic; font-family:'Playfair Display', serif; color:rgba(247,241,232,0.48);}
.bar-track {height:6px; background:rgba(247,241,232,0.08); border-radius:4px; overflow:hidden;}
.bar-fill {height:100%; border-radius:4px; width:0; transition:width 0.6s ease;}
.bar-fill.setosa{background:#7ecba1;}
.bar-fill.versicolor{background:#6ba3d6;}
.bar-fill.virginica{background:#c57ebf;}
.bar-pct {font-size:0.8rem; color:rgba(247,241,232,0.42); text-align:right;}
.dataset-panel {background:rgba(10,20,13,0.82); border:1px solid rgba(200,169,110,0.14); border-radius:22px; padding:1.8rem;}
.dataset-header {display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin-bottom:1.2rem;}
.dataset-eyebrow {font-family:'Playfair Display', serif; font-size:1rem; margin-bottom:0.35rem; color:#e8d5a8;}
.dataset-copy {color:rgba(247,241,232,0.5); margin:0;}
.dataset-meta {color:rgba(247,241,232,0.24); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.13em;}
.dataset-tabs {display:flex; flex-wrap:wrap; gap:0.7rem; margin-bottom:1rem;}
.dataset-tabs label {background:rgba(247,241,232,0.05); color:#dcd7c9; border:1px solid rgba(247,241,232,0.12); border-radius:999px; padding:0.55rem 0.95rem; cursor:pointer; transition:all 0.2s;}
.dataset-tabs input:checked + label {background:rgba(200,169,110,0.15); color:#f7f1e8; border-color:rgba(200,169,110,0.35);}
.dataset-table-wrapper {max-height:340px; overflow:auto; border-radius:16px;}
.dataset-table {width:100%; border-collapse:collapse; min-width:680px;}
.dataset-table th, .dataset-table td {padding:0.9rem 0.85rem; text-align:left; color:rgba(247,241,232,0.7); border-bottom:1px solid rgba(247,241,232,0.08);}
.dataset-table th {color:rgba(247,241,232,0.35); font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;}
.dataset-table tbody tr:hover {background:rgba(247,241,232,0.05);}
</style>
'''

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Botanical Classifier</div>
      <h1 class="hero-title">Identify the <em>Iris</em></h1>
      <p class="subtitle">Random Forest · 100 Trees · 70/30 Split · 100% Test Accuracy</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
dataset = load_dataset()

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>Input Measurements</h2>', unsafe_allow_html=True)
    cols = st.columns(2, gap='large')
    with cols[0]:
        st.slider('Sepal Length (cm)', 4.3, 7.9, st.session_state.sepal_length, 0.1, key='sepal_length')
        st.slider('Sepal Width (cm)', 2.0, 4.4, st.session_state.sepal_width, 0.1, key='sepal_width')
    with cols[1]:
        st.slider('Petal Length (cm)', 1.0, 6.9, st.session_state.petal_length, 0.1, key='petal_length')
        st.slider('Petal Width (cm)', 0.1, 2.5, st.session_state.petal_width, 0.1, key='petal_width')

    st.markdown('<div class="sample-buttons">', unsafe_allow_html=True)
    st.markdown('<h2>Try Samples</h2>', unsafe_allow_html=True)
    sample_cols = st.columns(3)
    samples = {
        'I. setosa': (5.1, 3.5, 1.4, 0.2),
        'I. versicolor': (6.0, 2.9, 4.5, 1.5),
        'I. virginica': (6.3, 3.3, 6.0, 2.5)
    }
    for col, (name, values) in zip(sample_cols, samples.items()):
        col.button(name, on_click=apply_sample, args=values)

    if st.button('Identify Species', on_click=identify):
        pass

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.identified:
    predicted = st.session_state.predicted
    probs = st.session_state.probs
    st.markdown(
        f"""
        <div class='card result-card'>
          <div class='result-glow {predicted}'></div>
          <div class='result-top'>
            <div>
              <div class='result-label'>Identified Species</div>
              <div class='result-name {predicted}'>{predicted.capitalize()}</div>
            </div>
            <div class='result-conf-block'>
              <div class='result-label'>Confidence</div>
              <div class='result-conf-big'>{probs[predicted]*100:.1f}%</div>
            </div>
          </div>
          <div class='bars-label'>All species probabilities</div>
          <div class='bar-row'>
            <div class='bar-name'>I. setosa</div>
            <div class='bar-track'><div class='bar-fill setosa' style='width:{probs['setosa']*100:.1f}%'></div></div>
            <div class='bar-pct'>{probs['setosa']*100:.1f}%</div>
          </div>
          <div class='bar-row'>
            <div class='bar-name'>I. versicolor</div>
            <div class='bar-track'><div class='bar-fill versicolor' style='width:{probs['versicolor']*100:.1f}%'></div></div>
            <div class='bar-pct'>{probs['versicolor']*100:.1f}%</div>
          </div>
          <div class='bar-row'>
            <div class='bar-name'>I. virginica</div>
            <div class='bar-track'><div class='bar-fill virginica' style='width:{probs['virginica']*100:.1f}%'></div></div>
            <div class='bar-pct'>{probs['virginica']*100:.1f}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='card'><p style='color:rgba(247,241,232,0.45);'>Click Identify Species to see a prediction.</p></div>", unsafe_allow_html=True)

st.markdown('<div class="dataset-panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="dataset-header"><div><div class="dataset-eyebrow">Iris Dataset Reference</div><p class="dataset-copy">Use this table to compare your inputs against real measurements in the dataset.</p></div><div class="dataset-meta">150 samples · 3 species · 4 features</div></div>',
    unsafe_allow_html=True,
)

selected_filter = st.radio('Filter by Species', ['All Species', 'Setosa', 'Versicolor', 'Virginica'], horizontal=True, key='dataset_filter')
if selected_filter == 'All Species':
    filtered_data = dataset
else:
    species_map = {'Setosa': 'setosa', 'Versicolor': 'versicolor', 'Virginica': 'virginica'}
    filtered_data = dataset[dataset['species'] == species_map[selected_filter]]

st.markdown('<div class="dataset-table-wrapper">', unsafe_allow_html=True)
st.dataframe(filtered_data, height=340, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')
st.markdown('<p style="color:rgba(247,241,232,0.45);">Iris Dataset · scikit-learn RandomForestClassifier · Model: rf_model.json</p>', unsafe_allow_html=True)
