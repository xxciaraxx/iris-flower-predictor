import streamlit as st
import pandas as pd
import json
import os

# Load the model
@st.cache_data
def load_model():
    with open('rf_model.json', 'r') as f:
        return json.load(f)

# Load the dataset
@st.cache_data
def load_dataset():
    return pd.read_csv('iris.csv')

# Prediction function
def predict_rf(model, features):
    votes = {'setosa': 0, 'versicolor': 0, 'virginica': 0}
    for tree in model['trees']:
        votes[traverse_tree(tree, features)] += 1
    total = len(model['trees'])
    probs = {k: v / total for k, v in votes.items()}
    predicted = max(votes, key=votes.get)
    return predicted, probs

def traverse_tree(node, features):
    if node['leaf']:
        return node['class']
    feature = node['feature']
    threshold = node['threshold']
    if features[feature] <= threshold:
        return traverse_tree(node['left'], features)
    else:
        return traverse_tree(node['right'], features)

# Streamlit app
st.set_page_config(page_title="Iris Species Identifier", page_icon="🌸")

st.title("Identify the Iris")
st.markdown("Random Forest · 100 Trees · 70/30 Split · 100% Test Accuracy")

# Load data
model = load_model()
dataset = load_dataset()

# Input section
st.header("Input Measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 3.8, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

features = {
    'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width
}

# Sample buttons
st.subheader("Try Samples")
samples = {
    "I. setosa": (5.1, 3.5, 1.4, 0.2),
    "I. versicolor": (6.0, 2.9, 4.5, 1.5),
    "I. virginica": (6.3, 3.3, 6.0, 2.5)
}

cols = st.columns(3)
for i, (name, vals) in enumerate(samples.items()):
    if cols[i].button(name):
        features['sepal_length'] = vals[0]
        features['sepal_width'] = vals[1]
        features['petal_length'] = vals[2]
        features['petal_width'] = vals[3]
        st.rerun()

# Predict button
if st.button("Identify Species"):
    predicted, probs = predict_rf(model, features)
    
    st.success(f"Identified Species: **{predicted.capitalize()}**")
    st.metric("Confidence", f"{probs[predicted]*100:.1f}%")
    
    st.subheader("All Species Probabilities")
    for species in ['setosa', 'versicolor', 'virginica']:
        st.progress(probs[species])
        st.write(f"{species.capitalize()}: {probs[species]*100:.1f}%")

# Dataset reference
st.header("Iris Dataset Reference")
st.write("Use this table to compare your inputs against real measurements in the dataset.")

# Filter tabs
filter_options = ["All Species", "Setosa", "Versicolor", "Virginica"]
selected_filter = st.radio("Filter by Species", filter_options, horizontal=True)

if selected_filter == "All Species":
    filtered_data = dataset
else:
    species_map = {"Setosa": "setosa", "Versicolor": "versicolor", "Virginica": "virginica"}
    filtered_data = dataset[dataset['species'] == species_map[selected_filter]]

st.dataframe(filtered_data, height=400)

st.markdown("---")
st.write("Iris Dataset · scikit-learn RandomForestClassifier · Model: rf_model.json")