# My folder's structure ->

# ml_streamlit_app/
# │
# ├── app.py                   # Streamlit application code
# ├── model.pkl                # Trained model
# ├── requirements.txt         # Python packages required
# └── README.md                # (Optional) description of the app



# Training and saving the model..
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Loading data and training model..
iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

# Saving the model..
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Creating Streamlit app..
# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Page title
st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to classify the Iris species.")

# Feature inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]
probs = model.predict_proba(features)[0]

# Map target names
iris = load_iris()
target_name = iris.target_names[prediction]

st.subheader("🔍 Prediction:")
st.success(f"The predicted Iris species is **{target_name}**.")

# Show probability as bar chart
st.subheader("📊 Prediction Probabilities:")
df_probs = pd.DataFrame([probs], columns=iris.target_names)
st.bar_chart(df_probs.T)

# Show decision region (optional visualization)
st.subheader("🌐 Visualizing Input Data:")
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
fig, ax = plt.subplots()
ax.scatter(iris_df[iris.feature_names[0]], iris_df[iris.feature_names[1]],
           c=iris.target, cmap='viridis', alpha=0.5, label='Dataset')
ax.scatter(sepal_length, sepal_width, c='red', s=100, label='Your input', edgecolors='black')
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.legend()
st.pyplot(fig)
