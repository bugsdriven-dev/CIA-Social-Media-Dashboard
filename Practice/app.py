# ================================
# SOCIAL MEDIA ADDICTION ANALYZER
# FINAL COMPLETE VERSION (PATCHED)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from mpl_toolkits.mplot3d import Axes3D

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(
    page_title="Social Media Addiction Analyzer",
    page_icon="📱",
    layout="wide"
)

# ================================
# LOAD DATASET
# ================================

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))


# ================================
# ADD MISSING VALUES FOR DEMO
# ================================

df_missing = df.copy()

for col in df_missing.select_dtypes(include=np.number).columns[:3]:
    df_missing.loc[df_missing.sample(frac=0.1).index, col] = np.nan


# ================================
# SIDEBAR
# ================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/747/747376.png",
    width=120
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Preprocessing",
        "Normalization",
        "Clustering",
        "Association Rules",
        "Visualization",
        "Insights",
        "Prediction",
        "Conclusion"
    ]
)


# ================================
# HEADER
# ================================

st.markdown("""
<style>
.header {
background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
padding:30px;
border-radius:15px;
color:white;
text-align:center;
}
.card {
background: linear-gradient(135deg,#667eea,#764ba2);
padding:20px;
border-radius:15px;
color:white;
text-align:center;
font-size:20px;
font-weight:bold;
}
</style>

<div class="header">
<h1>📱 Social Media Addiction Analyzer</h1>
<p>AI Powered Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ================================
# OVERVIEW
# ================================

if menu == "Overview":

    col1,col2,col3,col4 = st.columns(4)

    col1.markdown(f'<div class="card">Rows<br>{df.shape[0]}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">Columns<br>{df.shape[1]}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">Missing<br>{df_missing.isnull().sum().sum()}</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="card">Features<br>{len(df.columns)}</div>', unsafe_allow_html=True)

    st.dataframe(df.head())


# ================================
# PREPROCESSING
# ================================

elif menu == "Data Preprocessing":

    st.title("Missing Values Table")

    st.dataframe(df_missing.isnull().sum())

    df_filled = df_missing.fillna(df_missing.mean(numeric_only=True))

    st.title("After Filling Missing Values")

    st.dataframe(df_filled.head())


# ================================
# NORMALIZATION (UPDATED LEGEND)
# ================================

elif menu == "Normalization":

    st.title("Normalization Graph")

    numeric_cols = df.select_dtypes(include=np.number).columns[:3]

    scaler = StandardScaler()

    before = df[numeric_cols].head(20)

    after = scaler.fit_transform(before)

    fig, ax = plt.subplots()

    ax.plot(before.values, linestyle="--", marker="o", label="Before Normalization")

    ax.plot(after, linestyle="-", marker="x", label="After Normalization")

    ax.legend()

    ax.set_title("Before vs After Normalization")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Values")

    st.pyplot(fig)


# ================================
# CLUSTERING
# ================================

elif menu == "Clustering":

    numeric_cols = df.select_dtypes(include=np.number).columns[:2]

    X = df[numeric_cols].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42)

    y = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_

    fig, ax = plt.subplots()

    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1],c=y,cmap="rainbow")

    ax.scatter(
        centroids[:,0],
        centroids[:,1],
        c="black",
        marker="X",
        s=300,
        label="Centroids"
    )

    ax.legend()

    st.pyplot(fig)


# ================================
# ASSOCIATION RULES (REAL APRIORI)
# ================================

elif menu == "Association Rules":

    st.title("Apriori Algorithm")

    df_bin = df.select_dtypes(include=np.number)

    df_bin = df_bin.apply(lambda x: x > x.mean())

    freq = apriori(df_bin, min_support=0.2, use_colnames=True)

    rules = association_rules(freq, metric="confidence", min_threshold=0.5)

    st.dataframe(rules)

    st.title("Support vs Confidence Graph")

    fig, ax = plt.subplots()

    ax.scatter(rules['support'], rules['confidence'], c="red")

    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")

    st.pyplot(fig)

    # FP Growth
    st.title("FP Growth")

    fp = fpgrowth(df_bin, min_support=0.2, use_colnames=True)

    st.dataframe(fp)


# ================================
# VISUALIZATION
# ================================

elif menu == "Visualization":

    chart = st.selectbox(
        "Chart Type",
        ["Histogram","Bar Chart","Scatter Plot","3D Scatter"]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns

    x = st.selectbox("X axis", numeric_cols)

    y = st.selectbox("Y axis", numeric_cols)

    fig = plt.figure()

    if chart == "Histogram":

        plt.hist(df[x])

        st.pyplot(fig)

    elif chart == "Bar Chart":

        plt.bar(df[x].head(20), df[y].head(20))

        st.pyplot(fig)

    elif chart == "Scatter Plot":

        plt.scatter(df[x], df[y])

        st.pyplot(fig)

    elif chart == "3D Scatter":

        z = st.selectbox("Z axis", numeric_cols)

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(df[x], df[y], df[z], c="blue")

        st.pyplot(fig)


# ================================
# INSIGHTS (CONFUSION MATRIX)
# ================================

elif menu == "Insights":

    st.title("Confusion Matrix")

    numeric_cols = df.select_dtypes(include=np.number).columns

    X = df[numeric_cols[:-1]]

    y = pd.qcut(df[numeric_cols[-1]], q=2, labels=[0,1])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    model = DecisionTreeClassifier()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    cm = confusion_matrix(y_test,pred)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay(cm).plot(ax=ax)

    st.pyplot(fig)

    st.write("Confusion Matrix shows prediction accuracy")


# ================================
# PREDICTION
# ================================

elif menu == "Prediction":

    numeric_cols = df.select_dtypes(include=np.number).columns

    X = df[numeric_cols[:-1]]

    y = pd.qcut(df[numeric_cols[-1]], q=2, labels=[0,1])

    model = DecisionTreeClassifier()

    model.fit(X,y)

    inputs = []

    for col in X.columns:

        val = st.number_input(col, float(X[col].min()), float(X[col].max()))

        inputs.append(val)

    if st.button("Predict"):

        result = model.predict([inputs])

        if result[0]==1:
            st.error("High Addiction Risk")
        else:
            st.success("Low Addiction Risk")


# ================================
# CONCLUSION
# ================================

elif menu == "Conclusion":

    st.title("Conclusion")

    st.write("""
This project analyzes social media addiction using:

• Preprocessing  
• Normalization  
• Clustering  
• Apriori Algorithm  
• FP Growth  
• Machine Learning  
• Prediction  

This helps identify addiction risk.
""")
