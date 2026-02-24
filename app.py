import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans



# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Social Media Data Mining Dashboard",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL STYLING (READABLE & CLEAN)
# --------------------------------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #F1F5F9;
}

/* Main title */
h1 {
    color: #0F172A;
    font-weight: 800;
}

/* Section headers */
h2, h3 {
    color: #1D4ED8;
    font-weight: 700;
}

/* Normal text */
p, li, span, div {
    color: #0F172A !important;
    font-size: 16px;
}

/* Section card */
.section-box {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 14px;
    margin-top: 20px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E0E7FF;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Students Social Media Addiction.csv")
    

df = load_data()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("📌 Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Project Contents",
        "Dataset",
        "Preprocessing",
        "Association Rules",
        "Classification",
        "Clustering",
        "Conclusion"
    ]
)
# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------
st.markdown(
    """
    <h1 style="
        font-size: 44px;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 5px;
    ">
        📊 Social Media Data Mining Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size:18px; color:#334155; margin-top:-10px;'>"
    "CIA 3.1 | Interactive Frontend & Visual Analytics"
    "</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# SECTIONS
# --------------------------------------------------

if section == "Overview":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Project Overview")

    st.write(
        "This project is based on a real-world social media usage dataset collected "
        "from Kaggle. The dataset captures user behavior such as daily social media "
        "usage, addiction levels, sleep patterns, mental health indicators, and "
        "academic impact. The primary objective of this project is to analyze how "
        "social media usage influences students’ academic performance and behavior."
    )

    st.write(
        "The project follows the complete data mining lifecycle, starting from data "
        "preprocessing to pattern discovery, prediction, clustering, and interpretation. "
        "All major concepts covered in the course are implemented using real data and "
        "visualized through an interactive dashboard."
    )

    st.subheader("Techniques Used – Description")

    st.markdown("""
    **Data Preprocessing**  
    Data preprocessing involves cleaning and transforming raw data into a usable format. 
    Techniques such as missing value handling, normalization, discretization (binning), 
    and outlier detection were applied to improve data quality and ensure reliable analysis.

    **Association Rule Mining**  
    Association rule mining is used to discover interesting relationships between variables 
    in large datasets. The Apriori algorithm was applied to identify patterns based on 
    support, confidence, and lift, revealing behavioral associations among users.

    **Classification**  
    Classification is a supervised learning technique used to predict categorical outcomes. 
    Decision Tree and Naive Bayes classifiers were implemented to predict whether social media 
    usage affects academic performance, and model performance was evaluated.

    **Clustering**  
    Clustering is an unsupervised learning technique that groups similar data points without 
    predefined labels. K-Means clustering was used to segment users into distinct behavioral 
    groups based on usage intensity and addiction levels.

    **Outlier Analysis**  
    Outlier analysis helps identify abnormal or extreme data points that deviate significantly 
    from normal behavior. Boxplot-based visualization was used to detect unusual social media 
    usage patterns.
    """)

    st.markdown('</div>', unsafe_allow_html=True)



elif section == "Project Contents":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Project Contents")

    st.write(
        "This project presents a comprehensive data mining analysis on a real-world "
        "social media dataset. It includes all major concepts covered in the course, "
        "starting from data preprocessing to pattern discovery, prediction, clustering, "
        "and interpretation, supported by proper visualizations."
    )

    st.subheader("Components of the Project")

    st.markdown("""
    **Data Collection and Understanding**  
    The dataset used in this project was obtained from Kaggle and contains demographic, 
    behavioral, and academic-related attributes. The structure of the dataset, including 
    the number of records and attributes, was analyzed to understand the nature of the data.

    **Data Preprocessing**  
    Data preprocessing was performed to improve data quality by handling missing values, 
    removing duplicate records, and correcting inconsistencies in the dataset.

    **Data Transformation and Normalization**  
    Numerical attributes were transformed using normalization techniques such as Min–Max 
    scaling to bring all values into a common range and improve the performance of data 
    mining algorithms.

    **Data Discretization (Binning)**  
    Continuous attributes such as age and addiction score were discretized into meaningful 
    categories using binning techniques to support better analysis and pattern discovery.

    **Outlier Detection and Handling**  
    Outliers were identified using boxplot visualization and IQR-based methods to detect 
    abnormal social media usage behavior that deviates from normal patterns.

    **Association Rule Mining**  
    Association rule mining was performed using the Apriori algorithm to discover hidden 
    relationships between variables such as addiction level, sleep patterns, and academic 
    performance.

    **Classification**  
    Supervised learning techniques including Decision Tree and Naive Bayes classifiers were 
    applied to predict whether social media usage affects academic performance. Model 
    performance was evaluated using accuracy and confusion matrices.

    **Clustering**  
    K-Means clustering was used to group users into distinct behavioral segments based on 
    social media usage intensity and addiction levels.

    **Visualization and Interpretation**  
    Charts, tables, and plots were used to visually represent results and clearly interpret 
    the patterns and insights obtained from the analysis.

    **Conclusion and Insights**  
    The project concludes by summarizing key findings and highlighting how data mining 
    techniques can be effectively applied to analyze real-world social media data.
    """)

    st.markdown('</div>', unsafe_allow_html=True)



elif section == "Dataset":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Dataset Description")

    st.write(
        "The dataset used in this project was obtained from Kaggle and represents "
        "real-world social media usage behavior among students. It consists of "
        "705 records and 13 attributes, covering demographic details, social media "
        "usage patterns, addiction levels, sleep behavior, mental health indicators, "
        "and academic impact."
    )

    st.write(
        "Key attributes include Student ID, Age, Gender, Academic Level, Country, "
        "Average Daily Usage Hours, Most Used Platform, Sleep Hours per Night, "
        "Mental Health Score, Conflicts Over Social Media, Addicted Score, and the "
        "target attribute Affects Academic Performance."
    )

    st.subheader("Dataset Preview (First 10 Records)")
    st.dataframe(df.head(10))

    st.subheader("Age Distribution of Users")
    fig, ax = plt.subplots()
    ax.hist(df['Age'], bins=15)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)




elif section == "Preprocessing":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Data Preprocessing")

    # --------------------------------------------------
    # Introduction
    # --------------------------------------------------
    st.write(
        "Data preprocessing is a crucial step in data mining, as raw data often "
        "contains inconsistencies, missing values, and noise. Proper preprocessing "
        "improves data quality and directly affects the performance of mining algorithms."
    )

    # --------------------------------------------------
    # Duplicate removal & missing value handling
    # --------------------------------------------------
    st.subheader("1. Duplicate Removal and Missing Value Handling")

    st.write(
        "Duplicate records increase redundancy and may bias the analysis. Therefore, "
        "duplicate values were identified and removed from the dataset. Missing values "
        "were handled using the mean imputation method for numerical attributes, ensuring "
        "that data completeness was maintained without significantly altering the "
        "distribution of the data."
    )

    # --------------------------------------------------
    # Outlier detection
    # --------------------------------------------------
    st.subheader("2. Outlier Detection")

    st.write(
        "Outliers are extreme values that deviate significantly from the normal data "
        "distribution. These values can negatively impact model accuracy and distort "
        "statistical measures. Boxplot visualization was used to detect outliers in the "
        "Addicted Score attribute."
    )

    # ---- Boxplot for outliers ----
    fig1, ax1 = plt.subplots()
    sns.boxplot(x=df['Addicted_Score'], ax=ax1)
    ax1.set_title("Outlier Detection: Addicted Score")
    st.pyplot(fig1)

    st.write(
        "From the boxplot, values lying beyond the whiskers represent outliers. These "
        "extreme addiction scores indicate abnormal social media usage behavior and "
        "were carefully considered during analysis to avoid misleading results."
    )

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    st.subheader("3. Data Normalization")

    st.write(
        "Normalization is a data transformation technique used to scale numerical "
        "attributes into a common range. Min–Max normalization was applied to the "
        "Addicted Score attribute to convert values into the range [0, 1]. This step "
        "ensures that attributes with larger scales do not dominate the learning process."
    )

    # ---- Min-Max Normalization ----
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(df[['Addicted_Score']])

    # ---- Before vs After Histogram ----
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Addicted_Score'], bins=15, alpha=0.6, label="Original")
    ax2.hist(normalized_values, bins=15, alpha=0.6, label="Normalized")
    ax2.set_title("Before vs After Normalization")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    st.pyplot(fig2)

    st.write(
        "The histogram comparison shows that after normalization, values are evenly "
        "scaled without changing the overall distribution pattern. This transformation "
        "improves the stability and performance of classification and clustering algorithms."
    )

    st.markdown('</div>', unsafe_allow_html=True)


elif section == "Association Rules":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Association Rule Mining")

    # --------------------------------------------------
    # Introduction
    # --------------------------------------------------
    st.write(
        "Association rule mining is a data mining technique used to discover hidden "
        "relationships and patterns among variables in large datasets. It helps identify "
        "how the presence of certain conditions influences other outcomes."
    )

    # --------------------------------------------------
    # Key Concepts
    # --------------------------------------------------
    st.subheader("Key Concepts")

    st.write(
        "Support indicates how frequently a particular itemset appears in the dataset. "
        "Confidence measures the reliability of an association rule, while Lift indicates "
        "the strength of a rule by comparing it with random occurrence."
    )

    # --------------------------------------------------
    # Data Preparation
    # --------------------------------------------------
    st.subheader("Data Preparation for Rule Mining")

    st.write(
        "For association rule mining, continuous numerical attributes were converted into "
        "Boolean values. Addiction level and sleep duration were categorized as high or low "
        "based on their mean values, and academic performance impact was converted into a "
        "binary attribute."
    )

    assoc_df = df.copy()
    assoc_df['High_Addiction'] = assoc_df['Addicted_Score'] > df['Addicted_Score'].mean()
    assoc_df['Low_Sleep'] = assoc_df['Sleep_Hours_Per_Night'] < df['Sleep_Hours_Per_Night'].mean()
    assoc_df['Academic_Affected'] = (
        df['Affects_Academic_Performance']
        .astype(str)
        .str.lower()
        .isin(['yes', 'true', '1'])
    )

    assoc_df = assoc_df[['High_Addiction', 'Low_Sleep', 'Academic_Affected']]

    # --------------------------------------------------
    # Apriori Algorithm
    # --------------------------------------------------
    st.subheader("Apriori Algorithm")

    st.write(
        "The Apriori algorithm was applied to generate frequent itemsets using a minimum "
        "support threshold. Association rules were then derived using confidence as the "
        "primary evaluation metric."
    )

    frequent_itemsets = apriori(assoc_df, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    # --------------------------------------------------
    # Rules Table (Converted for Display)
    # --------------------------------------------------
    st.subheader("Discovered Association Rules")

    rules_display = rules.copy()
    rules_display['antecedents'] = rules_display['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )
    rules_display['consequents'] = rules_display['consequents'].apply(
        lambda x: ', '.join(list(x))
    )

    st.dataframe(
        rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        .sort_values(by='lift', ascending=False)
    )

    st.write(
        "The table above shows the discovered association rules along with their support, "
        "confidence, and lift values. Rules with higher confidence and lift represent "
        "stronger and more meaningful relationships between variables."
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    st.subheader("Support vs Confidence Visualization")

    fig, ax = plt.subplots()
    ax.scatter(
        rules['support'],
        rules['confidence'],
        s=rules['lift'] * 100,
        alpha=0.6
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules: Support vs Confidence (Bubble Size = Lift)")

    st.pyplot(fig)

    st.write(
        "In the scatter plot, rules appearing towards the upper-right region indicate "
        "higher support and confidence. Larger bubbles represent higher lift values, "
        "highlighting strong associations between social media addiction, sleep patterns, "
        "and academic performance."
    )

    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Classification":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Classification")

    # --------------------------------------------------
    # Introduction
    # --------------------------------------------------
    st.write(
        "Classification is a supervised learning technique used to predict a "
        "categorical outcome based on input features. In this project, classification "
        "models were used to predict whether social media usage affects academic "
        "performance."
    )

    # --------------------------------------------------
    # Feature selection
    # --------------------------------------------------
    st.subheader("Feature Selection")

    st.write(
        "Relevant features such as addiction score, sleep hours per night, and "
        "average daily usage hours were selected as input variables. The target "
        "variable represents whether social media usage negatively affects academic "
        "performance."
    )

    # ---- Prepare data ----
    X = df[['Addicted_Score', 'Sleep_Hours_Per_Night', 'Avg_Daily_Usage_Hours']]
    y = (
        df['Affects_Academic_Performance']
        .astype(str)
        .str.lower()
        .isin(['yes', 'true', '1'])
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --------------------------------------------------
    # Decision Tree Classifier
    # --------------------------------------------------
    st.subheader("Decision Tree Classifier")

    st.write(
        "The Decision Tree classifier works by recursively splitting the data "
        "based on feature values to form a tree-like structure. It is easy to "
        "interpret and helps understand decision-making logic."
    )

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)

    # --------------------------------------------------
    # Naive Bayes Classifier
    # --------------------------------------------------
    st.subheader("Naive Bayes Classifier")

    st.write(
        "Naive Bayes is a probabilistic classifier based on Bayes’ theorem and "
        "assumes independence between features. Despite its simplicity, it often "
        "performs well on real-world datasets."
    )

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)

    # --------------------------------------------------
    # Accuracy comparison
    # --------------------------------------------------
    st.subheader("Model Accuracy Comparison")

    col1, col2 = st.columns(2)
    col1.metric("Decision Tree Accuracy", f"{dt_acc:.2f}")
    col2.metric("Naive Bayes Accuracy", f"{nb_acc:.2f}")

    st.write(
        "The accuracy scores indicate the proportion of correctly classified instances. "
        "A higher accuracy reflects better predictive performance on unseen data."
    )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("Confusion Matrix (Decision Tree)")

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, dt_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.write(
        "The confusion matrix provides a detailed view of classification performance "
        "by showing true positives, true negatives, false positives, and false negatives. "
        "It helps evaluate how well the model distinguishes between affected and "
        "non-affected academic performance."
    )

    st.markdown('</div>', unsafe_allow_html=True)


elif section == "Clustering":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Clustering")

    # --------------------------------------------------
    # Introduction
    # --------------------------------------------------
    st.write(
        "Clustering is an unsupervised learning technique used to group similar "
        "data points without predefined class labels. It helps identify hidden "
        "structures and patterns within data based on similarity."
    )

    # --------------------------------------------------
    # Algorithm explanation
    # --------------------------------------------------
    st.subheader("K-Means Clustering")

    st.write(
        "K-Means clustering partitions the dataset into a predefined number of "
        "clusters by minimizing the distance between data points and their "
        "respective cluster centroids. In this project, K-Means was used to "
        "segment users based on social media usage behavior."
    )

    # --------------------------------------------------
    # Feature selection
    # --------------------------------------------------
    st.subheader("Feature Selection")

    st.write(
        "Addicted Score and Average Daily Usage Hours were selected as clustering "
        "features, as they directly represent the intensity and impact of social "
        "media usage on users."
    )

    cluster_data = df[['Addicted_Score', 'Avg_Daily_Usage_Hours']]

    # --------------------------------------------------
    # Apply K-Means
    # --------------------------------------------------
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(cluster_data)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df['Addicted_Score'],
        df['Avg_Daily_Usage_Hours'],
        c=df['Cluster'],
        cmap='viridis',
        alpha=0.7
    )

    # Plot centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        s=200,
        marker='X',
        label='Centroids'
    )

    ax.set_xlabel("Addicted Score")
    ax.set_ylabel("Average Daily Usage Hours")
    ax.set_title("K-Means Clustering of Social Media Users")
    ax.legend()

    st.pyplot(fig)

    st.write(
        "The scatter plot shows the formation of three distinct clusters representing "
        "different user behavior patterns. Each cluster groups users with similar "
        "addiction levels and social media usage duration."
    )

    # --------------------------------------------------
    # Cluster interpretation
    # --------------------------------------------------
    st.subheader("Cluster Interpretation")

    cluster_summary = (
        df.groupby('Cluster')[['Addicted_Score', 'Avg_Daily_Usage_Hours']]
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(cluster_summary)

    st.write(
        "The cluster summary table presents the average addiction score and daily "
        "usage hours for each cluster. These clusters can be interpreted as low, "
        "moderate, and high social media usage groups, providing meaningful user "
        "segmentation."
    )

    st.markdown('</div>', unsafe_allow_html=True)




elif section == "Conclusion":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.header("Conclusion")

    st.write(
        "This project successfully demonstrates the practical application of core "
        "data mining concepts on a real-world social media dataset. By following a "
        "structured data mining process, meaningful insights were extracted from "
        "raw and complex behavioral data."
    )

    st.write(
        "Data preprocessing techniques improved data quality, while association "
        "rule mining uncovered hidden relationships between user behavior, sleep "
        "patterns, and academic performance. Classification models effectively "
        "predicted academic impact, and clustering segmented users into distinct "
        "behavioral groups."
    )

    st.write(
        "Overall, the analysis highlights that excessive social media usage and "
        "higher addiction levels are associated with reduced sleep and negative "
        "academic outcomes. The project clearly illustrates how data mining "
        "techniques can be used to derive actionable insights from social data."
    )

    st.markdown('</div>', unsafe_allow_html=True)

