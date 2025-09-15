import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import xgboost as xgb

st.set_page_config(page_title="⚡ Smart Energy Management", layout="wide")
st.title("⚡ Power Consumption — Modeling & Analysis")

st.markdown("""
Upload the **power consumption CSV** (or use sample).  
This app supports **regression, classification, and clustering** with visualization tools.
""")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------- Correlation Heatmap ----------
    st.subheader("📈 Feature Correlation")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.info("⚠️ No numeric columns available for correlation heatmap.")

    # ---------- Regression ----------
    st.subheader("📈 Regression: Predict Zone_1 Consumption")
    if "Zone_1" in df.columns:
        features = [col for col in df.columns if col not in ["DateTime", "Zone_1"]]
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        y = df["Zone_1"]

        if not X.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("**R² Score:**", round(r2_score(y_test, y_pred), 4))
            st.write("**MSE:**", round(mean_squared_error(y_test, y_pred), 4))

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6, color="green")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Zone 1: Actual vs Predicted")
            st.pyplot(fig)
        else:
            st.warning("No numeric features available for regression.")
    else:
        st.info("⚠️ Column `Zone 1` not found in dataset. Skipping regression.")

    # ---------- Clustering ----------
    st.subheader("🌀 K-Means Clustering on Zone Consumption")
    expected_features = ["Zone_1", "Zone_2", "Zone_3"]
    cluster_features = [col for col in expected_features if col in df.columns]

    if cluster_features:
        scaled_for_clust = StandardScaler().fit_transform(df[cluster_features])

        # Elbow Method
        sse = []
        k_range = range(1, 10)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_for_clust)
            sse.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_range, sse, marker="o")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("SSE (Inertia)")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig)

        # Fit final model
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(scaled_for_clust)

        st.write("✅ Clustering Results with features:", cluster_features)
        st.dataframe(df[cluster_features + ["Cluster"]].head())

        fig, ax = plt.subplots()
        ax.scatter(df[cluster_features[0]], df[cluster_features[1]], 
                   c=df["Cluster"], cmap="viridis", alpha=0.7)
        ax.set_xlabel(cluster_features[0])
        ax.set_ylabel(cluster_features[1] if len(cluster_features) > 1 else cluster_features[0])
        ax.set_title("Clustering of Zones")
        st.pyplot(fig)
    else:
        st.info("⚠️ No `Zone_1`, `Zone_2`, or `Zone_3` columns found. Skipping clustering.")

    # ---------- Classification (Optional Example) ----------
    st.subheader("🧩 Classification Example")
    if "Zone_1" in df.columns:
        # Binary classification: High vs Low consumption
        df["High_Consumption"] = (df["Zone_1"] > df["Zone_1"].median()).astype(int)
        features = [col for col in df.columns if col not in ["DateTime", "Zone_1", "High_Consumption"]]
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        y = df["High_Consumption"]

        if not X.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = LogisticRegression(max_iter=500)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.warning("No numeric features available for classification.")
else:
    st.info("⬆️ Please upload a CSV file to continue.")
