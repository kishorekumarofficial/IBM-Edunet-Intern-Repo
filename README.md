# Sustainable Energy AI/ML - Power Consumption Prediction Model

## 📌 Project Overview
This project focuses on **Sustainable Energy Management** by leveraging **AI/ML models** to analyze and predict power consumption patterns.  
It supports **Regression, Classification, and Clustering** flows, helping in forecasting, anomaly detection, and efficient energy utilization.

The system is implemented as an **interactive Streamlit web application**, where users can upload their own dataset (CSV) and explore various machine learning techniques.

---

## 🎯 Learning Objectives / Goals
- To build a **data-driven approach** for sustainable energy management.
- To analyze **power consumption datasets** using AI/ML.
- To apply **predictive modeling** for future energy demand.
- To perform **K-Means clustering & visualization** for pattern discovery.
- To evaluate model performance with metrics such as **Confusion Matrix, Accuracy, and Elbow Method**.
- To create a **user-friendly web interface** for experimentation.

---

## 🛠 Tools & Technologies Used
- **Python 3.10+**
- **Streamlit** – for interactive web app
- **Pandas, NumPy** – for data preprocessing
- **Matplotlib, Seaborn** – for visualization
- **Scikit-learn** – ML models, clustering, evaluation metrics
- **XGBoost** – for advanced regression/classification
- **Jupyter Notebook / VS Code** – development environment

---

## 📂 Methodology
1. **Data Collection**  
   - Input: `power_consumption.csv` (sample dataset included)  
   - Columns include: `DateTime`, `Temperature`, `Humidity`, `Wind Speed`, `Diffuse Flows`, `Zone 1/2/3`

2. **Data Preprocessing**  
   - Handle missing values  
   - Convert `DateTime` into usable features (hour, day, month)  
   - Scaling numerical features for clustering  

3. **Exploratory Data Analysis (EDA)**  
   - Correlation heatmap  
   - Time series trends  
   - Feature distributions  

4. **Modeling**  
   - **Regression Models:** Linear Regression, XGBoost Regressor  
   - **Classification Models:** Decision Tree, Logistic Regression, Random Forest  
   - **Clustering:** K-Means (Elbow method for optimal K)  

5. **Evaluation**  
   - Regression: RMSE, MAE, R² Score  
   - Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
   - Clustering: Elbow Method, Cluster visualization  

6. **Deployment**  
   - Interactive **Streamlit App** where users can upload their dataset and test models.  

---

## 🚩 Problem Statement
Energy demand is continuously rising, and traditional methods of monitoring power consumption are inefficient.  
There is a need for a **predictive and analytical system** to optimize energy utilization for **sustainability and cost reduction**.

---

## 💡 Solution
We propose an **AI/ML-driven Power Consumption Prediction Model** that:  
- Analyzes historical consumption data.  
- Predicts future energy demand.  
- Detects anomalies in usage.  
- Identifies consumption patterns using clustering.  
- Provides a **web-based platform** for real-time analysis.  

---

## 📊 Results & Visuals
- Heatmap of correlations between features.  
- Time-series plots of energy consumption.  
- Elbow curve to determine optimal number of clusters.  
- Confusion Matrix for classification results.  
- Predicted vs Actual plots for regression models.  

---

## ✅ Conclusion
- AI/ML models can **accurately forecast energy demand** and optimize usage.  
- **Clustering** reveals hidden consumption patterns.  
- The system provides a **scalable, user-friendly interface** for researchers, students, and industries.  
- Future scope includes integration with **IoT devices** and **real-time monitoring systems**.  

---
1. Install Dependencies
pip install -r requirements.txt

2. Run Streamlit App
streamlit run app.py

📂 Project Structure
📦 Sustainable-Energy-AI-ML
 ┣ 📜 app.py                # Streamlit application
 ┣ 📜 requirements.txt      # Dependencies
 ┣ 📜 README.md             # Project documentation
 ┣ 📂 data
 ┃ ┗ 📜 power_consumption.csv (sample dataset)
 ┣ 📂 notebooks
 ┃ ┗ 📜 analysis.ipynb      # Exploratory analysis & testing
 ┗ 📂 models                # Saved models (optional)


**👨‍💻 Author
Kishore Kumar**

