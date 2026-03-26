# 💼 HR Attrition Prediction App

## 📌 Overview
This project is an end-to-end Machine Learning application that predicts whether an employee is likely to leave the company based on various factors such as age, salary, job satisfaction, and work-life balance.

The project includes:
- Data Analysis (EDA)
- Model Training with SMOTE (handling class imbalance)
- Multiple ML Models
- Model Evaluation & Tuning
- Interactive Web App using Streamlit

---

## 🚀 Features
- 🔮 Predict employee attrition (Yes / No)
- 📊 Shows probability of attrition
- 📈 Visual insights (Attrition vs OverTime)
- 🔍 Feature importance explanation
- 💡 Business recommendations
- 🎯 Risk classification (Low / Medium / High)

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- Removed irrelevant columns
- Handled categorical variables (Encoding)
- Feature scaling using StandardScaler

### 2. Handling Imbalance
- Applied **SMOTE** to balance dataset

### 3. Models Used
- Logistic Regression ✅ (Selected Best Model)
- Decision Tree
- Random Forest
- KNN

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~76% |
| Recall | ~56% |
| ROC-AUC | ~0.75 |

👉 Logistic Regression with SMOTE was selected as the best model due to better recall and balanced performance.

---

## 🖥️ Streamlit App

### 🔹 Input Features:
- Age
- Monthly Income
- Distance from Home
- Job Level
- OverTime
- Job Involvement
- Job Satisfaction
- Work-Life Balance
- Total Working Years
- Years at Company

### 🔹 Output:
- Attrition Prediction
- Probability Score
- Risk Level
- Feature Importance
- Business Insight

---

## 📂 Project Structure
```
├── app.py # Streamlit UI
├── train_model.py # Model training script
├── attrition_model.pkl # Trained model
├── scaler.pkl # Scaler
├── features.pkl # Feature columns
├── HR-Employee-Attrition.csv
├── requirements.txt
└── README.md

---
```
## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/hr-attrition-app.git
cd hr-attrition-app

Install Dependencies
pip install -r requirements.txt


Run Application
streamlit run app.py

```
💡 Business Insights
- Employees working overtime have significantly higher attrition
- Lower salary is linked to higher attrition risk
- Work-life balance plays a key role in retention


🏆 Conclusion

This project demonstrates how machine learning can be used to:
- Predict employee attrition
- Identify key risk factors
- Support HR decision-making


📌 Future Improvements
- Add more features to UI (Department, JobRole)
- Deploy on cloud (Streamlit Cloud / Render)
- Use advanced models (XGBoost, Neural Networks)


👤 Author

KUMAR C

⭐ If you like this project, give it a star!
