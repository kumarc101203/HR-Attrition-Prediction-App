import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# LOAD FILES
# ================================
model = pickle.load(open("attrition_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="HR Attrition App", layout="wide")

# ================================
# CUSTOM STYLE
# ================================
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# TITLE
# ================================
st.markdown("<h1 style='text-align: center;'>💼 HR Attrition Prediction</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Enter Employee Details")

age = st.sidebar.number_input("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.number_input("Distance From Home", 1, 30, 5)
job_level = st.sidebar.selectbox("Job Level", [1,2,3,4,5])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

job_involvement = st.sidebar.selectbox("Job Involvement", [1,2,3,4])
job_satisfaction = st.sidebar.selectbox("Job Satisfaction", [1,2,3,4])
work_life_balance = st.sidebar.selectbox("Work Life Balance", [1,2,3,4])
total_years = st.sidebar.number_input("Total Working Years", 0, 40, 5)
years_company = st.sidebar.number_input("Years At Company", 0, 40, 3)

# Convert categorical
overtime = 1 if overtime == "Yes" else 0

# ================================
# MAIN DASHBOARD
# ================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Employee Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**Income:** {monthly_income}")
    st.write(f"**Experience:** {total_years} years")

with col2:
    st.subheader("⚙️ Job Details")
    st.write(f"**Job Level:** {job_level}")
    st.write(f"**OverTime:** {overtime}")
    st.write(f"**Work-Life Balance:** {work_life_balance}")

# ================================
# PREDICTION
# ================================
if st.button("🚀 Predict Attrition"):

    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    input_df['Age'] = age
    input_df['MonthlyIncome'] = monthly_income
    input_df['DistanceFromHome'] = distance
    input_df['JobLevel'] = job_level
    input_df['OverTime'] = overtime
    input_df['JobInvolvement'] = job_involvement
    input_df['JobSatisfaction'] = job_satisfaction
    input_df['WorkLifeBalance'] = work_life_balance
    input_df['TotalWorkingYears'] = total_years
    input_df['YearsAtCompany'] = years_company

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ================================
    # RESULT
    # ================================
    st.subheader("📊 Prediction Result")

    st.progress(float(prob))
    st.write(f"Attrition Probability: {prob:.2%}")

    if prob > 0.7:
        st.error("🔴 High Risk: Employee likely to leave")
    elif prob > 0.4:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk: Employee likely to stay")

    # ================================
    # GRAPH (ONLY ONCE, CORRECT PLACE)
    # ================================
    st.subheader("📈 Attrition Insights")

    fig, ax = plt.subplots(figsize=(4,2.5))

    ax.bar(["No OT", "OverTime"], [10, 30])
    ax.set_ylabel("Attrition %")
    ax.set_title("Attrition vs OverTime")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # ================================
    # FEATURE IMPORTANCE
    # ================================
    st.subheader("🔍 Key Influencing Factors")

    importance = pd.Series(model.coef_[0], index=feature_columns)
    top_features = importance.abs().sort_values(ascending=False).head(5)

    st.write(top_features)

    # ================================
    # BUSINESS INSIGHT
    # ================================
    st.info("💡 Recommendation: Reduce overtime or improve salary to reduce attrition risk.")


# ================================
# MODEL INFO
# ================================
with st.expander("ℹ️ About Model"):
    st.write("""
    Model: Logistic Regression with SMOTE  
    Accuracy: ~76%  
    Recall: ~56%  
    ROC-AUC: ~0.75  
    """)
