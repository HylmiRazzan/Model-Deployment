import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Penentuan direktori utama secara dinamis
BASE_DIR = Path(__file__).resolve().parent

def load_file(filename):
    """Mencari file model di folder artifacts atau di folder utama."""
    path_artifacts = BASE_DIR / "artifacts" / filename
    path_root = BASE_DIR / filename
    
    if path_artifacts.exists():
        return joblib.load(path_artifacts)
    elif path_root.exists():
        return joblib.load(path_root)
    else:
        st.error(f"File '{filename}' tidak ditemukan di folder artifacts maupun root.")
        st.stop()

# 1. Memuat Pipeline Keseluruhan (Preprocessor + Model sekaligus)
# Pastikan Anda memanggil model terbaik Anda dari tahap sebelumnya
model_pipeline = load_file("score_prediction_xgboost.pkl")

def get_input():
    with st.form("input_form"):
        st.subheader("Input Data Nasabah")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 100, 30)
            annual_income = st.number_input("Annual Income", 0.0, 1000000.0, 50000.0)
            monthly_inhand = st.number_input("Monthly Inhand Salary", 0.0, 100000.0, 4000.0)
            month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
            occupation = st.selectbox("Occupation", ["Engineer", "Teacher", "Doctor", "Lawyer", "Scientist", "Mechanic", "Writer", "Media_Manager", "Entrepreneur", "Developer", "Manager", "Other"])

        with col2:
            num_bank_accounts = st.number_input("Num Bank Accounts", 0, 20, 2)
            num_credit_card = st.number_input("Num Credit Card", 0, 20, 2)
            interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 15.0)
            num_of_loan = st.number_input("Num of Loan", 0, 20, 2)
            delay_from_due = st.number_input("Delay from due date (days)", 0, 100, 10)

        with col3:
            num_delayed_payment = st.number_input("Num of Delayed Payment", 0, 50, 5)
            changed_credit_limit = st.number_input("Changed Credit Limit", 0.0, 100.0, 10.0)
            num_credit_inquiries = st.number_input("Num Credit Inquiries", 0, 50, 2)
            outstanding_debt = st.number_input("Outstanding Debt", 0.0, 100000.0, 1000.0)
            credit_utilization = st.number_input("Credit Utilization Ratio", 0.0, 100.0, 30.0)

        st.divider()

        col4, col5 = st.columns(2)

        with col4:
            total_emi = st.number_input("Total EMI per month", 0.0, 50000.0, 500.0)
            amount_invested = st.number_input("Amount invested monthly", 0.0, 50000.0, 100.0)
            monthly_balance = st.number_input("Monthly Balance", 0.0, 100000.0, 300.0)
            
            # Input menggunakan bulan sesuai permintaan
            credit_history_months = st.number_input("Credit History Age", 0, 600, 48)

        with col5:
            credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
            payment_min = st.selectbox("Payment of Min Amount", ["No", "Yes"])
            spent = st.selectbox("Spent", ["low", "high"])
            value_payment = st.selectbox("Value Payment", ["small", "medium", "large"])

        st.divider()
        st.subheader("Informasi Jenis Pinjaman")
        
        col7, col8, col9 = st.columns(3)
        with col7:
            auto_loan = st.selectbox("Auto Loan", ["No", "Yes"])
            credit_builder = st.selectbox("Credit-Builder Loan", ["No", "Yes"])
            debt_consolidation = st.selectbox("Debt Consolidation Loan", ["No", "Yes"])
            home_equity = st.selectbox("Home Equity Loan", ["No", "Yes"])

        with col8:
            mortgage = st.selectbox("Mortgage Loan", ["No", "Yes"])
            not_specified = st.selectbox("Not Specified", ["No", "Yes"])
            payday = st.selectbox("Payday Loan", ["No", "Yes"])
            personal = st.selectbox("Personal Loan", ["No", "Yes"])

        with col9:
            student_loan = st.selectbox("Student Loan", ["No", "Yes"])
            loan_missing = st.selectbox("Loan Missing", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Now")

    if submitted:
        # Fungsi internal untuk konversi Yes/No menjadi 1/0
        def to_bin(val):
            return 1 if val == "Yes" else 0
            
        # Mengembalikan DataFrame satu baris untuk langsung dieksekusi model
        return pd.DataFrame([{
            "Age": age,
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_inhand,
            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate,
            "Num_of_Loan": num_of_loan,
            "Delay_from_due_date": delay_from_due,
            "Num_of_Delayed_Payment": num_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": num_credit_inquiries,
            "Outstanding_Debt": outstanding_debt,
            "Total_EMI_per_month": total_emi,
            "Amount_invested_monthly": amount_invested,
            "Monthly_Balance": monthly_balance,
            "Credit_Utilization_Ratio": credit_utilization,
            
            # Dibagi 12 karena model dilatih menggunakan format tahun
            "Credit_History_Age": credit_history_months, 
            
            "Month": month,
            "Occupation": occupation,
            "Credit_Mix": credit_mix,
            "Payment_of_Min_Amount": payment_min,
            "Spent": spent,
            "Value_Payment": value_payment,
            
            # Pemanggilan fungsi to_bin untuk logika biner pinjaman
            "Auto Loan": to_bin(auto_loan),
            "Credit-Builder Loan": to_bin(credit_builder),
            "Debt Consolidation Loan": to_bin(debt_consolidation),
            "Home Equity Loan": to_bin(home_equity),
            "Mortgage Loan": to_bin(mortgage),
            "Not Specified": to_bin(not_specified),
            "Payday Loan": to_bin(payday),
            "Personal Loan": to_bin(personal),
            "Student Loan": to_bin(student_loan),
            "loan_missing": to_bin(loan_missing)
        }])
    return None

def main():
    st.set_page_config(page_title="Credit Score Predictor", layout="wide")
    st.title("Sistem Klasifikasi Skor Kredit")
    
    input_df = get_input()

    if input_df is not None:
        
        # Eksekusi langsung ke pipeline (sudah termasuk transform dan predict)
        prediction_code = model_pipeline.predict(input_df)[0]
        
        # Mapping dari output numerik model (0, 1, 2) ke label teks
        mapping_balik = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        status = mapping_balik.get(prediction_code, "Unknown")

        st.markdown("---")
        st.subheader("Prediction Results")

        if status == "Good":
            st.success(f"### Prediction: {status} Credit Score")
        elif status == "Standard":
            st.warning(f"### Prediction: {status} Credit Score")
        else:
            st.error(f"### Prediction: {status} Credit Score")

if __name__ == "__main__":
    main()