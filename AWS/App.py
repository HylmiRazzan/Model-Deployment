import json
import os
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

# 1. Pengaturan Identitas Endpoint SageMaker
# Secara otomatis mengambil nama dari EC2, atau menggunakan nama default jika dijalankan lokal
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "CSPredictionsRF-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# 2. Pembuatan Jalur Komunikasi ke AWS (Di-cache agar aplikasi tidak lambat)
@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

# 3. Fungsi Pengirim Data ke SageMaker
def invoke_endpoint(input_data: dict) -> dict:
    runtime = get_runtime_client()
    
    # Membungkus data ke dalam format JSON yang bisa dibaca oleh inference.py
    payload = {"instances": [input_data]}
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

# 4. Fungsi Antarmuka Pengguna (UI)
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
            credit_history_months = st.number_input("Credit History Age", 0, 600, 48)

        with col5:
            credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
            payment_min = st.selectbox("Payment of Min Amount", ["No", "Yes"])
            spent = st.selectbox("Spent", ["low", "high"])
            value_payment = st.selectbox("Value Payment", ["small", "medium", "large"])

        st.divider()
        st.subheader("Type of Loan")
        
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
        def to_bin(val):
            return 1 if val == "Yes" else 0
            
        # Perhatikan: Kita tidak lagi mengembalikan Pandas DataFrame, melainkan Dictionary biasa
        # Hal ini karena JSON lebih mudah menerima Dictionary bawaan Python
        return {
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
            "Credit_History_Age": credit_history_months, 
            "Month": month,
            "Occupation": occupation,
            "Credit_Mix": credit_mix,
            "Payment_of_Min_Amount": payment_min,
            "Spent": spent,
            "Value_Payment": value_payment,
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
        }
    return None

# 5. Program Utama
def main():
    st.set_page_config(page_title="Credit Score Predictor", layout="wide")
    st.title("Classification Credit Score System")
    
    input_data = get_input()

    if input_data is not None:
        try:
            # Mengirim data ke SageMaker
            result = invoke_endpoint(input_data)
            
        except NoCredentialsError:
            st.error("Gagal terhubung ke AWS: Kredensial tidak ditemukan. Pastikan EC2 memiliki IAM Role LabInstanceProfile.")
        except ClientError as e:
            st.error(f"Terjadi kesalahan pada server AWS: {e.response['Error'].get('Message', str(e))}")
        except Exception as e:
            st.error(f"Kesalahan tidak terduga: {str(e)}")
            
        else:
            # Mengambil jawaban dari SageMaker (Karena model kita mengembalikan nama label langsung seperti 'Good')
            status = result["labels"][0]
            probs = result["probabilities"][0]

            st.markdown("***")
            st.subheader("Prediction Results")

            # Menampilkan warna yang sesuai dengan hasil
            if status == "Good":
                st.success(f"### Prediction: {status} Credit Score")
            elif status == "Standard":
                st.warning(f"### Prediction: {status} Credit Score")
            else:
                st.error(f"### Prediction: {status} Credit Score")
                
            # Menampilkan diagram probabilitas tambahan agar terlihat profesional
            st.write("Detail Probabilitas Kelas:")
            st.bar_chart({
                "Poor": probs[0],
                "Standard": probs[1],
                "Good": probs[2]
            })

if __name__ == "__main__":
    main()