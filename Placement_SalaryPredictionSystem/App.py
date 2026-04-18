import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --- KONFIGURASI PATH ---
BASE_DIR = Path(__file__).resolve().parent

def load_file(filename):
    path_artifacts = BASE_DIR / "artifacts" / filename
    path_root = BASE_DIR / filename
    
    if path_artifacts.exists():
        return joblib.load(path_artifacts)
    elif path_root.exists():
        return joblib.load(path_root)
    else:
        st.error(f"File '{filename}' Not Found.")
        st.stop()

artifact = load_file("preprocess_artifact.pkl")
preprocess = artifact["preprocessor"]
feature_names = artifact["feature_names"]

model_clas = load_file("model_clas.pkl")
model_reg = load_file("model_reg.pkl")

def get_input():
    with st.form("input_form"):
        st.subheader("Student Profile Input")

        col1, col2, col3 = st.columns(3)

        with col1:
            cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
            tenth = st.number_input("10th %", 0, 100, 75)
            twelfth = st.number_input("12th %", 0, 100, 75)
            backlogs = st.number_input("Backlogs", 0, 10, 0)
            study_hours = st.number_input("Study Hours", 0.0, 12.0, 4.0)

        with col2:
            attendance = st.number_input("Attendance %", 0, 100, 80)
            projects = st.number_input("Projects", 0, 20, 2)
            internships = st.number_input("Internships", 0, 10, 1)
            coding = st.number_input("Coding Skill", 0, 10, 5)
            comm = st.number_input("Communication", 0, 10, 5)

        with col3:
            aptitude = st.number_input("Aptitude", 0, 10, 5)
            hackathons = st.number_input("Hackathons", 0, 10, 0)
            cert = st.number_input("Certifications", 0, 10, 1)
            sleep = st.number_input("Sleep Hours", 0, 12, 6)
            stress = st.number_input("Stress Level", 0, 10, 5)

        st.divider()
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            market = st.number_input("Marketability", 0, 10, 5)
        with col_m2:
            external = st.number_input("External Score", 0, 10, 5)
        with col_m3:
            potential = st.number_input("Potential", 0, 10, 5)

        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            branch = st.selectbox("Branch", ["CS", "IT", "ECE", "ME"])
            part_time = st.selectbox("Part Time Job", ["No", "Yes"])

        with col_cat2:
            internet = st.selectbox("Internet Access", ["No", "Yes"])
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
            extracurricular = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

        submitted = st.form_submit_button("Predict Now")

    if submitted:
        return pd.DataFrame([{
            "cgpa": cgpa,
            "tenth_percentage": tenth,
            "twelfth_percentage": twelfth,
            "backlogs": backlogs,
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "projects_completed": projects,
            "internships_completed": internships,
            "coding_skill_rating": coding,
            "communication_skill_rating": comm,
            "aptitude_skill_rating": aptitude,
            "hackathons_participated": hackathons,
            "certifications_count": cert,
            "sleep_hours": sleep,
            "stress_level": stress,
            "Marketability": market,
            "External": external,
            "Potential": potential,
            "gender": gender,
            "branch": branch,
            "part_time_job": part_time,
            "internet_access": internet,
            "family_income_level": family_income,
            "extracurricular_involvement": extracurricular,
            "city_tier": city_tier
        }])
    return None

def main():
    st.set_page_config(page_title="Placement Predictor", page_icon="🎓")
    st.title("Placement & Salary Prediction System")
    
    input_df = get_input()

    if input_df is not None:
        # Transformasi Data
        X_transformed = preprocess.transform(input_df)
        X = pd.DataFrame(X_transformed, columns=feature_names)

        # Prediksi Klasifikasi
        placed = model_clas.predict(X)[0]

        st.markdown("---")
        st.subheader("Prediction Results")

        if placed == 0:
            st.error("### Prediction: Not Placed")
            st.write("Saran: Tingkatkan nilai CGPA dan ikuti lebih banyak magang (internship).")
        else:
            st.success("### Prediction: Placed")
            
            # Prediksi Regresi (Hanya jika Placed)
            salary = model_reg.predict(X)[0]
            st.metric(label="Estimated Salary (LPA)", value=f"{salary:.2f}")

if __name__ == "__main__":
    main()
