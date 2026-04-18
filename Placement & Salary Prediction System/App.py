import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

artifact = joblib.load(os.path.join(BASE_DIR, "preprocess_artifact.pkl"))
preprocess = artifact["preprocessor"]
feature_names = artifact["feature_names"]

model_clas = joblib.load(os.path.join(BASE_DIR, "model_clas.pkl"))
model_reg = joblib.load(os.path.join(BASE_DIR, "model_reg.pkl"))

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

        market = st.number_input("Marketability", 0, 10, 5)
        external = st.number_input("External Score", 0, 10, 5)
        potential = st.number_input("Potential", 0, 10, 5)

        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CS", "IT", "ECE", "ME"])

        part_time = st.selectbox("Part Time Job", ["No", "Yes"])
        internet = st.selectbox("Internet Access", ["No", "Yes"])

        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        extracurricular = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

        submitted = st.form_submit_button("🚀 Predict")

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

    st.title("Placement & Salary Prediction System")
    st.write("Clean ML Pipeline: Classification & Regression")

    input_df = get_input()

    if input_df is not None:

        X = preprocess.transform(input_df)
        X = pd.DataFrame(X, columns=feature_names)


        placed = model_clas.predict(X)[0]

        st.subheader("Result")

        if placed == 0:
            st.error("Not Placed")
        else:
            st.success("Placed")

            salary = model_reg.predict(X)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Salary (LPA)", f"{salary:.2f}")

            with col2:
                st.info("Benchmark: 5 LPA")

            st.bar_chart({"Salary": [salary], "Benchmark": [5]})


if __name__ == "__main__":
    main()