import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def preprocess():

    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("A_merged.csv")
    df['Marketability'] = ((df['cgpa'] * 0.4)+(df['coding_skill_rating'] *0.4) + (df['internships_completed'] * 0.3) 
                       + (df['projects_completed'] * 0.2))
    df['External'] = (df['hackathons_participated'] * 0.8) + (df['certifications_count'] * 0.2)
    df['Potential'] = df['aptitude_skill_rating'] / (df['backlogs']+1)

    y_class = df["placement_status"].map({
        "Not Placed": 0,
        "Placed": 1
    })

    y_reg = df["salary_lpa"]

    x = df.drop(["salary_lpa", "Student_ID", "placement_status"], axis=1)

    # (CLASSIFICATION BASE)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    num_cols = [
        'cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
        'study_hours_per_day', 'attendance_percentage', 'projects_completed',
        'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
        'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
        'sleep_hours', 'stress_level', 'Marketability', 'External', 'Potential'
    ]

    ohe_cols = ['gender', 'branch']

    yesno_cols = ['part_time_job', 'internet_access']

    lowmedhigh_cols = ['family_income_level', 'extracurricular_involvement']

    tier_cols = ['city_tier']

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    ohe_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    yesno_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[['No', 'Yes'], ['No', 'Yes']]))
    ])

    lowmedhigh_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[['Low', 'Medium', 'High'], ['Low', 'Medium', 'High']]))
    ])

    tier_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[['Tier 1', 'Tier 2', 'Tier 3']]))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("ohe", ohe_transformer, ohe_cols),
        ("yesno", yesno_transformer, yesno_cols),
        ("lowmedhigh", lowmedhigh_transformer, lowmedhigh_cols),
        ("tier", tier_transformer, tier_cols)
    ])

    x_train_p = preprocessor.fit_transform(x_train) #hasilnya array abis di preprocess(encoding)
    x_test_p = preprocessor.transform(x_test)

    ohe_names = preprocessor.named_transformers_["ohe"] \
        .named_steps["encoder"] \
        .get_feature_names_out(ohe_cols)

    feature_names = (
        num_cols +
        list(ohe_names) +
        yesno_cols +
        lowmedhigh_cols +
        tier_cols
    )

    #CONVERT TO DATAFRAME
    x_train_df = pd.DataFrame(x_train_p, columns=feature_names, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_p, columns=feature_names, index=x_test.index)

    #REGRESSION DATA (FILTER)
    reg_train = y_train == 1

    x_train_reg = x_train_df[reg_train]
    y_train_reg = y_reg.loc[y_train.index][reg_train]

    artifact = {
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }

    #TRAIN CLASSIFICATION
    train_class_df = x_train_df.copy()
    train_class_df['placement_status'] = y_train.values
    train_class_df.to_csv("artifacts/train_classification.csv", index=False)

    #TRAIN CLASSIFICATION
    test_class_df = x_test_df.copy()
    test_class_df['placement_status'] = y_test.values
    test_class_df.to_csv("artifacts/test_classification.csv", index=False)

    #TRAIN REGRESSION
    reg_train_mask = y_train == 1
    x_train_reg = x_train_df[reg_train_mask.values]
    y_train_reg = y_reg.loc[y_train.index][reg_train_mask.values]

    train_reg_df = x_train_reg.copy()
    train_reg_df['salary_lpa'] = y_train_reg.values
    train_reg_df.to_csv("artifacts/train_regression.csv", index=False)

    #TEST REGRESSION
    reg_test_mask = y_test == 1
    x_test_reg = x_test_df[reg_test_mask.values]
    y_test_reg = y_reg.loc[y_test.index][reg_test_mask.values]

    test_reg_df = x_test_reg.copy()
    test_reg_df['salary_lpa'] = y_test_reg.values
    test_reg_df.to_csv("artifacts/test_regression.csv", index=False)
    joblib.dump(artifact, "artifacts/preprocess_artifact.pkl")


    return (
        x_train_df,
        x_test_df,
        y_train,
        y_test,
        x_train_reg,
        y_train_reg
    )


if __name__ == "__main__":
    preprocess()