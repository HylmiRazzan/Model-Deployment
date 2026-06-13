import re
from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer

class CreditPreprocessor:
    """Handles dataset loading, header formatting, and building pre-processing."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _parse_type_of_loan(x):
        """Memecah teks pada kolom pinjaman menjadi daftar yang rapi."""
        if pd.isna(x):
            return []
        x = re.sub(r'\s+and\s+', ', ', str(x))
        return list(set(i.strip() for i in x.split(',') if i.strip()))

    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melakukan pembersihan awal sebelum data masuk ke Scikit-Learn pipeline."""
        
        # 1. Membuang kolom yang tidak diperlukan
        columns_to_drop = ["Unnamed: 0", "ID", "SSN", "Customer_ID", "Name"]
        # Hanya buang kolom jika kolom tersebut memang ada di dataset
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_cols_to_drop, axis=1)

        # 2. Membuang baris duplikat
        df = df.drop_duplicates()

        # 3. Penanganan string kotor menjadi NaN
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
        df['Occupation'] = df['Occupation'].replace('_______', np.nan)

        # 4. Memecah fitur Payment_Behaviour
        df['Spent'] = df['Payment_Behaviour'].str.split('_').str[0].str.lower()
        df['Value_Payment'] = df['Payment_Behaviour'].str.split('_').str[2].str.lower()
        df = df.drop(['Payment_Behaviour'], axis=1)

        # 5. Penanganan khusus Type_of_Loan
        df['Type_of_Loan_list'] = df['Type_of_Loan'].apply(self._parse_type_of_loan)
        df['loan_missing'] = (df['Type_of_Loan'].isna()).astype(int)

        mlb = MultiLabelBinarizer()
        loan_ohe = pd.DataFrame(
            mlb.fit_transform(df['Type_of_Loan_list']),
            columns=mlb.classes_,
            index=df.index
        )
        df = pd.concat([df, loan_ohe], axis=1)
        df.drop(columns=['Type_of_Loan', 'Type_of_Loan_list'], inplace=True)

        # 6. Pembersihan Umur Riwayat Kredit
        years = df['Credit_History_Age'].str.extract(r'(\d+) Years', expand=False).astype(float)
        months = df['Credit_History_Age'].str.extract(r'(\d+) Months', expand=False).astype(float)
        df['Credit_History_Age'] = years + (months / 12)

        # 7. Pembersihan karakter "_" dan konversi ke angka yang aman
        df["Age"] = pd.to_numeric(df["Age"].astype(str).str.strip("_"), errors='coerce')
        df["Annual_Income"] = pd.to_numeric(df["Annual_Income"].astype(str).str.strip("_"), errors='coerce')
        df["Num_of_Loan"] = pd.to_numeric(df["Num_of_Loan"].astype(str).str.strip("_"), errors='coerce')
        df["Num_of_Delayed_Payment"] = pd.to_numeric(df["Num_of_Delayed_Payment"].astype(str).str.strip("_"), errors='coerce')
        df["Changed_Credit_Limit"] = pd.to_numeric(df["Changed_Credit_Limit"].astype(str).str.strip("_"), errors='coerce')
        df["Outstanding_Debt"] = pd.to_numeric(df["Outstanding_Debt"].astype(str).str.strip("_"), errors='coerce')
        df["Amount_invested_monthly"] = pd.to_numeric(df["Amount_invested_monthly"].astype(str).str.strip("_"), errors='coerce')

        # 8. Penanganan khusus Monthly_Balance
        df = df[df["Monthly_Balance"] != "__-333333333333333333333333333__"]
        df["Monthly_Balance"] = pd.to_numeric(df["Monthly_Balance"].astype(str).str.strip("_"), errors='coerce')

        # 9. Pemfilteran batas wajar (Outlier Bounds)
        bounds = {
            'Age': (18, 100),
            'Annual_Income': (7000, df['Annual_Income'].quantile(0.99)),
            'Monthly_Inhand_Salary': (1, df['Monthly_Inhand_Salary'].quantile(0.99)),
            'Num_Bank_Accounts': (0, 10),
            'Num_Credit_Card': (0, 10),
            'Num_of_Loan': (0, 10),
            'Delay_from_due_date': (0, 61),
            'Num_of_Delayed_Payment': (0, 27),
            'Changed_Credit_Limit': (0, 30),
            'Num_Credit_Inquiries': (0, 15),
            'Outstanding_Debt': (1, df['Outstanding_Debt'].quantile(0.99)),
            'Credit_Utilization_Ratio': (0, 100),
            'Total_EMI_per_month': (0, df['Total_EMI_per_month'].quantile(0.99)),
            'Amount_invested_monthly': (0, 10000),
            'Monthly_Balance': (1, df['Monthly_Balance'].quantile(0.99))
        }

        for col, (low, high) in bounds.items():
            if col in df.columns:
                df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

        return df

    def clean_and_split(self, data_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads data, standardizes schema names, and partitions records into train/test sets."""
        df = pd.read_csv(Path(data_path), sep=",")

        # Panggil fungsi pembersihan yang baru dibuat
        df = self._clean_raw_data(df)

        credit_score_mapping = {
            'Poor': 0,
            'Standard': 1,
            'Good': 2
        }

        x = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score'].map(credit_score_mapping)

        return train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def get_transformer(self, x_train: pd.DataFrame) -> ColumnTransformer:
        """Assembles internal transformers required to cleanly handle numerical and text features."""
        
        # Daftar outlier (loan_missing sudah tersedia karena fungsi _clean_raw_data)
        num_outlier = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
        no_outlier = ['Credit_Utilization_Ratio', 'Credit_History_Age']

        nominal_features = ['Month', 'Occupation'] 

        ordinal_features = ['Credit_Mix', 'Payment_of_Min_Amount', 'Spent', 'Value_Payment']

        biner_features = ['loan_missing', 'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']

        goodbad_order = ['Bad', 'Standard', 'Good']
        yesno_order = ['No', 'Yes']
        lowhigh_order = ['low', 'high']
        smalllarge_order = ['small', 'medium', 'large']
        
        outlier_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        no_outlier_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        biner_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        nominal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Ordinal Pipeline dengan pengaman jika ada teks yang tidak terduga
        ordinal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                categories=[
                    goodbad_order,
                    yesno_order,
                    lowhigh_order,
                    smalllarge_order
                ],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])
        
        # Dinamis: Ambil semua nama kolom yang dihasilkan oleh MultiLabelBinarizer
        # Ini penting karena jumlah jenis pinjaman bisa berbeda
        all_train_cols = list(x_train.columns)
        known_cols = set(num_outlier + no_outlier + biner_features + nominal_features + ordinal_features)
        loan_ohe_features = [col for col in all_train_cols if col not in known_cols]

        ohe_pass_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        return ColumnTransformer(
                transformers=[
                    ('num_outlier', outlier_pipeline, num_outlier),
                    ('num_no_outlier', no_outlier_pipeline, no_outlier),
                    ('biner', biner_pipeline, biner_features),
                    ('nom', nominal_pipeline, nominal_features),
                    ('ord', ordinal_pipeline, ordinal_features),
                    ('loan_types', ohe_pass_pipeline, loan_ohe_features)
                ],
                remainder='drop'
            )