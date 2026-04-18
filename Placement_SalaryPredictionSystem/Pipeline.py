from Data_Ingestion import ingest_data
from Preprocess import preprocess
from Train_Clas import train_clas
from Train_Reg import train_reg
from Evaluation_Class import evaluate_class
from Evaluation_Reg import evaluate_reg

# Threshold untuk Pipeline
THRESHOLD_MAE = 1.5
THRESHOLD_R2 = 0.7
THRESHOLD_F1 = 0.8  # Tambahkan untuk klasifikasi

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("Step 2: Preprocessing")
    preprocess()

    print("Step 3: Training Models")
    run_id_clas = train_clas()
    run_id_reg = train_reg()

    print("Step 4: Evaluation")
    f1 = evaluate_class(run_id_clas)
    mae, r2 = evaluate_reg(run_id_reg)

    if f1 >= THRESHOLD_F1 and mae <= THRESHOLD_MAE and r2 >= THRESHOLD_R2:
        print("Pipeline Approved: Model siap dideploy!")
    else:
        print("Pipeline Rejected: Model gagal memenuhi kriteria performa.")

if __name__ == "__main__":
    run_pipeline()