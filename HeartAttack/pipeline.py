from data_ingestion import ingest_data
from pre_processing import preprocess
from train import train
from evaluation import evaluate

RECALL_THRESHOLD = 0.9

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("Step 2: Preprocessing")
    preprocess()

    print("Step 3: Training")
    run_id = train()

    print("Step 4: Evaluation")
    recall = evaluate(run_id)

    if recall >= RECALL_THRESHOLD:
        print("Model approved for deployment")
    else:
        print("Model rejected")

if __name__ == "__main__":
    run_pipeline()
