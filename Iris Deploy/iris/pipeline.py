from data_ingestion import ingest_data
from pre_processing import preprocess
from train import train
from evaluation import evaluate

ACCURACY_THRESHOLD = 0.9

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("Step 2: Preprocessing")
    train_scaled, test_scaled=preprocess()

    print("Step 3: Training")
    run_id = train(train_scaled)

    print("Step 4: Evaluation")
    accuracy,precision,recall = evaluate(test_scaled,run_id)

    if accuracy >= ACCURACY_THRESHOLD:
        print("Model approved for deployment")
    else:
        print("Model rejected")

if __name__ == "__main__":
    run_pipeline()
