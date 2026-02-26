from pathlib import Path
import pandas as pd

# Base directory = iris/
BASE_DIR = Path(__file__).parent

# Define folders
RAW_DIR = BASE_DIR #/folder
INGESTED_DIR = BASE_DIR /"ingested"

# Define files
INPUT_FILE = RAW_DIR / "IRIS.csv"
OUTPUT_FILE = INGESTED_DIR / "IRIS.csv"

def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data
    df = pd.read_csv(INPUT_FILE)

    # Basic validation
    assert not df.empty, "Dataset is empty"

    # Save ingested data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Data ingested from {INPUT_FILE} → {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()
