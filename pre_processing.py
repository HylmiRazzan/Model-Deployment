import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess():
    df = pd.read_csv("ingested/Heart Attack Data Set.csv")

    x = df.drop(["target"], axis = 1)
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

if __name__ == "__main__":
    preprocess()