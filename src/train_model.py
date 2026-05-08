from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data_preprocessing import build_preprocessor


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "heart.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "heart_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "final_logistic_regression_model.pkl"


def load_data():
    """
    Load cleaned heart disease dataset.

    If the cleaned dataset does not exist, it will be created
    from the raw dataset by removing duplicated records.
    """
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print("Loaded processed dataset.")

    else:
        df = pd.read_csv(RAW_DATA_PATH)
        df = df.drop_duplicates()

        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)

        print("Processed dataset created and saved.")

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def build_model_pipeline():
    """
    Build final Logistic Regression pipeline.
    """
    preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])

    return pipeline


def train_model(force_retrain=False):
    """
    Train final model only if it does not already exist,
    unless force_retrain is set to True.
    """
    if MODEL_PATH.exists() and not force_retrain:
        print("Saved model already exists.")
        print("Skipping training.")
        return

    X, y = load_data()

    pipeline = build_model_pipeline()
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("Final model trained and saved successfully.")


if __name__ == "__main__":
    train_model()