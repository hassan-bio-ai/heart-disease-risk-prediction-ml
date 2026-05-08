from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_logistic_regression_model.pkl"


def load_model():
    """
    Load the saved final model pipeline.
    """
    return joblib.load(MODEL_PATH)


def get_risk_level(probability):
    """
    Convert probability into risk category.
    """
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"


def predict_patient_risk(patient_data):
    """
    Predict heart disease risk for a single patient.
    """
    model = load_model()

    input_df = pd.DataFrame([patient_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    risk_level = get_risk_level(probability)

    return {
        "prediction": int(prediction),
        "risk_probability": round(float(probability), 4),
        "risk_percentage": round(float(probability) * 100, 2),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    sample_patient = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    result = predict_patient_risk(sample_patient)

    print(result)