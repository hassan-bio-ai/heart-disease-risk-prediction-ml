from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


def build_preprocessor():
    """
    Build preprocessing pipeline for numerical and categorical features.

    Numerical features are scaled using RobustScaler.
    Categorical features are encoded using OneHotEncoder.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor