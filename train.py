import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Config (matches your notebook)
# ----------------------------
DATA_PATH = "data/heart_disease_clean.csv"   # your notebook reads this
MODEL_PATH = "model/model.joblib"

SELECTED_FEATURES = ['cp_4', 'ca', 'thal_7.0', 'exang', 'oldpeak', 'thalach', 'sex', 'age']


def detect_outliers_iqr_bounds(df: pd.DataFrame, col: str):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def cap_outliers(df: pd.DataFrame, columns):
    df = df.copy()
    for col in columns:
        lower, upper = detect_outliers_iqr_bounds(df, col)
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df


def build_clean_df_from_ucimlrepo(save_csv: bool = True) -> pd.DataFrame:
    """
    Reproduces the cleaning + encoding steps in your notebook for UCI id=45.
    Creates a dataframe with a binary 'target' column.
    """
    from ucimlrepo import fetch_ucirepo

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    # Replace '?' with NaN
    X = X.replace('?', np.nan)

    # Cast columns (same lists as your notebook)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    X[num_cols] = X[num_cols].apply(pd.to_numeric)
    X[cat_cols] = X[cat_cols].apply(pd.to_numeric)

    # Impute (same as notebook)
    X['ca'] = X['ca'].fillna(X['ca'].median())
    X['thal'] = X['thal'].fillna(X['thal'].mode()[0])

    # Cap outliers (same as notebook)
    outlier_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    X = cap_outliers(X, outlier_cols)

    # Binary target (same as notebook)
    y_bin = (y['num'] > 0).astype(int)

    # One-hot (same as notebook)
    X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

    df = X.copy()
    df['target'] = y_bin.values

    if save_csv:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved cleaned dataset to: {DATA_PATH}")

    return df


def load_training_df() -> pd.DataFrame:
    # Prefer the exact same CSV your notebook uses
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df

    # If missing, rebuild it from UCI repo using notebook steps
    print(f"Did not find {DATA_PATH}. Rebuilding from UCI (id=45) using notebook cleaning steps...")
    return build_clean_df_from_ucimlrepo(save_csv=True)


def main():
    df = load_training_df()

    # Ensure required columns exist
    missing = [c for c in SELECTED_FEATURES + ['target'] if c not in df.columns]
    if missing:
        raise ValueError(
            "Your training data is missing required columns:\n"
            f"{missing}\n\n"
            "This usually means your cleaning/encoding differs from the notebook, "
            "or your CSV is not the same format."
        )

    X = df[SELECTED_FEATURES].copy()
    y = df['target'].copy()

    # Split (same params as notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scaler + SVC (matches notebook behavior; scaling fits on train only)
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVC Accuracy: {acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save artifact (pipeline + feature list)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "selected_features": SELECTED_FEATURES
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model artifact to: {MODEL_PATH}")


if __name__ == "__main__":
    main()