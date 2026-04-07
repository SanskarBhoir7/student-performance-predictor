"""
Data Preprocessing Module
=========================
Handles loading, cleaning, feature selection, splitting, and scaling
of student performance data.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Configuration ──────────────────────────────────────────────────
FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "distractions",
]
TARGET = "next_score"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Data Loading ───────────────────────────────────────────────────
def load_data(path: str | None = None) -> pd.DataFrame:
    """Load CSV and validate schema."""
    path = path or DATA_PATH
    df = pd.read_csv(path)

    # Validate required columns exist
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ── Data Cleaning ──────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and cap outliers using IQR method."""
    df = df.copy()

    # 1. Handle missing values
    missing_count = df[FEATURES + [TARGET]].isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values — filling with median")
        for col in FEATURES + [TARGET]:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    else:
        print("✅ No missing values")

    # 2. Cap outliers using IQR (1.5x rule)
    outliers_capped = 0
    for col in FEATURES + [TARGET]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        before = len(df[(df[col] < lower) | (df[col] > upper)])
        df[col] = df[col].clip(lower, upper)
        outliers_capped += before

    if outliers_capped > 0:
        print(f"📊 Capped {outliers_capped} outlier values")
    else:
        print("✅ No outliers detected")

    return df


# ── Feature Selection ──────────────────────────────────────────────
def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y."""
    X = df[FEATURES]
    y = df[TARGET]
    print(f"📋 Features: {FEATURES}")
    print(f"🎯 Target: {TARGET}")
    return X, y


# ── Split & Scale ──────────────────────────────────────────────────
def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split into train/test and apply StandardScaler.
    Scaler is fitted ONLY on training data to prevent data leakage.

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )

    print(f"📦 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ── Persistence ────────────────────────────────────────────────────
def save_scaler(scaler: StandardScaler, path: str | None = None) -> None:
    """Save fitted scaler to disk."""
    path = path or os.path.join(MODELS_DIR, "scaler.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"💾 Scaler saved to {path}")


def load_scaler(path: str | None = None) -> StandardScaler:
    """Load fitted scaler from disk."""
    path = path or os.path.join(MODELS_DIR, "scaler.pkl")
    return joblib.load(path)


# ── Full Pipeline ──────────────────────────────────────────────────
def run_pipeline(data_path: str | None = None):
    """Execute complete preprocessing pipeline. Returns all artifacts."""
    print("=" * 50)
    print("🔄 DATA PREPROCESSING PIPELINE")
    print("=" * 50)

    df = load_data(data_path)
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    save_scaler(scaler)

    print("=" * 50)
    print("✅ Preprocessing complete!")
    return X_train, X_test, y_train, y_test, scaler, df


if __name__ == "__main__":
    run_pipeline()
