"""
Model Training Module
=====================
Trains multiple models, compares performance, and saves the best one.
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import run_pipeline, MODELS_DIR


# ── Model Definitions ─────────────────────────────────────────────
def get_models() -> dict:
    """Return dictionary of model name → model instance."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_split=10, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }


# ── Evaluation ─────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """Compute MAE, RMSE, R² for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
    }


# ── Training Pipeline ─────────────────────────────────────────────
def train_and_compare():
    """Train all models, compare, save the best."""

    # Step 1: Preprocess data
    X_train, X_test, y_train, y_test, scaler, df = run_pipeline()

    # Step 2: Train & evaluate each model
    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING & COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 55)

    best_model = None
    best_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        marker = ""
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = model
            best_name = name
            marker = " ⭐"

        print(
            f"{name:<25} {metrics['MAE']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}{marker}"
        )

    # Step 3: Print winner
    print("-" * 55)
    print(f"\n🏆 Best Model: {best_name} (R² = {best_r2:.4f})")

    # Step 4: Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved to {model_path}")

    # Step 5: Save comparison results
    comparison_path = os.path.join(MODELS_DIR, "model_comparison.json")
    comparison_data = {
        "best_model": best_name,
        "results": results,
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"📊 Comparison saved to {comparison_path}")

    # Step 6: Feature importance (if tree-based)
    if hasattr(best_model, "feature_importances_"):
        from src.data_preprocessing import FEATURES

        importances = best_model.feature_importances_
        print("\n📈 Feature Importance:")
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx:
            bar = "█" * int(importances[i] * 40)
            print(f"   {FEATURES[i]:<20} {importances[i]:.4f} {bar}")

    print("\n✅ Training pipeline complete!")
    return best_model, best_name, results


if __name__ == "__main__":
    train_and_compare()
