"""
Prediction Module
=================
Loads the trained model and provides prediction, confidence interval,
risk assessment, and actionable insights.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURES, MODELS_DIR


# ── Model Loading ──────────────────────────────────────────────────
_model = None
_scaler = None
_model_info = None


def _ensure_loaded():
    """Lazy-load model and scaler on first use."""
    global _model, _scaler, _model_info

    if _model is None:
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        info_path = os.path.join(MODELS_DIR, "model_comparison.json")

        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)

        if os.path.exists(info_path):
            with open(info_path) as f:
                _model_info = json.load(f)


def get_model_info() -> dict | None:
    """Return model comparison info."""
    _ensure_loaded()
    return _model_info


# ── Prediction ─────────────────────────────────────────────────────
def predict_score(
    study_hours: float,
    attendance: float,
    previous_score: float,
    assignment_score: float,
    sleep_hours: float,
    distractions: float,
) -> float:
    """
    Predict the next exam score for a student.

    Returns: predicted score clipped to [0, 100]
    """
    _ensure_loaded()

    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score, assignment_score, sleep_hours, distractions]],
        columns=FEATURES,
    )
    input_scaled = _scaler.transform(input_data)
    prediction = _model.predict(input_scaled)[0]
    return float(np.clip(prediction, 0, 100))


# ── Confidence Interval ───────────────────────────────────────────
def predict_with_confidence(
    study_hours: float,
    attendance: float,
    previous_score: float,
    assignment_score: float,
    sleep_hours: float,
    distractions: float,
) -> dict:
    """
    Predict score with confidence interval.
    Uses individual tree predictions for ensemble models,
    or a fixed margin for linear models.

    Returns: {"prediction": float, "lower": float, "upper": float, "std": float}
    """
    _ensure_loaded()

    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score, assignment_score, sleep_hours, distractions]],
        columns=FEATURES,
    )
    input_scaled = _scaler.transform(input_data)

    prediction = float(_model.predict(input_scaled)[0])

    # For tree-based ensembles, use individual tree predictions
    if hasattr(_model, "estimators_"):
        tree_predictions = np.array([
            tree.predict(input_scaled)[0]
            if not isinstance(tree, np.ndarray)
            else tree[0].predict(input_scaled)[0]
            for tree in _model.estimators_
        ])
        std = float(np.std(tree_predictions))
    else:
        # For linear models, use a fixed margin based on training RMSE
        info = get_model_info()
        if info and info.get("best_model") in info.get("results", {}):
            std = info["results"][info["best_model"]].get("RMSE", 5.0) * 0.5
        else:
            std = 5.0

    lower = float(np.clip(prediction - 1.96 * std, 0, 100))
    upper = float(np.clip(prediction + 1.96 * std, 0, 100))

    return {
        "prediction": round(prediction, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "std": round(std, 2),
    }


# ── Risk Level ─────────────────────────────────────────────────────
def get_risk_level(predicted_score: float) -> dict:
    """
    Classify student risk based on predicted score.

    Returns: {"level": str, "color": str, "emoji": str, "description": str}
    """
    if predicted_score >= 70:
        return {
            "level": "Low Risk",
            "color": "green",
            "emoji": "🟢",
            "description": "Student is performing well. Maintain current habits.",
        }
    elif predicted_score >= 50:
        return {
            "level": "Medium Risk",
            "color": "orange",
            "emoji": "🟡",
            "description": "Student may need support. Monitor closely and consider interventions.",
        }
    else:
        return {
            "level": "High Risk",
            "color": "red",
            "emoji": "🔴",
            "description": "Student is at risk of poor performance. Immediate intervention recommended.",
        }


# ── Insights Engine ────────────────────────────────────────────────
def get_insights(
    study_hours: float,
    attendance: float,
    previous_score: float,
    assignment_score: float,
    sleep_hours: float,
    distractions: float,
) -> list[dict]:
    """
    Generate actionable insights based on input features.

    Returns: list of {"type": "warning"|"info"|"success", "message": str}
    """
    insights = []

    # Attendance insights
    if attendance < 60:
        insights.append({
            "type": "warning",
            "message": "⚠️ Very low attendance (<60%). This is a major risk factor. "
                       "Attending classes regularly can significantly improve performance.",
        })
    elif attendance < 75:
        insights.append({
            "type": "warning",
            "message": "⚠️ Attendance is below 75%. Increasing attendance is one of the "
                       "strongest predictors of better scores.",
        })
    else:
        insights.append({
            "type": "success",
            "message": "✅ Good attendance! This is a strong foundation for performance.",
        })

    # Study hours insights
    if study_hours < 3:
        insights.append({
            "type": "warning",
            "message": "📚 Study hours are very low (<3 hrs). Even a small increase in "
                       "study time can meaningfully boost scores.",
        })
    elif study_hours < 5:
        insights.append({
            "type": "info",
            "message": "📚 Study hours are moderate. Consider increasing to 5-7 hours "
                       "for optimal results (diminishing returns beyond 8 hrs).",
        })
    else:
        insights.append({
            "type": "success",
            "message": "📚 Good study commitment! Note: returns diminish above 8 hours.",
        })

    # Sleep insights
    if sleep_hours < 5:
        insights.append({
            "type": "warning",
            "message": "😴 Severe sleep deprivation (<5 hrs). This has a quadratic "
                       "negative impact on performance. Aim for 7 hours.",
        })
    elif sleep_hours < 6 or sleep_hours > 8.5:
        insights.append({
            "type": "info",
            "message": "😴 Sleep is not optimal. Research shows 7 hours is the sweet "
                       "spot — both too little and too much sleep hurt performance.",
        })
    else:
        insights.append({
            "type": "success",
            "message": "😴 Sleep hours are in the optimal range (6-8.5 hrs). Great!",
        })

    # Distractions insight
    if distractions > 3.5:
        insights.append({
            "type": "warning",
            "message": "🎮 High distraction level (>3.5). Each unit of distraction "
                       "reduces predicted score by ~3 points. Try to minimize.",
        })
    elif distractions > 2:
        insights.append({
            "type": "info",
            "message": "🎮 Moderate distraction level. Reducing distractions can "
                       "provide a noticeable score boost.",
        })

    # Assignment score insight
    if assignment_score < 50:
        insights.append({
            "type": "warning",
            "message": "📝 Assignment score is low (<50). Improving assignment "
                       "performance directly contributes to exam scores.",
        })

    # Previous score context
    if previous_score > 80:
        insights.append({
            "type": "success",
            "message": "📈 Strong previous performance provides a solid foundation.",
        })
    elif previous_score < 50:
        insights.append({
            "type": "info",
            "message": "📈 Previous score is below average. Focus on building "
                       "fundamentals — other factors can compensate.",
        })

    return insights


# ── What-If Analysis ───────────────────────────────────────────────
def what_if_analysis(
    base_features: dict,
    vary_feature: str,
    steps: int = 50,
) -> pd.DataFrame:
    """
    Vary one feature across its range while holding others constant.

    Returns: DataFrame with columns [vary_feature, "predicted_score"]
    """
    _ensure_loaded()

    # Define realistic ranges for each feature
    feature_ranges = {
        "study_hours": (1, 10),
        "attendance": (50, 100),
        "previous_score": (0, 100),
        "assignment_score": (0, 100),
        "sleep_hours": (4, 9),
        "distractions": (0, 5),
    }

    if vary_feature not in feature_ranges:
        raise ValueError(f"Unknown feature: {vary_feature}")

    low, high = feature_ranges[vary_feature]
    values = np.linspace(low, high, steps)

    predictions = []
    for val in values:
        features = base_features.copy()
        features[vary_feature] = val
        score = predict_score(**features)
        predictions.append(score)

    return pd.DataFrame({vary_feature: values, "predicted_score": predictions})


# ── Goal-Seek Optimization ─────────────────────────────────────────
def recommend_improvements(
    current_features: dict,
    target_score: float,
    max_options: int = 3,
) -> dict:
    """
    Search for the easiest feature modifications to reach a target score.
    Freezes previous_score.
    """
    _ensure_loaded()

    current_pred = predict_score(**current_features)
    if current_pred >= target_score:
        return {"status": "already_reached", "current": current_pred, "options": []}

    import itertools

    study_opts = list(set([current_features["study_hours"], min(10.0, current_features["study_hours"] + 1.5), min(10.0, current_features["study_hours"] + 3)]))
    attn_opts = list(set([current_features["attendance"], min(100.0, current_features["attendance"] + 5), min(100.0, current_features["attendance"] + 15)]))
    
    sleep_opts = [current_features["sleep_hours"]]
    if current_features["sleep_hours"] < 7:
        sleep_opts.extend([min(7.0, current_features["sleep_hours"] + 1), 7.0])
    elif current_features["sleep_hours"] > 8:
        sleep_opts.append(max(7.0, current_features["sleep_hours"] - 1))
    sleep_opts = list(set(sleep_opts))

    dist_opts = list(set([current_features["distractions"], max(0.0, current_features["distractions"] - 1), 0.0]))
    assg_opts = list(set([current_features["assignment_score"], min(100.0, current_features["assignment_score"] + 10), min(100.0, current_features["assignment_score"] + 20)]))

    valid_states = []

    for s, a, sl, d, asg in itertools.product(study_opts, attn_opts, sleep_opts, dist_opts, assg_opts):
        effort = 0.0
        effort += (s - current_features["study_hours"]) * 2.0
        effort += (a - current_features["attendance"]) * 0.5
        effort += abs(sl - current_features["sleep_hours"]) * 1.0
        effort += (current_features["distractions"] - d) * 1.5
        effort += (asg - current_features["assignment_score"]) * 0.2

        if effort <= 0:
            continue

        test_features = {
            "study_hours": s,
            "attendance": a,
            "previous_score": current_features["previous_score"],
            "assignment_score": asg,
            "sleep_hours": sl,
            "distractions": d,
        }

        pred = predict_score(**test_features)

        if pred > current_pred + 0.5:
            valid_states.append({
                "features": test_features,
                "predicted_score": pred,
                "effort": effort,
            })

    if not valid_states:
        return {"status": "impossible", "current": current_pred, "options": []}

    meeting_target = [state for state in valid_states if state["predicted_score"] >= target_score]

    if meeting_target:
        meeting_target.sort(key=lambda x: x["effort"])
        return {
            "status": "success",
            "current": current_pred,
            "options": meeting_target[:max_options],
        }
    else:
        valid_states.sort(key=lambda x: x["predicted_score"], reverse=True)
        return {
            "status": "best_effort",
            "current": current_pred,
            "options": valid_states[:1],
        }


# ── CSP Solver Integration ─────────────────────────────────────────
def csp_recommend(
    current_features: dict,
    target_score: float = 80.0,
    max_solutions: int = 3,
) -> list[dict]:
    """
    Use the CSP solver to find valid input combinations that achieve
    the target predicted score.

    Parameters:
        current_features: current student feature values
        target_score: minimum score to achieve (default 80)
        max_solutions: max number of recommendations to return

    Returns:
        List of dicts with 'features', 'predicted_score', 'effort'
    """
    _ensure_loaded()
    from src.csp_solver import CSPSolver

    solver = CSPSolver(model=_model, scaler=_scaler)
    return solver.solve(current_features, target_score, max_solutions)


if __name__ == "__main__":
    # Quick test
    result = predict_with_confidence(
        study_hours=6, attendance=80, previous_score=70,
        assignment_score=65, sleep_hours=7, distractions=2,
    )
    risk = get_risk_level(result["prediction"])
    insights = get_insights(6, 80, 70, 65, 7, 2)

    print(f"\n🎯 Predicted Score: {result['prediction']}")
    print(f"📊 Confidence: [{result['lower']} – {result['upper']}]")
    print(f"{risk['emoji']} Risk: {risk['level']}")
    print(f"\n💡 Insights:")
    for ins in insights:
        print(f"   {ins['message']}")
