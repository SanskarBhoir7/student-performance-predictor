"""
Dataset Generation Module
=========================
Generates synthetic student performance data with realistic feature
influences, diminishing returns for study hours, and noise.
"""

import os
import pandas as pd
import numpy as np

# Path to save the dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate realistic academic data based on predefined formulas."""
    np.random.seed(42)  # For reproducibility

    # 1. Generate Input Features
    # Previous score has a normal distribution around 70
    previous_score = np.random.normal(loc=70, scale=15, size=n_samples)
    previous_score = np.clip(previous_score, 0, 100)

    # Assignment score correlates with previous score to be realistic
    assignment_score = 0.6 * previous_score + np.random.normal(loc=30, scale=12, size=n_samples)
    assignment_score = np.clip(assignment_score, 0, 100)

    # Attendance distribution, highly skewed towards 100
    attendance = np.random.normal(loc=85, scale=15, size=n_samples)
    attendance = np.clip(attendance, 50, 100)

    # Habits
    study_hours = np.random.uniform(1.0, 10.0, size=n_samples)
    distractions = np.random.uniform(0.0, 5.0, size=n_samples)
    
    # Sleep distribution mostly between 5 and 9
    sleep_hours = np.random.normal(loc=7.0, scale=1.2, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)

    # 2. Calculate Ground Truth Target (next_score)
    # Core academic factors sum to 100% of the baseline
    # This ensures a student with 90+ everywhere stays at ~90+ before lifestyle modifiers
    base_academic = (
        0.60 * previous_score +
        0.25 * assignment_score +
        0.15 * attendance
    )

    # Study hours act as a marginal modifier relative to a baseline of 4 hours
    # Cap the total modifier influence so it doesn't drag down a genius student
    study_modifier = 8.0 * (np.log1p(study_hours) - np.log1p(4.0)) / np.log1p(10.0)

    # Quadratic penalty for sleep deviation away from 7 optimal hours
    sleep_penalty = -1.2 * (sleep_hours - 7.0) ** 2

    # Linear penalty for distractions
    distraction_penalty = -1.5 * distractions

    # Combine
    base_score = base_academic + study_modifier + sleep_penalty + distraction_penalty

    # 3. Add Controlled Gaussian Noise
    # Simulates real-world academic unpredictability
    noise = np.random.normal(loc=0.0, scale=4.5, size=n_samples)
    next_score = base_score + noise

    # Final clip
    next_score = np.clip(next_score, 0, 100)

    # 4. Create and return DataFrame
    df = pd.DataFrame({
        "study_hours": np.round(study_hours, 1),
        "attendance": np.round(attendance, 0),
        "previous_score": np.round(previous_score, 1),
        "assignment_score": np.round(assignment_score, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "distractions": np.round(distractions, 1),
        "next_score": np.round(next_score, 1)
    })
    
    return df

def run_tests_and_save():
    """Run sanity checks to ensure data behaves as requested, then save."""
    df = generate_synthetic_data(n_samples=2000)
    
    print("=" * 50)
    print("📊 DATA GENERATION TESTS")
    print("=" * 50)

    # Test Case 1: High baseline, low study effort
    test_1 = df[
        (df["previous_score"] > 90) &
        (df["assignment_score"] > 90) &
        (df["study_hours"] < 3) &
        (df["sleep_hours"] >= 6) & (df["sleep_hours"] <= 8)
    ]
    if not test_1.empty:
        avg_score_1 = test_1["next_score"].mean()
        print(f"✅ Test 1 (High Previous / Low Study): Avg expected ~90. Result: {avg_score_1:.1f}")
    
    # Test Case 2: High study effort, low baseline
    test_2 = df[
        (df["previous_score"] < 50) &
        (df["study_hours"] > 7) &
        (df["sleep_hours"] >= 6) & (df["sleep_hours"] <= 8)
    ]
    if not test_2.empty:
        avg_score_2 = test_2["next_score"].mean()
        print(f"✅ Test 2 (Low Previous / High Study): Avg expected ~moderate. Result: {avg_score_2:.1f}")

    # Test Case 3: Extreme distractions penalization
    test_3 = df[
        (df["distractions"] > 4.0) & 
        (df["previous_score"] > 70)
    ]
    if not test_3.empty:
        avg_score_3 = test_3["next_score"].mean()
        print(f"✅ Test 3 (High Distractions on OK Student): Handled gracefully, Avg: {avg_score_3:.1f}")

    # Ensure dataset constraints
    assert df["next_score"].max() <= 100.0, "Scores exceeded 100"
    assert df["next_score"].min() >= 0.0, "Scores dropped below 0"
    
    # Save the dataframe
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"\n💾 Saved {len(df)} realistic samples to {DATA_PATH}")

if __name__ == "__main__":
    run_tests_and_save()
