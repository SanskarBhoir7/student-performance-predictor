"""
Rule-Based Reasoning Engine (Knowledge-Based System)
=====================================================
A structured, interpretable IF-THEN rule system for student performance
assessment. Completely independent from ML logic — this module uses
explicit domain knowledge encoded as rules.

Each rule has:
    - condition: callable that evaluates input features
    - category: risk | study | lifestyle | academic
    - priority: high | medium | low
    - message: human-readable recommendation
    - rule_id: unique identifier for traceability

This is a classical AI component (knowledge-based system) that
complements the ML predictions with interpretable reasoning.
"""


# ── Rule Definitions ───────────────────────────────────────────────
RULES = [
    # ── Risk Rules ─────────────────────────────────────────────────
    {
        "rule_id": "R01",
        "category": "risk",
        "priority": "high",
        "condition": lambda f: f.get("predicted_score", 100) < 50,
        "message": "🔴 CRITICAL: Predicted score is below 50. Immediate academic "
                   "intervention is strongly recommended.",
    },
    {
        "rule_id": "R02",
        "category": "risk",
        "priority": "medium",
        "condition": lambda f: 50 <= f.get("predicted_score", 100) < 65,
        "message": "🟡 WARNING: Predicted score is in the marginal zone (50–65). "
                   "Proactive measures are advised to avoid falling into risk.",
    },
    {
        "rule_id": "R03",
        "category": "risk",
        "priority": "high",
        "condition": lambda f: f["attendance"] < 75 and f["study_hours"] < 3,
        "message": "🔴 COMPOUND RISK: Both attendance (<75%) and study hours (<3 hrs) "
                   "are critically low. This combination strongly predicts poor outcomes.",
    },

    # ── Attendance Rules ───────────────────────────────────────────
    {
        "rule_id": "A01",
        "category": "academic",
        "priority": "high",
        "condition": lambda f: f["attendance"] < 60,
        "message": "📋 Attendance is dangerously low (<60%). Regular class attendance "
                   "is one of the strongest predictors of academic success.",
    },
    {
        "rule_id": "A02",
        "category": "academic",
        "priority": "medium",
        "condition": lambda f: 60 <= f["attendance"] < 75,
        "message": "📋 Attendance is below the recommended 75% threshold. Aim to "
                   "attend at least 75% of classes for measurable improvement.",
    },
    {
        "rule_id": "A03",
        "category": "academic",
        "priority": "low",
        "condition": lambda f: f["attendance"] >= 90,
        "message": "✅ Excellent attendance (≥90%). This is a strong foundation "
                   "for academic performance.",
    },

    # ── Study Hours Rules ──────────────────────────────────────────
    {
        "rule_id": "S01",
        "category": "study",
        "priority": "high",
        "condition": lambda f: f["study_hours"] < 2,
        "message": "📚 Study hours are critically low (<2 hrs/day). Increasing "
                   "study time is the highest-impact change available.",
    },
    {
        "rule_id": "S02",
        "category": "study",
        "priority": "medium",
        "condition": lambda f: 2 <= f["study_hours"] < 4,
        "message": "📚 Study hours are below average (2–4 hrs). Increasing to "
                   "5–7 hours daily can significantly improve outcomes.",
    },
    {
        "rule_id": "S03",
        "category": "study",
        "priority": "low",
        "condition": lambda f: f["study_hours"] > 9,
        "message": "📚 Note: Studying >9 hours shows diminishing returns and "
                   "may indicate inefficient study methods. Quality > quantity.",
    },

    # ── Sleep Rules ────────────────────────────────────────────────
    {
        "rule_id": "L01",
        "category": "lifestyle",
        "priority": "high",
        "condition": lambda f: f["sleep_hours"] < 5,
        "message": "😴 Severe sleep deprivation (<5 hrs). Sleep has a quadratic "
                   "effect on cognition — aim for 7 hours for optimal performance.",
    },
    {
        "rule_id": "L02",
        "category": "lifestyle",
        "priority": "medium",
        "condition": lambda f: f["sleep_hours"] > 8.5,
        "message": "😴 Excessive sleep (>8.5 hrs) can be counterproductive. "
                   "The optimal range is 6.5–8 hours.",
    },
    {
        "rule_id": "L03",
        "category": "lifestyle",
        "priority": "low",
        "condition": lambda f: 6.5 <= f["sleep_hours"] <= 8,
        "message": "✅ Sleep hours are in the optimal range. Good sleep hygiene "
                   "supports memory consolidation and focus.",
    },

    # ── Distraction Rules ──────────────────────────────────────────
    {
        "rule_id": "L04",
        "category": "lifestyle",
        "priority": "high",
        "condition": lambda f: f["distractions"] > 4,
        "message": "🎮 Distraction level is very high (>4). Each unit of distraction "
                   "reduces predicted score by approximately 3 points.",
    },
    {
        "rule_id": "L05",
        "category": "lifestyle",
        "priority": "medium",
        "condition": lambda f: 2.5 < f["distractions"] <= 4,
        "message": "🎮 Moderate-to-high distractions. Consider reducing screen time "
                   "and creating a focused study environment.",
    },

    # ── Assignment Rules ───────────────────────────────────────────
    {
        "rule_id": "A04",
        "category": "academic",
        "priority": "high",
        "condition": lambda f: f["assignment_score"] < 40,
        "message": "📝 Assignment score is critically low (<40). Assignments are "
                   "both a direct grade component and practice for exams.",
    },
    {
        "rule_id": "A05",
        "category": "academic",
        "priority": "medium",
        "condition": lambda f: 40 <= f["assignment_score"] < 60,
        "message": "📝 Assignment score is below average. Dedicating more effort "
                   "to assignments can directly improve exam readiness.",
    },

    # ── Positive Compound Rules ────────────────────────────────────
    {
        "rule_id": "P01",
        "category": "academic",
        "priority": "low",
        "condition": lambda f: (f["attendance"] >= 85 and f["study_hours"] >= 5
                                and f["assignment_score"] >= 70),
        "message": "🌟 Strong academic profile: good attendance, study hours, and "
                   "assignment scores. Maintain these habits for continued success.",
    },
    {
        "rule_id": "P02",
        "category": "lifestyle",
        "priority": "low",
        "condition": lambda f: (6 <= f["sleep_hours"] <= 8
                                and f["distractions"] <= 1.5),
        "message": "🌟 Excellent lifestyle balance: healthy sleep and low "
                   "distractions provide an ideal foundation for learning.",
    },
]


# Priority ordering for sorting
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ── Rule Engine ────────────────────────────────────────────────────
def evaluate_rules(features: dict) -> list[dict]:
    """
    Evaluate all rules against the given student features.

    Parameters:
        features: dict with keys including study_hours, attendance,
                  sleep_hours, distractions, assignment_score.
                  Optionally includes 'predicted_score' from the ML model.

    Returns:
        List of triggered rules sorted by priority (high → low),
        each containing: rule_id, category, priority, message
    """
    triggered = []

    for rule in RULES:
        try:
            if rule["condition"](features):
                triggered.append({
                    "rule_id": rule["rule_id"],
                    "category": rule["category"],
                    "priority": rule["priority"],
                    "message": rule["message"],
                })
        except (KeyError, TypeError):
            # Skip rules that reference missing features
            continue

    # Sort by priority: high first, then medium, then low
    triggered.sort(key=lambda r: _PRIORITY_ORDER.get(r["priority"], 99))
    return triggered


def get_category_summary(triggered_rules: list[dict]) -> dict[str, list[dict]]:
    """
    Group triggered rules by category for structured display.

    Returns:
        Dict mapping category name → list of triggered rules
    """
    categories = {}
    for rule in triggered_rules:
        cat = rule["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(rule)
    return categories


# ── Pretty-print for CLI ──────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with sample features
    sample = {
        "study_hours": 2.0,
        "attendance": 60,
        "previous_score": 45,
        "assignment_score": 50,
        "sleep_hours": 4.5,
        "distractions": 4.0,
        "predicted_score": 42.0,
    }

    print("=" * 60)
    print("📋 RULE-BASED ASSESSMENT")
    print("=" * 60)

    results = evaluate_rules(sample)
    categories = get_category_summary(results)

    for cat_name, rules in categories.items():
        print(f"\n── {cat_name.upper()} ──")
        for r in rules:
            priority_tag = f"[{r['priority'].upper()}]"
            print(f"  {r['rule_id']} {priority_tag:>8}  {r['message']}")

    print(f"\n📊 Total rules triggered: {len(results)}")
