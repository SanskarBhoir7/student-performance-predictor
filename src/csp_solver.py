"""
Constraint Satisfaction Problem (CSP) Solver
=============================================
Uses backtracking search with constraint propagation to find valid
student input combinations that achieve a target predicted score.

Algorithm:
    - Variables: controllable student features (study_hours, attendance,
      sleep_hours, distractions, assignment_score)
    - Domains: discretized realistic value ranges for each variable
    - Constraints: hard limits (attendance ≥ 75, sleep 5-8, etc.)
    - Scoring: the trained ML model evaluates each complete assignment
    - Search: backtracking with MRV heuristic and forward checking

Note: `previous_score` is frozen — students cannot change past scores.
"""

import numpy as np
import pandas as pd
from src.data_preprocessing import FEATURES


# ── Constraint Definitions ─────────────────────────────────────────
VARIABLE_DOMAINS = {
    "study_hours": list(np.arange(1.0, 10.5, 0.5)),       # 1 to 10, step 0.5
    "attendance": list(range(75, 101, 5)),                  # 75 to 100, step 5
    "sleep_hours": list(np.arange(5.0, 8.5, 0.5)),         # 5 to 8, step 0.5
    "distractions": list(np.arange(0.0, 3.5, 0.5)),        # 0 to 3, step 0.5
    "assignment_score": list(range(40, 101, 5)),            # 40 to 100, step 5
}

# Variables the CSP can modify (previous_score is frozen)
CSP_VARIABLES = list(VARIABLE_DOMAINS.keys())


def _unary_constraints_satisfied(variable: str, value: float) -> bool:
    """Check unary constraints for a single variable assignment."""
    if variable == "study_hours":
        return 1.0 <= value <= 10.0
    elif variable == "attendance":
        return value >= 75
    elif variable == "sleep_hours":
        return 5.0 <= value <= 8.0
    elif variable == "distractions":
        return value <= 3.0
    elif variable == "assignment_score":
        return value >= 40
    return True


def _binary_constraints_satisfied(assignment: dict) -> bool:
    """Check constraints involving multiple variables."""
    # Time feasibility: study + sleep must leave room for daily life
    study = assignment.get("study_hours")
    sleep = assignment.get("sleep_hours")
    if study is not None and sleep is not None:
        if study + sleep > 18:  # 24 - 6 hrs for classes/meals/commute
            return False
    return True


# ── CSP Solver ─────────────────────────────────────────────────────
class CSPSolver:
    """
    Backtracking CSP solver that uses a trained ML model as the
    evaluation function to find input combinations achieving a
    target predicted score.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def _predict(self, features: dict) -> float:
        """Score a complete feature assignment using the ML model."""
        input_data = pd.DataFrame(
            [[features[f] for f in FEATURES]],
            columns=FEATURES,
        )
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)[0]
        return float(np.clip(prediction, 0, 100))

    def _select_unassigned_variable(self, assignment: dict, domains: dict) -> str:
        """
        MRV heuristic: pick the unassigned variable with the smallest
        remaining domain (most constrained first).
        """
        unassigned = [v for v in CSP_VARIABLES if v not in assignment]
        return min(unassigned, key=lambda v: len(domains[v]))

    def _forward_check(self, variable: str, value: float,
                       assignment: dict, domains: dict) -> dict | None:
        """
        After assigning variable=value, prune incompatible values
        from remaining domains. Returns pruned domains, or None if
        a domain becomes empty (failure).
        """
        new_domains = {v: list(d) for v, d in domains.items()}
        test_assignment = dict(assignment)
        test_assignment[variable] = value

        for other_var in CSP_VARIABLES:
            if other_var in test_assignment or other_var == variable:
                continue
            pruned = []
            for val in new_domains[other_var]:
                tmp = dict(test_assignment)
                tmp[other_var] = val
                if _binary_constraints_satisfied(tmp):
                    pruned.append(val)
            if not pruned:
                return None  # Domain wipeout → backtrack
            new_domains[other_var] = pruned

        return new_domains

    def _backtrack(self, assignment: dict, domains: dict,
                   fixed_features: dict, target_score: float,
                   solutions: list, max_solutions: int) -> bool:
        """
        Recursive backtracking search.
        Stops after finding max_solutions valid assignments.
        """
        if len(solutions) >= max_solutions:
            return True  # enough solutions found

        # If all variables assigned → evaluate
        if len(assignment) == len(CSP_VARIABLES):
            full_features = dict(fixed_features)
            full_features.update(assignment)
            score = self._predict(full_features)
            if score >= target_score:
                solutions.append({
                    "features": dict(full_features),
                    "predicted_score": round(score, 2),
                })
            return len(solutions) >= max_solutions

        var = self._select_unassigned_variable(assignment, domains)

        for value in domains[var]:
            # Check unary constraints
            if not _unary_constraints_satisfied(var, value):
                continue

            # Check binary constraints with current assignment
            test_assignment = dict(assignment)
            test_assignment[var] = value
            if not _binary_constraints_satisfied(test_assignment):
                continue

            # Forward checking
            new_domains = self._forward_check(var, value, assignment, domains)
            if new_domains is None:
                continue  # Forward check failed → skip

            assignment[var] = value
            if self._backtrack(assignment, new_domains, fixed_features,
                               target_score, solutions, max_solutions):
                del assignment[var]
                return True
            del assignment[var]

        return False

    def solve(self, current_features: dict, target_score: float = 80.0,
              max_solutions: int = 5) -> list[dict]:
        """
        Find valid input combinations that achieve the target score.

        Parameters:
            current_features: current student feature values
            target_score: minimum predicted score to achieve
            max_solutions: maximum number of solutions to return

        Returns:
            List of dicts, each with 'features' and 'predicted_score',
            sorted by effort (minimal changes from current state first).
        """
        # Fixed features that students cannot change
        fixed_features = {
            "previous_score": current_features["previous_score"],
        }

        # Build initial domains
        domains = {}
        for var in CSP_VARIABLES:
            domains[var] = [
                v for v in VARIABLE_DOMAINS[var]
                if _unary_constraints_satisfied(var, v)
            ]

        # Collect more solutions than needed, then rank by effort
        solutions = []
        self._backtrack({}, domains, fixed_features, target_score,
                        solutions, max_solutions=max_solutions * 3)

        # Rank by effort: prefer solutions closest to current state
        for sol in solutions:
            effort = 0.0
            for var in CSP_VARIABLES:
                diff = abs(sol["features"][var] - current_features.get(var, 0))
                effort += diff
            sol["effort"] = round(effort, 2)

        solutions.sort(key=lambda s: s["effort"])
        return solutions[:max_solutions]
