# 🎓 Student Performance Predictor

An end-to-end **Hybrid AI system** that predicts student exam performance by combining Machine Learning, Constraint Satisfaction Problem (CSP) solving, Rule-Based Reasoning, and Decision Tree analysis.

This project moves beyond a simple prediction by functioning as a **Decision Support System**. It provides confidence intervals, risk assessment, actionable insights, what-if analysis, CSP-based goal optimization, and interpretable rule-based assessment.

---

## 🏗️ System Architecture

The project is structured as a complete pipeline integrating multiple AI paradigms:

```text
Data → Preprocessing → Model Training & Evaluation → Prediction → Hybrid AI Layer → UI
                                                          │
                                          ┌───────────────┼───────────────┐
                                          │               │               │
                                    ML Prediction    CSP Solver    Rule-Based Engine
                                    (5 Models)    (Backtracking)   (IF-THEN Rules)
```

- **Data Preprocessing**: Handled in `src/data_preprocessing.py`. Implements outlier capping (IQR), median imputation, and explicit feature selection.
- **Model Training**: Handled in `src/train_model.py`. Evaluates multiple algorithms and saves the best model.
- **Prediction Module**: Handled in `src/predict.py`. Loads the trained artifact and provides contextual metrics (confidence, risk, insights).
- **CSP Solver**: Handled in `src/csp_solver.py`. Uses backtracking search with the ML model as a scoring function.
- **Rule-Based Engine**: Handled in `src/rule_engine.py`. Expert-defined IF-THEN rules for interpretable assessment.
- **Web Interface**: Built with Streamlit (`app/app.py`).

### 📂 Directory Structure

```text
student-performance-predictor/
│
├── data/
│   └── student_data.csv          # Raw data file
│
├── models/
│   ├── model.pkl                 # Best performing ML model (generated)
│   ├── scaler.pkl                # Fitted standard scaler (generated)
│   └── model_comparison.json     # Training metrics for all 5 models (generated)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Clean, scale, and split data
│   ├── train_model.py            # Train & evaluate models (5 algorithms)
│   ├── predict.py                # Inference, business logic, and CSP integration
│   ├── csp_solver.py             # Constraint Satisfaction Problem solver
│   └── rule_engine.py            # Rule-Based Reasoning engine
│
├── app/
│   └── app.py                    # Streamlit Web UI (Hybrid AI dashboard)
│
├── notebook/
│   └── analysis.ipynb            # Jupyter notebook for thorough EDA
│
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

---

## ✨ Key Features

### 1. Robust Predictive Modeling
We evaluate five algorithms:
- **Linear Regression**: Baseline model.
- **Ridge Regression**: Regularized baseline — often the best performer on this dataset.
- **Decision Tree**: Interpretable, rule-like splits that align with academic AI syllabus concepts (see below).
- **Random Forest**: Captures non-linear relationships via ensemble of decision trees.
- **Gradient Boosting**: Sequential ensemble method optimized for tabular data.

*The pipeline automatically compares MAE, RMSE, and R² scores, and promotes the best model to production.*

### 2. Decision Support Capabilities
- **Confidence Intervals**: Rather than a single number, the system provides a range (e.g., 78.3 ± 5) using individual tree variance or RMSE bounds.
- **Risk Level Assessment**: Automatically flags students as Low, Medium, or High risk.
- **Contextual Insights**: Generating targeted warnings and tips based on the exact combination of user inputs (e.g., "Diminishing returns on study hours beyond this point").
- **What-If Analysis**: An interactive Plotly chart allowing users to see the marginal impact of altering a single variable (e.g., "What if I sleep 2 hours more?").

---

## 🧠 Classical AI Integration

This project integrates classical AI algorithms from the academic syllabus alongside modern ML, making it a **hybrid AI system**.

### Constraint Satisfaction Problem (CSP)

**Module**: `src/csp_solver.py`

The CSP solver answers the question: *"What input combinations will help me achieve a target score?"*

- **Variables**: `study_hours`, `attendance`, `sleep_hours`, `distractions`, `assignment_score` (5 controllable variables; `previous_score` is frozen since students can't change past performance)
- **Domains**: Discretized realistic ranges (e.g., study_hours ∈ [1, 10], attendance ∈ [75, 100])
- **Constraints**:
  - `study_hours` ∈ [1, 10]
  - `attendance` ≥ 75%
  - `sleep_hours` ∈ [5, 8]
  - `distractions` ≤ 3
  - `study_hours + sleep_hours` ≤ 18 (time feasibility)
- **Algorithm**: Backtracking search with **MRV heuristic** (Minimum Remaining Values) and **forward checking** for constraint propagation
- **Scoring**: The trained ML model (`model.pkl`) serves as the evaluation function — a complete variable assignment is accepted only if the predicted score ≥ target
- Solutions are ranked by **effort** (minimal changes from the student's current state)

### Rule-Based Reasoning (Knowledge-Based System)

**Module**: `src/rule_engine.py`

A structured IF-THEN rule system providing interpretable, expert-defined academic assessment:

| Rule ID | Category | Condition | Recommendation |
|---------|----------|-----------|----------------|
| R01 | Risk | `predicted_score < 50` | Critical intervention needed |
| R03 | Risk | `attendance < 75 AND study_hours < 3` | Compound risk alert |
| A01 | Academic | `attendance < 60` | Increase attendance urgently |
| S01 | Study | `study_hours < 2` | Increase study time |
| L01 | Lifestyle | `sleep_hours < 5` | Address sleep deprivation |
| P01 | Academic | `attendance ≥ 85 AND study_hours ≥ 5` | Positive reinforcement |

Rules are fully independent from ML logic, organized by category (risk, academic, study, lifestyle) and priority (high, medium, low).

### Decision Tree (Syllabus Alignment)

**Integrated in**: `src/train_model.py`

The Decision Tree Regressor (`DecisionTreeRegressor`) is a foundational supervised learning algorithm relevant to the AI/ML syllabus:

- **How it works**: Recursively partitions the feature space using axis-aligned splits, creating interpretable decision rules (e.g., "IF study_hours > 5 AND attendance > 80 THEN predicted_score ≈ 78")
- **When it's useful**: When interpretability is more important than raw accuracy; when a quick baseline is needed; when understanding feature interactions matters
- **In this project**: It serves as both a standalone model and the building block for ensemble methods (Random Forest, Gradient Boosting). Comparing its performance against ensembles demonstrates the value of bagging and boosting — key syllabus concepts.
- **Hyperparameters**: `max_depth=8`, `min_samples_split=10` (controlled to prevent overfitting)

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+ (Built with Python 3.12+ compatibility in mind).

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data & Train the Model
This will run the preprocessing, train all 5 models, save the best one to `models/`, and log metrics.
```bash
set PYTHONIOENCODING=utf-8 && python -m src.train_model
```

### 3. View Analytical Notebook
An advanced Jupyter Data Analysis and EDA notebook is available in `notebook/analysis.ipynb`.
Open it in your preferred editor.

### 4. Run the Web Interface
Launch the Streamlit dashboard:
```bash
streamlit run app/app.py
```

---

## 📊 Dataset & Features

This system currently relies on synthetic data designed with realistic, non-linear relationships:
- **Study Hours** (Logarithmic effect, diminishing returns)
- **Sleep Hours** (Quadratic effect, 7 hours is optimal)
- **Distractions** (Linear penalty)
- **Attendance**, **Previous Score**, **Assignment Score**

Using a controlled synthetic dataset enforces a strong foundational understanding of Ground Truth before extending this to real-world deployment.

---

## 🔮 Future Improvements

- Integrate an explicit database (e.g., SQLite or PostgreSQL) via SQLAlchemy instead of resting entirely on a flat CSV.
- Containerize the application via Docker.
- Orchestrate the training pipeline using a tool like Airflow or Prefect.
- Expose the prediction module as a FastAPI REST endpoint.

