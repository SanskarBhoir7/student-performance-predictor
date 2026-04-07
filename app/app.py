"""
Student Performance Predictor — Streamlit Application
=====================================================
A professional UI for predicting student exam scores with
confidence intervals, risk assessment, insights, and what-if analysis.
"""

import sys
import os

# Add project root to path so src imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

from src.predict import (
    predict_with_confidence,
    get_risk_level,
    get_insights,
    what_if_analysis,
    get_model_info,
    recommend_improvements,
    csp_recommend,
)
from src.data_preprocessing import FEATURES
from src.rule_engine import evaluate_rules, get_category_summary


# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15), transparent);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.65);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Score Display ── */
    .score-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .score-card:hover {
        transform: translateY(-2px);
    }
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.8;
    }
    .confidence-range {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* ── Risk Badge ── */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .risk-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ── Insight Cards ── */
    .insight-card {
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .insight-warning {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid #f59e0b;
    }
    .insight-info {
        background: rgba(59, 130, 246, 0.08);
        border-left: 3px solid #3b82f6;
    }
    .insight-success {
        background: rgba(16, 185, 129, 0.08);
        border-left: 3px solid #10b981;
    }

    /* ── Model Info Card ── */
    .model-info {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
    }

    /* ── Section Headers ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Slider styling ── */
    .stSlider > div > div {
        padding-top: 0.5rem;
    }

    /* ── Button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        font-size: 0.8rem;
        opacity: 0.4;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Predictor</h1>
    <p>ML-powered score prediction with confidence intervals, risk analysis & actionable insights</p>
</div>
""", unsafe_allow_html=True)


# ── Layout ─────────────────────────────────────────────────────────
col_input, col_spacer, col_results = st.columns([4, 0.5, 6])

# ── Input Panel ────────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="section-title">📝 Student Information</div>', unsafe_allow_html=True)
    
    study_hours = st.slider(
        "📚 Study Hours (per day)",
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="Average daily study hours outside of class"
    )

    attendance = st.slider(
        "📋 Attendance (%)",
        min_value=50, max_value=100, value=80, step=1,
        help="Percentage of classes attended"
    )

    previous_score = st.slider(
        "📈 Previous Exam Score",
        min_value=0, max_value=100, value=65, step=1,
        help="Score in the most recent exam"
    )

    assignment_score = st.slider(
        "📄 Assignment Score",
        min_value=0, max_value=100, value=60, step=1,
        help="Average score on assignments"
    )

    sleep_hours = st.slider(
        "😴 Sleep Hours (per night)",
        min_value=4.0, max_value=9.0, value=7.0, step=0.5,
        help="Average hours of sleep per night (optimal: ~7 hrs)"
    )

    distractions = st.slider(
        "🎮 Distraction Level",
        min_value=0.0, max_value=5.0, value=2.0, step=0.5,
        help="0 = fully focused, 5 = highly distracted"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    # Button removed to make app reactive and instantly show features

# ── Results Panel ──────────────────────────────────────────────────
with col_results:
    if True: # Always show results
        # Get prediction with confidence
        result = predict_with_confidence(
            study_hours=study_hours,
            attendance=float(attendance),
            previous_score=float(previous_score),
            assignment_score=float(assignment_score),
            sleep_hours=sleep_hours,
            distractions=distractions,
        )

        score = result["prediction"]
        lower = result["lower"]
        upper = result["upper"]

        # Get risk level
        risk = get_risk_level(score)

        # Score Card
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Predicted Score</div>
            <div class="score-value">{score:.1f}</div>
            <div class="confidence-range">
                📊 Confidence Interval: {lower:.1f} – {upper:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk Badge
        risk_class = risk["level"].split()[0].lower()
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span class="risk-badge risk-{risk_class}">
                {risk["emoji"]} {risk["level"]}
            </span>
        </div>
        <p style="text-align: center; font-size: 0.85rem; opacity: 0.7;">
            {risk["description"]}
        </p>
        """, unsafe_allow_html=True)

        # ── Insights ──────────────────────────────────────────────
        st.markdown('<div class="section-title">💡 Actionable Insights</div>', unsafe_allow_html=True)

        insights = get_insights(
            study_hours, float(attendance), float(previous_score),
            float(assignment_score), sleep_hours, distractions,
        )
        for ins in insights:
            css_class = f"insight-{ins['type']}"
            st.markdown(
                f'<div class="insight-card {css_class}">{ins["message"]}</div>',
                unsafe_allow_html=True,
            )

# ── Bottom Section ────────────────────────────────────────────────
st.markdown("<hr style='opacity: 0.15; margin: 2rem 0;'>", unsafe_allow_html=True)
col_b_left, col_b_right = st.columns([1, 1], gap="large")

with col_b_left:
    if True:
        # ── What-If Analysis ──────────────────────────────────────
        st.markdown('<div class="section-title">🔬 What-If Analysis</div>', unsafe_allow_html=True)

        base_features = {
            "study_hours": study_hours,
            "attendance": float(attendance),
            "previous_score": float(previous_score),
            "assignment_score": float(assignment_score),
            "sleep_hours": sleep_hours,
            "distractions": distractions,
        }

        feature_labels = {
            "study_hours": "📚 Study Hours",
            "attendance": "📋 Attendance",
            "previous_score": "📈 Previous Score",
            "assignment_score": "📄 Assignment Score",
            "sleep_hours": "😴 Sleep Hours",
            "distractions": "🎮 Distractions",
        }

        selected_feature = st.selectbox(
            "Vary this feature to see how your score changes:",
            options=FEATURES,
            format_func=lambda x: feature_labels.get(x, x),
        )

        wif_df = what_if_analysis(base_features, selected_feature)

        # Create a Plotly chart
        fig = go.Figure()

        # Main line
        fig.add_trace(go.Scatter(
            x=wif_df[selected_feature],
            y=wif_df["predicted_score"],
            mode="lines",
            line=dict(color="#667eea", width=3),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.08)",
            name="Predicted Score",
        ))

        # Current value marker
        fig.add_trace(go.Scatter(
            x=[base_features[selected_feature]],
            y=[score],
            mode="markers+text",
            marker=dict(color="#f59e0b", size=14, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[f"You: {score:.1f}"],
            textposition="top center",
            textfont=dict(size=13, color="#f59e0b"),
            name="Current Value",
        ))

        fig.update_layout(
            xaxis_title=feature_labels.get(selected_feature, selected_feature),
            yaxis_title="Predicted Score",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Model Info ────────────────────────────────────────────
        model_info = get_model_info()
        if model_info:
            best = model_info["best_model"]
            metrics = model_info["results"][best]

            st.markdown('<div class="section-title">🤖 Model Information</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="model-info">
                <strong>Active Model:</strong> {best}<br>
                <strong>R² Score:</strong> {metrics['R2']:.4f} &nbsp;|&nbsp;
                <strong>MAE:</strong> {metrics['MAE']:.4f} &nbsp;|&nbsp;
                <strong>RMSE:</strong> {metrics['RMSE']:.4f}
            </div>
            """, unsafe_allow_html=True)

            # Model comparison bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            results = model_info["results"]
            comparison_df = pd.DataFrame([
                {"Model": name, "R²": m["R2"], "MAE": m["MAE"]}
                for name, m in results.items()
            ])

            fig2 = go.Figure()
            colors = ["#667eea", "#764ba2", "#06b6d4", "#f59e0b", "#10b981"]
            for i, row in comparison_df.iterrows():
                fig2.add_trace(go.Bar(
                    x=[row["Model"]],
                    y=[row["R²"]],
                    name=row["Model"],
                    marker_color=colors[i % len(colors)],
                    text=[f"{row['R²']:.3f}"],
                    textposition="outside",
                ))

            fig2.update_layout(
                yaxis_title="R² Score",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=40, r=20, t=10, b=40),
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig2, use_container_width=True)

with col_b_right:
    if True:
        # ── Goal-Seek Calculator ──────────────────────────────────
        st.markdown('<div class="section-title">🎯 Goal-Seek Calculator</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.9rem; opacity:0.8;'>Want a higher score? Find out exactly what habits you need to change.</p>", unsafe_allow_html=True)
        
        col_t, col_bt = st.columns([2, 1])
        with col_t:
            target_goal = st.number_input("Target Score", min_value=float(score), max_value=100.0, value=min(100.0, score + 10.0), step=1.0)
        with col_bt:
            st.markdown("<br>", unsafe_allow_html=True)
            run_goal = st.button("Generate Action Plan")
            
        if run_goal:
            with st.spinner("Finding optimal path..."):
                plan = recommend_improvements(base_features, target_goal)
                
                if plan["status"] == "already_reached":
                    st.success("You are already predicted to reach this score!")
                elif plan["status"] == "impossible":
                    st.error("That target score might be out of reach given the current previous score.")
                else:
                    if plan["status"] == "best_effort":
                        st.warning(f"Target of {target_goal} might be slightly out of reach. Here is the **best possible** scenario:")
                    else:
                        st.success("Target is achievable! Here are the easiest ways to reach it:")
                    
                    for i, option in enumerate(plan["options"]):
                        opt_feats = option["features"]
                        changes = []
                        if opt_feats["study_hours"] > base_features["study_hours"]:
                            changes.append(f"↑ Add {opt_feats['study_hours'] - base_features['study_hours']:.1f} hrs of study")
                        if opt_feats["attendance"] > base_features["attendance"]:
                            changes.append(f"↑ Boost attendance by {opt_feats['attendance'] - base_features['attendance']:.0f}%")
                        if opt_feats["distractions"] < base_features["distractions"]:
                            changes.append(f"↓ Reduce distractions by {base_features['distractions'] - opt_feats['distractions']:.1f}")
                        if opt_feats["sleep_hours"] != base_features["sleep_hours"]:
                            diff = opt_feats["sleep_hours"] - base_features["sleep_hours"]
                            sign = "↑ Add" if diff > 0 else "↓ Reduce"
                            changes.append(f"{sign} {abs(diff):.1f} hrs of sleep")
                        if opt_feats["assignment_score"] > base_features["assignment_score"]:
                            changes.append(f"↑ Improve assignments by {opt_feats['assignment_score'] - base_features['assignment_score']:.0f} pts")
                            
                        st.markdown(f"**Plan {i+1}: Predicted Score ~ {option['predicted_score']:.1f}**")
                        for c in changes:
                            st.markdown(f"- {c}")
                        st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2'>", unsafe_allow_html=True)

        # ── CSP Solver (Expander) ─────────────────────────────────
        st.markdown('<div class="section-title">🧩 AI-Powered Optimization</div>', unsafe_allow_html=True)

        with st.expander("🎯 How to Achieve Target Score (CSP)", expanded=False):
            st.markdown(
                "<p style='font-size:0.9rem; opacity:0.8; margin-bottom:1rem;'>"
                "The <b>Constraint Satisfaction Problem (CSP)</b> solver uses backtracking search "
                "with your trained ML model to find valid input combinations that meet a target score, "
                "while respecting realistic constraints (attendance ≥ 75%, sleep 5-8 hrs, etc.)."
                "</p>",
                unsafe_allow_html=True,
            )

            col_csp_target, col_csp_btn = st.columns([2, 1])
            with col_csp_target:
                csp_target = st.number_input(
                    "Target Score for CSP",
                    min_value=50.0, max_value=100.0, value=80.0, step=5.0,
                    key="csp_target",
                )
            with col_csp_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                run_csp = st.button("🔍 Find Optimal Inputs", key="csp_btn")

            if run_csp:
                with st.spinner("Running CSP solver (backtracking search)..."):
                    csp_solutions = csp_recommend(base_features, csp_target, max_solutions=3)

                if not csp_solutions:
                    st.warning(
                        f"No valid combination found that achieves {csp_target:.0f} "
                        f"within the defined constraints. Try a lower target."
                    )
                else:
                    st.success(
                        f"Found {len(csp_solutions)} valid combination(s) "
                        f"achieving ≥ {csp_target:.0f}:"
                    )

                    for i, sol in enumerate(csp_solutions):
                        feats = sol["features"]
                        st.markdown(
                            f"**Solution {i+1}** — "
                            f"Predicted Score: **{sol['predicted_score']:.1f}** "
                            f"(effort: {sol['effort']:.1f})"
                        )

                        # Show as a compact table
                        changes = []
                        label_map = {
                            "study_hours": "📚 Study Hours",
                            "attendance": "📋 Attendance (%)",
                            "sleep_hours": "😴 Sleep Hours",
                            "distractions": "🎮 Distractions",
                            "assignment_score": "📄 Assignment Score",
                        }
                        for var in ["study_hours", "attendance", "sleep_hours",
                                    "distractions", "assignment_score"]:
                            curr = base_features[var]
                            rec = feats[var]
                            if abs(rec - curr) > 0.01:
                                direction = "↑" if rec > curr else "↓"
                                changes.append({
                                    "Feature": label_map.get(var, var),
                                    "Current": f"{curr:.1f}",
                                    "Recommended": f"{rec:.1f}",
                                    "Change": f"{direction} {abs(rec - curr):.1f}",
                                })

                        if changes:
                            st.table(pd.DataFrame(changes))
                        else:
                            st.info("Current inputs already satisfy the target!")

                        st.markdown(
                            "<hr style='margin: 0.5rem 0; opacity: 0.15'>",
                            unsafe_allow_html=True,
                        )

        # ── Rule-Based Assessment (Expander) ──────────────────────
        with st.expander("📋 Detailed Insights & Recommendations (Rule-Based)", expanded=False):
            st.markdown(
                "<p style='font-size:0.9rem; opacity:0.8; margin-bottom:1rem;'>"
                "This <b>Rule-Based Reasoning Engine</b> applies expert-defined IF-THEN rules "
                "to assess your academic profile. Unlike ML predictions, these rules are fully "
                "interpretable and based on established domain knowledge."
                "</p>",
                unsafe_allow_html=True,
            )

            # Build features dict with predicted score for rule evaluation
            rule_features = dict(base_features)
            rule_features["predicted_score"] = score

            triggered_rules = evaluate_rules(rule_features)

            if not triggered_rules:
                st.success("No concerns detected — all indicators are positive!")
            else:
                categories = get_category_summary(triggered_rules)
                category_icons = {
                    "risk": "🔴 Risk Assessment",
                    "academic": "📚 Academic Factors",
                    "study": "📖 Study Habits",
                    "lifestyle": "🌙 Lifestyle Factors",
                }
                priority_colors = {
                    "high": "#ef4444",
                    "medium": "#f59e0b",
                    "low": "#10b981",
                }

                for cat_name in ["risk", "academic", "study", "lifestyle"]:
                    if cat_name not in categories:
                        continue
                    rules = categories[cat_name]
                    cat_label = category_icons.get(cat_name, cat_name.title())
                    st.markdown(f"**{cat_label}**")

                    for r in rules:
                        color = priority_colors.get(r["priority"], "#6b7280")
                        st.markdown(
                            f'<div style="padding: 0.7rem 1rem; border-radius: 8px; '
                            f'margin-bottom: 0.4rem; font-size: 0.88rem; '
                            f'border-left: 3px solid {color}; '
                            f'background: {color}10;">'
                            f'<span style="opacity:0.5; font-size:0.75rem;">[{r["rule_id"]}]</span> '
                            f'{r["message"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("", unsafe_allow_html=True)

                st.markdown(
                    f"<p style='font-size:0.8rem; opacity:0.5; margin-top:0.5rem;'>"
                    f"📊 {len(triggered_rules)} rule(s) triggered"
                    f"</p>",
                    unsafe_allow_html=True,
                )

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit, scikit-learn & Plotly · Student Performance Predictor v2.0 (Hybrid AI)
</div>
""", unsafe_allow_html=True)
