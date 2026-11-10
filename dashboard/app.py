"""
EduGuard AI — Academic Intelligence & Early Warning Platform
Streamlit Fallback Analytics Dashboard
Theme: Dark Navy (#0B0F19, #0F172A) with Indigo (#6366F1) & Cyan (#06B6D4) Accents.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Matplotlib cache directory
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DISCLAIMER_TEXT,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_VERSION,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.feature_engineering import compute_derived_features
from src.predict import EduGuardPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="EduGuard AI — Academic Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Enterprise CSS Styling ---
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .advisory-banner {
        background-color: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366F1;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        color: #C7D2FE;
        font-size: 0.9rem;
    }
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return EduGuardPredictor(enable_shap=True)


@st.cache_data
def load_cohort_data():
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)
    return None


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_evaluation_metrics():
    if EVALUATION_METRICS_PATH.exists():
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


predictor = load_predictor()
df_cohort = load_cohort_data()
metadata = load_metadata()
eval_metrics = load_evaluation_metrics()

st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #6366F1; margin: 0;'>🛡️ EduGuard AI</h2>
        <p style='color: #94A3B8; font-size: 0.85rem; margin-top: 4px;'>School Academic Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Suite",
    [
        "1. Executive Overview",
        "2. Student Risk Intelligence",
        "3. Explainable AI Lab (SHAP)",
        "4. Cohort Intelligence",
        "5. What-If Simulator",
        "6. Model Reliability Observatory",
        "7. Responsible AI & Audit Center",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style='font-size: 0.8rem; color: #64748B;'>
        <b>Model:</b> {predictor.model_name} ({'Calibrated' if predictor.is_calibrated else 'Base'})<br>
        <b>Version:</b> v{predictor.version}<br>
        <b>Threshold:</b> {predictor.threshold:.2f}<br>
        <b>Target:</b> Next Score &lt; 40 / 100
    </div>
    """,
    unsafe_allow_html=True,
)


if nav_choice == "1. Executive Overview":
    st.title("Executive Overview")
    st.markdown(
        f"<div class='advisory-banner'><b>Advisory System Notice:</b> {DISCLAIMER_TEXT}</div>",
        unsafe_allow_html=True,
    )

    if df_cohort is not None:
        total_students = df_cohort["student_id"].nunique()
        total_snapshots = len(df_cohort)
        at_risk_count = int(df_cohort[TARGET_COLUMN].sum())
        at_risk_pct = (at_risk_count / total_snapshots) * 100
        t_eval = metadata.get("final_test_metrics_temporal", {})
        recall_display = f"{t_eval['recall']*100:.1f}%" if "recall" in t_eval else "N/A"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Simulated Cohort</div>
                    <div class='metric-value'>{total_students:,}</div>
                    <span class='metric-badge badge-low'>{total_students:,} Unique Students</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Avg Attendance & Completion</div>
                    <div class='metric-value'>{df_cohort['attendance_pct'].mean():.1f}%</div>
                    <span class='metric-badge badge-low'>Completion: {df_cohort['assignment_completion_pct'].mean():.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>At-Risk Snapshots</div>
                    <div class='metric-value'>{at_risk_pct:.1f}%</div>
                    <span class='metric-badge badge-high'>{at_risk_count:,} Flagged Benchmarks</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Temporal Test Recall</div>
                    <div class='metric-value'>{recall_display}</div>
                    <span class='metric-badge badge-low'>Threshold τ* = {predictor.threshold:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Cohort-Level Assessment Trajectory")
        chart_col1, chart_col2 = st.columns([3, 2])
        with chart_col1:
            period_grp = df_cohort.groupby(["assessment_period", TARGET_COLUMN])["previous_score"].mean().reset_index()
            period_grp["Status"] = period_grp[TARGET_COLUMN].map({0: "Passing (>= 40)", 1: "At-Risk (< 40)"})
            fig_trend = px.line(
                period_grp,
                x="assessment_period",
                y="previous_score",
                color="Status",
                color_discrete_map={"Passing (>= 40)": "#10B981", "At-Risk (< 40)": "#EF4444"},
                markers=True,
                title="Historical Score Trajectory across Assessment Cycles",
                labels={"assessment_period": "Assessment Period", "previous_score": "Average Score"},
            )
            fig_trend.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A")
            fig_trend.add_hline(y=40, line_dash="dash", line_color="#EF4444", annotation_text="Benchmark 40")
            st.plotly_chart(fig_trend, use_container_width=True)

        with chart_col2:
            risk_counts = df_cohort[TARGET_COLUMN].value_counts().reset_index()
            risk_counts["Label"] = risk_counts[TARGET_COLUMN].map({0: "Passing Benchmark", 1: "At-Risk (< 40)"})
            fig_donut = px.pie(
                risk_counts,
                values="count",
                names="Label",
                color="Label",
                color_discrete_map={"Passing Benchmark": "#10B981", "At-Risk (< 40)": "#EF4444"},
                hole=0.55,
                title="Cohort Risk Distribution",
            )
            fig_donut.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A")
            st.plotly_chart(fig_donut, use_container_width=True)


elif nav_choice == "2. Student Risk Intelligence":
    st.title("Student Risk Intelligence")
    st.markdown(
        "<div class='advisory-banner'>Enter pre-assessment indicators to compute calibrated risk probability, "
        "SHAP feature drivers, and rule-based teacher support suggestions.</div>",
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Pre-Assessment Student Inputs")
        att = st.slider("Classroom Attendance (%)", min_value=0.0, max_value=100.0, value=74.0, step=0.5)
        comp = st.slider("Assignment Completion (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
        late = st.number_input("Late Assignment Count", min_value=0, max_value=25, value=2, step=1)

        c1, c2 = st.columns(2)
        with c1:
            p_score = st.number_input("Current Period Score (0-100)", min_value=0.0, max_value=100.0, value=48.0, step=1.0)
            period = st.selectbox("Current Assessment Period", options=[1, 2, 3, 4], index=1)
        with c2:
            p_score_2 = st.number_input("Prior Period Score (0-100)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
            derived = compute_derived_features(previous_score=float(p_score), previous_score_2=float(p_score_2))
            st.metric("Rolling-2 Average Score", f"{derived['previous_score_avg']:.2f}")

        st.info(f"Derived Score Momentum (`score_trend`): **{derived['score_trend']:+.1f} points**")

    payload = {
        "attendance_pct": float(att),
        "assignment_completion_pct": float(comp),
        "late_assignment_count": int(late),
        "previous_score": float(p_score),
        "previous_score_2": float(p_score_2),
        "assessment_period": int(period),
    }
    pred_res = predictor.predict_single(payload, include_shap=True)

    with col_out:
        st.subheader("Real-Time Advisory Assessment")
        prob = pred_res["risk_probability"]
        band = pred_res["risk_band"]
        thresh = pred_res["applied_threshold"]
        badge_class = "badge-high" if band["tier"] == "HIGH_RISK" else ("badge-moderate" if band["tier"] == "MODERATE_RISK" else "badge-low")

        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 6px solid {band["color"]};'>
                <div class='metric-title'>Assessed Risk Tier</div>
                <div class='metric-value' style='color: {band["color"]};'>{band["label"]}</div>
                <span class='metric-badge {badge_class}'>Calibrated Probability: {prob*100:.1f}% | Threshold: {thresh:.2f}</span>
                <p style='margin-top: 10px; color: #CBD5E1; font-size: 0.95rem;'><b>Advisory Action:</b> {band["action"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Probability of Scoring < 40", "font": {"color": "#F8FAFC"}},
            number={"suffix": "%", "font": {"color": band["color"], "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                "bar": {"color": band["color"]},
                "bgcolor": "#0F172A",
                "threshold": {"line": {"color": "#EF4444", "width": 3}, "thickness": 0.75, "value": thresh * 100},
            },
        ))
        fig_gauge.update_layout(template="plotly_dark", height=220, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="#0B0F19")
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.subheader("Teacher Support Suggestions (Rule-Based Advisory)")
    for plan in pred_res["support_plan"]:
        st.markdown(
            f"""
            <div style='background: #1E293B; border-left: 4px solid #6366F1; padding: 12px; border-radius: 6px; margin-bottom: 10px;'>
                <b style='color: #818CF8;'>[{plan['priority'].upper()} · RULE-BASED] {plan['category']}:</b> {plan['advisory_action']}
            </div>
            """,
            unsafe_allow_html=True,
        )


elif nav_choice == "3. Explainable AI Lab (SHAP)":
    st.title("Explainable AI Lab (SHAP)")
    st.markdown(
        "<div class='advisory-banner'>SHAP attributes model risk probability to individual inputs prior to validation sigmoid calibration. "
        "<b>Important:</b> SHAP reflects statistical associations in synthetic data, NOT causal mechanisms.</div>",
        unsafe_allow_html=True,
    )

    tab_local, tab_global = st.tabs(["Local Student Explanation", "Global Model Insights"])
    with tab_local:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            att_l = st.slider("Attendance (%)", 0.0, 100.0, 68.0, key="sh_att")
            comp_l = st.slider("Assignment Completion (%)", 0.0, 100.0, 58.0, key="sh_comp")
        with c2:
            late_l = st.number_input("Late Submissions", 0, 20, 3, key="sh_late")
            period_l = st.selectbox("Period", [1, 2, 3, 4], index=2, key="sh_per")
        with c3:
            prev_l = st.number_input("Current Score", 0.0, 100.0, 38.0, key="sh_prev")
            prev2_l = st.number_input("Prior Score", 0.0, 100.0, 46.0, key="sh_prev2")
        with c4:
            sh_derived = compute_derived_features(float(prev_l), float(prev2_l))
            st.metric("Rolling-2 Avg Score", f"{sh_derived['previous_score_avg']:.2f}")
            st.metric("Score Momentum", f"{sh_derived['score_trend']:+.1f}")

        sample_dict = {
            "attendance_pct": float(att_l),
            "assignment_completion_pct": float(comp_l),
            "late_assignment_count": int(late_l),
            "previous_score": float(prev_l),
            "previous_score_2": float(prev2_l),
            "assessment_period": int(period_l),
        }

        if predictor.explainer is not None:
            local_exp = predictor.explainer.explain_instance(sample_dict)
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 4px solid #6366F1;'>
                    <b>Plain-Language Teacher Summary:</b><br>
                    <span style='color: #E2E8F0;'>{local_exp['plain_language_explanation']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            contrib_df = pd.DataFrame(local_exp["feature_contributions"])
            contrib_df["Feature"] = contrib_df["feature"].str.replace("_", " ").str.title()
            fig_local = px.bar(
                contrib_df.sort_values(by="shap_value", ascending=True),
                x="shap_value",
                y="Feature",
                orientation="h",
                color="direction",
                color_discrete_map={"INCREASES_RISK": "#EF4444", "REDUCES_RISK": "#10B981"},
                title=f"Local SHAP Impact (Pre-Calibration Expected Base: {local_exp['expected_base_value']:.3f})",
            )
            fig_local.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A")
            st.plotly_chart(fig_local, use_container_width=True)
        else:
            st.warning("SHAP explainer module is currently unavailable.")

    with tab_global:
        col_g1, col_g2 = st.columns(2)
        summary_fig = FIGURES_DIR / "09_shap_summary.png"
        bar_fig = FIGURES_DIR / "10_shap_bar.png"
        with col_g1:
            if summary_fig.exists():
                st.image(str(summary_fig), use_container_width=True)
        with col_g2:
            if bar_fig.exists():
                st.image(str(bar_fig), use_container_width=True)


elif nav_choice == "4. Cohort Intelligence":
    st.title("Cohort Intelligence")
    st.markdown(
        "<div class='advisory-banner'>Anonymized school cohort patterns across attendance and assignment completion tiers.</div>",
        unsafe_allow_html=True,
    )

    if df_cohort is not None:
        c1, c2 = st.columns(2)
        with c1:
            att_tiers = pd.cut(
                df_cohort["attendance_pct"],
                bins=[0, 60, 75, 85, 100],
                labels=["< 60% Critical", "60-75% Low", "75-85% Moderate", "85-100% High"],
            )
            att_grp = df_cohort.groupby(att_tiers, observed=False)[TARGET_COLUMN].mean().mul(100).reset_index()
            att_grp.columns = ["Attendance Tier", "At-Risk Rate (%)"]
            fig_att = px.bar(att_grp, x="Attendance Tier", y="At-Risk Rate (%)", color="At-Risk Rate (%)", color_continuous_scale="Reds", title="At-Risk Rate (%) Across Attendance Tiers")
            fig_att.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A")
            st.plotly_chart(fig_att, use_container_width=True)
        with c2:
            comp_tiers = pd.cut(
                df_cohort["assignment_completion_pct"],
                bins=[0, 50, 70, 85, 100],
                labels=["< 50% Lagging", "50-70% Partial", "70-85% Substantial", "85-100% Complete"],
            )
            comp_grp = df_cohort.groupby(comp_tiers, observed=False)[TARGET_COLUMN].mean().mul(100).reset_index()
            comp_grp.columns = ["Completion Tier", "At-Risk Rate (%)"]
            fig_comp = px.bar(comp_grp, x="Completion Tier", y="At-Risk Rate (%)", color="At-Risk Rate (%)", color_continuous_scale="Purples", title="At-Risk Rate (%) Across Assignment Completion Tiers")
            fig_comp.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A")
            st.plotly_chart(fig_comp, use_container_width=True)


elif nav_choice == "5. What-If Simulator":
    st.title("What-If Simulator (Real ML Inference)")
    st.markdown(
        "<div class='advisory-banner'><b>Non-Causal Disclaimer:</b> Evaluates predictive sensitivity of the calibrated pipeline under hypothetical edits. Does NOT establish causal effects.</div>",
        unsafe_allow_html=True,
    )

    col_base, col_hypo = st.columns(2)
    with col_base:
        st.subheader("Baseline Student Profile")
        b_att = st.slider("Baseline Attendance (%)", 0.0, 100.0, 62.0)
        b_comp = st.slider("Baseline Completion (%)", 0.0, 100.0, 50.0)
        b_late = st.number_input("Baseline Late Assignments", 0, 20, 3)
        b_prev = st.slider("Baseline Current Score", 0.0, 100.0, 36.0)
        b_prev2 = st.slider("Baseline Prior Score", 0.0, 100.0, 44.0)
        res_base = predictor.predict_single(
            {
                "attendance_pct": float(b_att),
                "assignment_completion_pct": float(b_comp),
                "late_assignment_count": int(b_late),
                "previous_score": float(b_prev),
                "previous_score_2": float(b_prev2),
                "assessment_period": 2,
            },
            include_shap=False,
        )

    with col_hypo:
        st.subheader("Hypothetical Support Scenario")
        h_att = st.slider("Post-Support Attendance (%)", 0.0, 100.0, min(100.0, b_att + 20.0))
        h_comp = st.slider("Post-Support Completion (%)", 0.0, 100.0, min(100.0, b_comp + 30.0))
        h_late = st.number_input("Post-Support Late Assignments", 0, 20, max(0, b_late - 2))
        h_prev = st.slider("Post-Support Current Score", 0.0, 100.0, min(100.0, b_prev + 12.0))
        res_hypo = predictor.predict_single(
            {
                "attendance_pct": float(h_att),
                "assignment_completion_pct": float(h_comp),
                "late_assignment_count": int(h_late),
                "previous_score": float(h_prev),
                "previous_score_2": float(b_prev),
                "assessment_period": 2,
            },
            include_shap=False,
        )

    st.markdown("---")
    p_base = res_base["risk_probability"] * 100
    p_hypo = res_hypo["risk_probability"] * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Risk Probability", f"{p_base:.1f}%")
    c2.metric("Hypothetical Risk Probability", f"{p_hypo:.1f}%", delta=f"{p_hypo - p_base:+.1f}%", delta_color="inverse")
    c3.metric("Assessed Tier Shift", f"{res_base['risk_band']['label']} -> {res_hypo['risk_band']['label']}")


elif nav_choice == "6. Model Reliability Observatory":
    st.title("Model Reliability Observatory")
    t_eval = metadata.get("final_test_metrics_temporal", {})
    cal_sum = metadata.get("calibration_summary", {})

    if not t_eval:
        st.error("Evaluation metrics missing. Run `python -m src.train` to generate genuine metrics.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Temporal Average Precision", f"{t_eval['average_precision']:.3f}")
        m2.metric("Safety-Net Recall", f"{t_eval['recall']*100:.1f}%")
        m3.metric("Specificity / Precision", f"{t_eval['specificity']*100:.1f}% / {t_eval['precision']*100:.1f}%")
        m4.metric("Calibrated Brier Score", f"{t_eval['brier_score']:.4f}", delta=f"from {cal_sum.get('temporal_test_brier_before', t_eval['brier_score']):.4f}")

        st.markdown("---")
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            cm_data = t_eval["confusion_matrix"]
            cm_matrix = [
                [cm_data["true_negatives"], cm_data["false_positives"]],
                [cm_data["false_negatives"], cm_data["true_positives"]],
            ]
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Label", y="Actual Outcome", color="Students"),
                x=["Passing (>= 40)", "At-Risk (< 40)"],
                y=["Passing (>= 40)", "At-Risk (< 40)"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Held-Out Temporal Test Confusion Matrix (τ*={t_eval['threshold']})",
            )
            fig_cm.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A")
            st.plotly_chart(fig_cm, use_container_width=True)

        with cm_c2:
            if eval_metrics and "model_benchmarks" in eval_metrics:
                bench_rows = []
                for m_name, m_data in eval_metrics["model_benchmarks"].items():
                    t_m = m_data["temporal_validation"]
                    cal_c = m_data.get("calibration_comparison", {})
                    bench_rows.append({
                        "Model": m_name,
                        "Temp AP": t_m["average_precision"],
                        "ROC-AUC": t_m["roc_auc"],
                        "F1": t_m["f1_score"],
                        "Brier (After)": t_m["brier_score"],
                        "Brier (Before)": cal_c.get("temporal_brier_before"),
                        "Latency (ms)": m_data["inference_latency_ms"],
                    })
                st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)

            cal_fig = FIGURES_DIR / "11_calibration_curve.png"
            if cal_fig.exists():
                st.image(str(cal_fig), use_container_width=True)


elif nav_choice == "7. Responsible AI & Audit Center":
    st.title("Responsible AI & Governance Center")
    st.markdown(f"<div class='advisory-banner'><b>Ethical Mandate:</b> {DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)

    val_m = metadata.get("validation_metrics", {})
    thresh_val = metadata.get("selected_threshold", predictor.threshold)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            ### 1. Synthetic Scope & Real-World Caveat
            - **100% Synthetic Simulation:** {metadata.get('data_generator_parameters', {}).get('total_snapshots', 10000):,} snapshots across {metadata.get('data_generator_parameters', {}).get('num_students', 2500):,} simulated students.
            - **Unverified Real-World Generalization:** Model weights have not been benchmarked on live institutional records.
            - **Protected Attributes Excluded:** Names, gender, ethnicity, religion, caste, income, and disability never enter the pipeline.
            """
        )
    with col2:
        st.markdown(
            f"""
            ### 2. Decision Threshold Governance
            - **Champion Architecture:** `{metadata.get('model_name', predictor.model_name)}` (Sigmoid calibrated).
            - **Validation Threshold (`τ* = {thresh_val:.2f}`):** Tuned on disjoint Period 3 validation selection fold yielding {val_m.get('recall', 0)*100:.1f}% recall, {val_m.get('precision', 0)*100:.1f}% precision, and {val_m.get('specificity', 0)*100:.1f}% specificity.
            - **Strict Advisory Use:** Teachers retain full discretion over all academic support.
            """
        )

    st.markdown("---")
    st.caption(f"EduGuard AI v{MODEL_VERSION} | Trained: {metadata.get('training_timestamp_utc', 'N/A')}")
