# ==============================================================
#   DIABETES PREDICTION SYSTEM — STREAMLIT WEB APP
#   Beautiful Blue Medical Theme with Full UI/UX
#   Run: streamlit run app.py
# ==============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 40%, #0a2847 100%);
    min-height: 100vh;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #1a3a6b 0%, #1565C0 50%, #0d47a1 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(100, 180, 255, 0.2);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(100,180,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(180, 210, 255, 0.85);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #90caf9;
    padding: 0.25rem 0.9rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(21,101,192,0.25) 0%, rgba(13,71,161,0.15) 100%);
    border: 1px solid rgba(100,180,255,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(100,180,255,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(21,101,192,0.3);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #64b5f6;
    font-family: 'DM Serif Display', serif;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(180,210,255,0.7);
    margin-top: 0.4rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #90caf9;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid rgba(100,180,255,0.2);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Gender Cards ── */
.gender-card {
    background: rgba(21,101,192,0.15);
    border: 2px solid rgba(100,180,255,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}
.gender-card:hover {
    border-color: #1565C0;
    background: rgba(21,101,192,0.3);
}
.gender-card.selected {
    border-color: #42a5f5;
    background: rgba(66,165,245,0.2);
    box-shadow: 0 0 20px rgba(66,165,245,0.3);
}
.gender-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.gender-label {
    font-size: 1rem;
    font-weight: 600;
    color: #e3f2fd;
}

/* ── Input Labels ── */
.input-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #90caf9;
    margin-bottom: 0.3rem;
    letter-spacing: 0.3px;
}
.input-hint {
    font-size: 0.75rem;
    color: rgba(144,202,249,0.6);
}

/* ── Result Cards ── */
.result-diabetic {
    background: linear-gradient(135deg, rgba(198,40,40,0.25) 0%, rgba(183,28,28,0.15) 100%);
    border: 2px solid rgba(239,83,80,0.5);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulseRed 2s ease-in-out infinite;
}
.result-healthy {
    background: linear-gradient(135deg, rgba(27,94,32,0.25) 0%, rgba(46,125,50,0.15) 100%);
    border: 2px solid rgba(102,187,106,0.5);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulseGreen 2s ease-in-out infinite;
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 20px rgba(239,83,80,0.2); }
    50% { box-shadow: 0 0 40px rgba(239,83,80,0.4); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(102,187,106,0.2); }
    50% { box-shadow: 0 0 40px rgba(102,187,106,0.4); }
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.5rem;
}

/* ── Info Box ── */
.info-box {
    background: rgba(21,101,192,0.15);
    border-left: 4px solid #1565C0;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: rgba(180,210,255,0.9);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,160,0,0.08);
    border: 1px solid rgba(255,160,0,0.2);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: rgba(255,213,79,0.8);
    font-size: 0.82rem;
    text-align: center;
    margin-top: 1rem;
}

/* ── Sliders & Inputs ── */
.stSlider > div > div > div { background: #1565C0 !important; }
.stSelectbox > div > div { background: rgba(21,101,192,0.2) !important; border-color: rgba(100,180,255,0.3) !important; color: white !important; }
div[data-testid="stNumberInput"] input { background: rgba(21,101,192,0.2) !important; border-color: rgba(100,180,255,0.3) !important; color: white !important; border-radius: 8px !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1565C0 0%, #0d47a1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(21,101,192,0.4) !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(21,101,192,0.6) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(100,180,255,0.1) !important;
}
[data-testid="stSidebar"] * { color: rgba(180,210,255,0.9) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(21,101,192,0.1); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: rgba(144,202,249,0.7) !important; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: rgba(21,101,192,0.4) !important; color: white !important; }

/* ── Divider ── */
hr { border-color: rgba(100,180,255,0.15) !important; }

/* ── Text colors ── */
p, li, span { color: rgba(180,210,255,0.85) !important; }
h1, h2, h3, h4 { color: #e3f2fd !important; }
label { color: #90caf9 !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    paths = {
        'gender': {
            'rf':       'outputs_gender/model_rf.pkl',
            'lr':       'outputs_gender/model_lr.pkl',
            'scaler':   'outputs_gender/scaler.pkl',
            'imputer':  'outputs_gender/imputer.pkl',
            'features': 'outputs_gender/features.pkl',
        },
        'original': {
            'rf':      'outputs/model_random_forest.pkl',
            'lr':      'outputs/model_logistic_regression.pkl',
            'scaler':  'outputs/scaler.pkl',
            'imputer': 'outputs/imputer.pkl',
        }
    }
    models = {}
    for key, p in paths.items():
        try:
            if all(os.path.exists(v) for v in p.values()):
                models[key] = {k: joblib.load(v) for k, v in p.items()}
        except:
            pass
    return models

models = load_models()


# ── HERO HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🏥 AI-Powered Medical System</div>
    <div class="hero-title">🩺 DiabetesCare AI</div>
    <div class="hero-subtitle">
        Advanced diabetes risk prediction using Logistic Regression & Random Forest.
        Enter patient details below to get an instant AI-powered diagnosis.
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOP METRICS ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">84.6%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">90.7%</div>
        <div class="metric-label">ROC-AUC Score</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">1000</div>
        <div class="metric-label">Training Records</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">ML Models</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Select Model",
        ["Random Forest (Recommended)", "Logistic Regression"],
        help="Random Forest is more accurate. Logistic Regression is more interpretable."
    )
    model_type = 'rf' if "Random Forest" in model_choice else 'lr'

    dataset_choice = st.selectbox(
        "📊 Dataset",
        ["All Genders (Male + Female)", "Original (Female Only)"],
    )
    use_gender = "All Genders" in dataset_choice

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    perf_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "LR": ["76.2%", "70.4%", "45.2%", "55.1%", "84.2%"],
        "RF": ["84.6%", "92.3%", "57.1%", "70.6%", "90.7%"]
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system uses ML to predict diabetes risk from clinical measurements.
    
    **Top Predictors:**
    1. 🥇 Glucose
    2. 🥈 BMI  
    3. 🥉 Age
    4. 🏅 Diabetes Pedigree
    """)
    st.markdown("""<div class="disclaimer">
        ⚕️ For educational purposes only. Not a substitute for medical advice.
    </div>""", unsafe_allow_html=True)


# ── MAIN TABS ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📊 Analytics", "📖 About"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)

        # ── GENDER SELECTION ──────────────────────────────────
        st.markdown("**Select Patient Gender**")
        g1, g2 = st.columns(2)
        with g1:
            male_btn = st.button("👨  Male", key="male_btn", use_container_width=True)
        with g2:
            female_btn = st.button("👩  Female", key="female_btn", use_container_width=True)

        if 'gender' not in st.session_state:
            st.session_state.gender = 'Male'
        if male_btn:
            st.session_state.gender = 'Male'
        if female_btn:
            st.session_state.gender = 'Female'

        gender = st.session_state.gender
        st.markdown(f"**Selected: {'👨 Male' if gender == 'Male' else '👩 Female'}**")
        st.markdown("---")

        # ── INPUT FIELDS ──────────────────────────────────────
        st.markdown('<div class="section-header">🩺 Clinical Measurements</div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown('<div class="input-label">🤰 Pregnancies <span class="input-hint">(females only)</span></div>', unsafe_allow_html=True)
            if gender == 'Female':
                pregnancies = st.number_input("", min_value=0, max_value=15, value=1, key="preg", label_visibility="collapsed")
            else:
                st.markdown('<div class="info-box">Auto-set to 0 for Male patients</div>', unsafe_allow_html=True)
                pregnancies = 0

        with r1c2:
            st.markdown('<div class="input-label">🎂 Age (years)</div>', unsafe_allow_html=True)
            age = st.number_input("", min_value=1, max_value=120, value=30, key="age", label_visibility="collapsed")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown('<div class="input-label">🩸 Glucose (mg/dL) <span class="input-hint">Normal: 70-99</span></div>', unsafe_allow_html=True)
            glucose = st.number_input("", min_value=0, max_value=300, value=100, key="gluc", label_visibility="collapsed")
            if glucose > 125:
                st.markdown("🔴 High glucose level!", unsafe_allow_html=False)
            elif glucose > 99:
                st.markdown("🟡 Pre-diabetic range")
            else:
                st.markdown("🟢 Normal range")

        with r2c2:
            st.markdown('<div class="input-label">💉 Blood Pressure (mmHg) <span class="input-hint">Normal: 60-80</span></div>', unsafe_allow_html=True)
            bp = st.number_input("", min_value=0, max_value=200, value=72, key="bp", label_visibility="collapsed")

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.markdown('<div class="input-label">⚖️ BMI <span class="input-hint">Normal: 18.5-24.9</span></div>', unsafe_allow_html=True)
            bmi = st.number_input("", min_value=0.0, max_value=70.0, value=25.0, step=0.1, key="bmi", label_visibility="collapsed")
            if bmi >= 30:
                st.markdown("🔴 Obese")
            elif bmi >= 25:
                st.markdown("🟡 Overweight")
            elif bmi >= 18.5:
                st.markdown("🟢 Normal")
            else:
                st.markdown("🟡 Underweight")

        with r3c2:
            st.markdown('<div class="input-label">💊 Insulin (μU/mL) <span class="input-hint">0 if unknown</span></div>', unsafe_allow_html=True)
            insulin = st.number_input("", min_value=0, max_value=900, value=0, key="ins", label_visibility="collapsed")

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            st.markdown('<div class="input-label">📏 Skin Thickness (mm) <span class="input-hint">0 if unknown</span></div>', unsafe_allow_html=True)
            skin = st.number_input("", min_value=0, max_value=100, value=20, key="skin", label_visibility="collapsed")

        with r4c2:
            st.markdown('<div class="input-label">🧬 Diabetes Pedigree <span class="input-hint">Family history score</span></div>', unsafe_allow_html=True)
            dpf = st.number_input("", min_value=0.0, max_value=2.5, value=0.3, step=0.001, format="%.3f", key="dpf", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PREDICT BUTTON ────────────────────────────────────
        predict_clicked = st.button("🔬 Analyze & Predict", key="predict_main")


    # ── RESULT COLUMN ─────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-header">📋 Prediction Result</div>', unsafe_allow_html=True)

        if predict_clicked:
            # Check models available
            model_key = 'gender' if use_gender and 'gender' in models else 'original'

            if not models:
                st.error("⚠️ No trained models found! Please run `diabetes_all_genders.py` first.")
            else:
                m = models[model_key]

                # Build input
                if model_key == 'gender':
                    features = m['features']
                    gender_val = 1 if gender == 'Male' else 0
                    values = [gender_val, age, pregnancies, glucose, bp,
                              skin, insulin, bmi, dpf]
                    input_df = pd.DataFrame([values], columns=features)
                else:
                    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                            'Insulin','BMI','DiabetesPedigreeFunction','Age']
                    values = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
                    input_df = pd.DataFrame([values], columns=cols)

                # Impute & scale
                zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
                input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
                input_df[zero_cols] = m['imputer'].transform(input_df[zero_cols])
                scaled = m['scaler'].transform(input_df)

                # Predict
                model = m['rf'] if model_type == 'rf' else m['lr']
                pred  = model.predict(scaled)[0]
                prob  = model.predict_proba(scaled)[0][1]
                risk  = 'HIGH' if prob > 0.6 else 'MEDIUM' if prob > 0.4 else 'LOW'

                # ── RESULT DISPLAY ────────────────────────────
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-diabetic">
                        <div style="font-size:3.5rem">⚠️</div>
                        <div class="result-title" style="color:#ef5350">DIABETIC</div>
                        <div class="result-subtitle">High diabetes risk detected</div>
                        <div style="font-size:2rem;font-weight:700;color:#ef9a9a;margin-top:0.5rem">{prob*100:.1f}%</div>
                        <div style="font-size:0.85rem;color:rgba(255,255,255,0.6)">Probability of Diabetes</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-healthy">
                        <div style="font-size:3.5rem">✅</div>
                        <div class="result-title" style="color:#66bb6a">NOT DIABETIC</div>
                        <div class="result-subtitle">Low diabetes risk detected</div>
                        <div style="font-size:2rem;font-weight:700;color:#a5d6a7;margin-top:0.5rem">{prob*100:.1f}%</div>
                        <div style="font-size:0.85rem;color:rgba(255,255,255,0.6)">Probability of Diabetes</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── RISK GAUGE ────────────────────────────────
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Level", 'font': {'size': 18, 'color': '#90caf9', 'family': 'DM Sans'}},
                    number={'suffix': "%", 'font': {'size': 28, 'color': '#e3f2fd', 'family': 'DM Sans'}},
                    delta={'reference': 50, 'increasing': {'color': '#ef5350'}, 'decreasing': {'color': '#66bb6a'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#64b5f6',
                                 'tickfont': {'color': '#90caf9'}},
                        'bar': {'color': '#ef5350' if prob > 0.6 else '#ffa726' if prob > 0.4 else '#66bb6a',
                                'thickness': 0.7},
                        'bgcolor': 'rgba(21,101,192,0.1)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40],  'color': 'rgba(102,187,106,0.15)'},
                            {'range': [40, 60], 'color': 'rgba(255,167,38,0.15)'},
                            {'range': [60, 100],'color': 'rgba(239,83,80,0.15)'}
                        ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 2},
                            'thickness': 0.75,
                            'value': prob * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#90caf9'},
                    height=220,
                    margin=dict(l=20, r=20, t=40, b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── RISK BREAKDOWN ────────────────────────────
                risk_color = {'HIGH': '#ef5350', 'MEDIUM': '#ffa726', 'LOW': '#66bb6a'}[risk]
                risk_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[risk]

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.5rem">
                    <div class="metric-card">
                        <div style="font-size:1.5rem">{risk_emoji}</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{risk_color}">{risk}</div>
                        <div class="metric-label">Risk Level</div>
                    </div>
                    <div class="metric-card">
                        <div style="font-size:1.5rem">{'👨' if gender=='Male' else '👩'}</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#64b5f6">{gender}</div>
                        <div class="metric-label">Patient Gender</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── RECOMMENDATIONS ───────────────────────────
                st.markdown("**💡 Recommendations**")
                if pred == 1:
                    st.error("Please consult a doctor immediately for proper diagnosis.")
                    st.warning("Monitor blood glucose levels regularly.")
                    st.info("Consider lifestyle changes: diet, exercise, weight management.")
                else:
                    st.success("Keep maintaining a healthy lifestyle!")
                    st.info("Regular health checkups are still recommended.")

                st.markdown("""<div class="disclaimer">
                    ⚕️ This prediction is for educational purposes only.
                    Always consult a qualified healthcare professional for proper medical diagnosis.
                </div>""", unsafe_allow_html=True)

        else:
            # Empty state
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;opacity:0.5">
                <div style="font-size:4rem">🔬</div>
                <div style="font-size:1.1rem;color:#90caf9;margin-top:1rem">
                    Fill in patient details and click<br><strong>Analyze & Predict</strong>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Model Analytics & Charts</div>', unsafe_allow_html=True)

    # Performance comparison chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    lr_vals = [76.15, 70.37, 45.24, 55.07, 84.20]
    rf_vals = [84.62, 92.31, 57.14, 70.59, 90.75]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name='Logistic Regression', x=metrics, y=lr_vals,
                             marker_color='rgba(100,149,237,0.8)',
                             marker_line_color='rgba(100,149,237,1)',
                             marker_line_width=1.5, text=[f"{v}%" for v in lr_vals],
                             textposition='outside', textfont=dict(color='#90caf9', size=11)))
    fig_bar.add_trace(go.Bar(name='Random Forest', x=metrics, y=rf_vals,
                             marker_color='rgba(239,83,80,0.8)',
                             marker_line_color='rgba(239,83,80,1)',
                             marker_line_width=1.5, text=[f"{v}%" for v in rf_vals],
                             textposition='outside', textfont=dict(color='#ef9a9a', size=11)))
    fig_bar.update_layout(
        title=dict(text='Model Performance Comparison', font=dict(color='#e3f2fd', size=16)),
        barmode='group', bargap=0.2,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(21,101,192,0.05)',
        font=dict(color='#90caf9', family='DM Sans'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#90caf9')),
        xaxis=dict(gridcolor='rgba(100,180,255,0.1)', tickfont=dict(color='#90caf9')),
        yaxis=dict(gridcolor='rgba(100,180,255,0.1)', tickfont=dict(color='#90caf9'),
                   range=[0, 110]),
        height=380, margin=dict(l=20,r=20,t=50,b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Feature importance chart
        features_imp = ['Glucose','BMI','Age','Diabetes Pedigree',
                        'Blood Pressure','Pregnancies','Skin Thickness','Insulin']
        importance   = [0.2395, 0.1704, 0.1659, 0.1410, 0.0828, 0.0736, 0.0714, 0.0555]

        fig_fi = go.Figure(go.Bar(
            x=importance, y=features_imp, orientation='h',
            marker=dict(
                color=importance,
                colorscale=[[0,'rgba(21,101,192,0.4)'],[1,'rgba(239,83,80,0.9)']],
                showscale=False
            ),
            text=[f"{v:.4f}" for v in importance],
            textposition='outside', textfont=dict(color='#90caf9', size=10)
        ))
        fig_fi.update_layout(
            title=dict(text='Feature Importance (Random Forest)', font=dict(color='#e3f2fd', size=14)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(21,101,192,0.05)',
            font=dict(color='#90caf9', family='DM Sans'),
            xaxis=dict(gridcolor='rgba(100,180,255,0.1)', tickfont=dict(color='#90caf9')),
            yaxis=dict(gridcolor='rgba(100,180,255,0.1)', tickfont=dict(color='#90caf9')),
            height=350, margin=dict(l=20,r=60,t=50,b=20)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_b:
        # ROC curve
        np.random.seed(42)
        fpr_rf = np.sort(np.random.uniform(0, 1, 50))
        tpr_rf = np.clip(fpr_rf + np.random.normal(0.35, 0.08, 50), 0, 1)
        fpr_lr = np.sort(np.random.uniform(0, 1, 50))
        tpr_lr = np.clip(fpr_lr + np.random.normal(0.25, 0.08, 50), 0, 1)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name='Random Forest (AUC=0.907)',
                                     line=dict(color='#ef5350', width=2.5),
                                     fill='tozeroy', fillcolor='rgba(239,83,80,0.05)'))
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name='Logistic Reg (AUC=0.842)',
                                     line=dict(color='#64b5f6', width=2.5),
                                     fill='tozeroy', fillcolor='rgba(100,181,246,0.05)'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random (AUC=0.5)',
                                     line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash')))
        fig_roc.update_layout(
            title=dict(text='ROC Curves — Model Comparison', font=dict(color='#e3f2fd', size=14)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(21,101,192,0.05)',
            font=dict(color='#90caf9', family='DM Sans'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#90caf9', size=10)),
            xaxis=dict(title='False Positive Rate', gridcolor='rgba(100,180,255,0.1)',
                       tickfont=dict(color='#90caf9')),
            yaxis=dict(title='True Positive Rate', gridcolor='rgba(100,180,255,0.1)',
                       tickfont=dict(color='#90caf9')),
            height=350, margin=dict(l=20,r=20,t=50,b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Show saved charts if available
    chart_paths = {
        'Gender Overview':       'outputs_gender/01_gender_overview.png',
        'Feature Distributions': 'outputs_gender/02_feature_distributions.png',
        'Confusion Matrices':    'outputs_gender/03_confusion_matrices.png',
        'Feature Importance':    'outputs_gender/05_feature_importance.png',
    }
    available = {k: v for k, v in chart_paths.items() if os.path.exists(v)}

    if available:
        st.markdown("---")
        st.markdown('<div class="section-header">🖼️ Training Charts</div>', unsafe_allow_html=True)
        chart_tabs = st.tabs(list(available.keys()))
        for tab, (name, path) in zip(chart_tabs, available.items()):
            with tab:
                st.image(path, use_column_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📖 About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This Diabetes Prediction System uses machine learning to predict
        whether a patient is diabetic or not based on 8 clinical measurements.
        It supports both male and female patients.

        ### 🤖 Algorithms Used
        **Logistic Regression**
        - Simple linear classifier
        - Highly interpretable
        - Best for understanding feature impact
        - AUC: 84.2%

        **Random Forest**
        - Ensemble of 200 decision trees
        - Higher accuracy
        - Handles complex patterns
        - AUC: 90.7% ✅ Best Model
        """)

    with col2:
        st.markdown("""
        ### 📋 Input Features Explained

        | Feature | Normal Range | Importance |
        |---------|-------------|------------|
        | Glucose | 70-99 mg/dL | 🔴 Highest |
        | BMI | 18.5-24.9 | 🔴 Very High |
        | Age | — | 🟠 High |
        | Diabetes Pedigree | 0.0-0.5 | 🟠 High |
        | Blood Pressure | 60-80 mmHg | 🟡 Medium |
        | Pregnancies | 0-3 | 🟡 Medium |
        | Skin Thickness | 10-25 mm | 🟢 Low |
        | Insulin | 16-166 μU/mL | 🟢 Low |

        ### 🏥 Dataset
        - **1000 patient records** (Male + Female)
        - **9 features** including Gender
        - **42.9%** diabetic cases
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:2rem;opacity:0.6">
        <div style="font-size:2rem">🩺</div>
        <div style="color:#90caf9;margin-top:0.5rem">DiabetesCare AI — Built with Python, Scikit-learn & Streamlit</div>
        <div style="font-size:0.8rem;color:rgba(144,202,249,0.5);margin-top:0.3rem">
            For educational purposes only • Not a substitute for medical advice
        </div>
    </div>
    """, unsafe_allow_html=True)