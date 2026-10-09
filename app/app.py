
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model_v1.pkl"

st.set_page_config(
    page_title="ChurnSense | Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PREMIUM UI THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b1120;
    color: #e5edf7;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(11,17,32,.96);
}
.block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 3rem;
}
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #263449;
}
[data-testid="stSidebar"] * {
    color: #dce7f5;
}
.brand {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.8px;
}
.brand span { color: #2dd4bf; }
.eyebrow {
    color: #2dd4bf;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .75rem;
    font-weight: 800;
}
.hero {
    background: linear-gradient(115deg,#14253b,#102b36);
    border: 1px solid #294456;
    border-radius: 20px;
    padding: 30px 32px;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 2.35rem;
    line-height: 1.2;
    color: #f8fafc;
    margin: 8px 0;
}
.hero p {
    color: #a9bdce;
    margin-bottom: 0;
}
h1, h2, h3 {
    color: #f1f5f9 !important;
    letter-spacing: -.5px;
}
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.08rem !important; }
p, label, [data-testid="stCaptionContainer"] {
    color: #a9b8cc;
}
.section-label {
    color: #2dd4bf;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 800;
    margin-bottom: 5px;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#17243a,#111c2e);
    border: 1px solid #2b3b52;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
}
[data-testid="stMetricLabel"] {
    color: #a9b8cc !important;
    font-size: .86rem;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.8rem;
    font-weight: 800;
}
[data-testid="stPlotlyChart"] {
    background: #111c2e;
    border: 1px solid #263449;
    border-radius: 15px;
    padding: 10px;
}
[data-testid="stForm"] {
    background: #111c2e;
    border: 1px solid #2b3b52;
    border-radius: 16px;
    padding: 24px;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #17243a;
    border-color: #34465f;
    border-radius: 9px;
}
.stButton > button, .stFormSubmitButton > button {
    background: #14b8a6;
    color: #071521;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 800;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #2dd4bf;
    color: #071521;
    border: none;
}
[data-testid="stRadio"] label {
    background: #17243a;
    border-radius: 9px;
    padding: 8px 10px;
    margin-bottom: 5px;
}
hr { border-color: #263449; }
div[data-testid="stAlert"] { border-radius: 12px; }
.footer {
    text-align: center;
    color: #64748b;
    font-size: .8rem;
    padding-top: 30px;
}
@media (max-width: 768px) {
    .block-container { padding: 1rem; }
    .hero h1 { font-size: 1.7rem; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA + MODEL
# =========================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    ).fillna(0)
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Training a new model on first run."
        )
    return joblib.load(MODEL_PATH)


def ensure_model_exists():
    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    st.warning("Missing model artifact detected. Training a fresh model ...")
    try:
        import run_churn_pipeline

        run_churn_pipeline.main()
    except Exception as exc:
        raise RuntimeError(f"Could not train churn model: {exc}") from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model was not created at {MODEL_PATH}.")


try:
    df = load_data()
    ensure_model_exists()
    model = load_model()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.info(
        "Check that the dataset exists in the project root and "
        "the saved model exists in the models folder."
    )
    st.stop()


# =========================================================
# SHARED HELPERS
# =========================================================
def chart_layout(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#b8c7d9", size=12),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def make_features(customer):
    X = pd.DataFrame([customer])
    X["tenure_group"] = pd.cut(
        X["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )
    return X


def header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 38, "color": "#f8fafc"}},
        title={"text": "Estimated churn probability",
               "font": {"size": 15, "color": "#b8c7d9"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "ticksuffix": "%",
                "tickcolor": "#94a3b8",
            },
            "bar": {"color": "#2dd4bf", "thickness": .24},
            "bgcolor": "#111c2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#173b3c"},
                {"range": [30, 60], "color": "#4a3c29"},
                {"range": [60, 100], "color": "#482b3b"},
            ],
        },
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5edf7"},
        margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">📡 Churn<span>Sense</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Executive Overview", "Prediction Studio", "Model Intelligence"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.caption("● Model loaded")
    st.caption("● Dataset loaded")
    st.caption("● Prediction engine ready")
    st.divider()
    st.caption("ChurnSense · ML-powered retention")


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":
    header(
        "CUSTOMER INTELLIGENCE / OVERVIEW",
        "Know your customers. Keep your customers.",
        "Explore churn patterns and discover where retention efforts "
        "may make the biggest difference.",
    )

    total_customers = len(df)
    churn_count = int((df["Churn"] == "Yes").sum())
    retained = total_customers - churn_count
    churn_rate = churn_count / total_customers * 100
    avg_monthly = df["MonthlyCharges"].mean()

    st.markdown('<div class="section-label">Portfolio at a glance</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total customers", f"{total_customers:,}")
    k2.metric("Churned customers", f"{churn_count:,}")
    k3.metric("Overall churn rate", f"{churn_rate:.1f}%")
    k4.metric("Avg. monthly charge", f"${avg_monthly:.2f}")

    st.write("")
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("### Customer retention mix")
        st.caption("Share of customers who stayed vs. churned")

        pie_data = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [retained, churn_count],
        })
        fig = px.pie(
            pie_data,
            names="Status",
            values="Customers",
            hole=.72,
            color="Status",
            color_discrete_map={
                "Retained": "#2dd4bf",
                "Churned": "#fb7185",
            },
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent",
            marker=dict(line=dict(color="#111c2e", width=4)),
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(chart_layout(fig, 340),
                        use_container_width=True)

    with right:
        st.markdown("### Churn by contract")
        st.caption("Percentage of customers leaving in each contract group")

        contract_data = (
            df.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="Churn rate")
            .sort_values("Churn rate", ascending=True)
        )
        fig = px.bar(
            contract_data,
            x="Churn rate",
            y="Contract",
            orientation="h",
            text=contract_data["Churn rate"].map(lambda x: f"{x:.1f}%"),
            color="Churn rate",
            color_continuous_scale=["#2dd4bf", "#60a5fa", "#fb7185"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            xaxis_title="Churn rate (%)",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(chart_layout(fig, 340),
                        use_container_width=True)

    st.markdown("### Monthly charges & customer churn")
    st.caption("Compare monthly-charge distributions for both customer groups")

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        barmode="overlay",
        opacity=.72,
        nbins=35,
        color_discrete_map={
            "Yes": "#fb7185",
            "No": "#2dd4bf",
        },
        labels={"Churn": "Customer outcome"},
    )
    fig.update_layout(
        xaxis_title="Monthly charges ($)",
        yaxis_title="Number of customers",
    )
    st.plotly_chart(chart_layout(fig, 350),
                    use_container_width=True)

    st.markdown("### Retention signals")

    signals = st.columns(3)
    signals[0].markdown("**01 · Contract commitment**")
    signals[0].write(
        "Month-to-month contracts have the highest churn rate in this dataset."
    )
    signals[1].markdown("**02 · Customer experience**")
    signals[1].write(
        "Review support and security services when investigating churn."
    )
    signals[2].markdown("**03 · Proactive outreach**")
    signals[2].write(
        "Use model scores to prioritize review—not to guarantee outcomes."
    )

    st.caption(
        "Dataset-level findings are descriptive and do not establish causation."
    )


# =========================================================
# PREDICTION STUDIO
# =========================================================
elif page == "Prediction Studio":
    header(
        "PREDICTIVE ANALYTICS / CUSTOMER RISK",
        "Prediction Studio",
        "Create a customer profile and estimate their likelihood of churn.",
    )

    st.info(
        "Enter the customer's service and billing details. "
        "The saved model will return a probability and class prediction."
    )

    with st.form("prediction_form"):
        st.markdown("### 01 — Customer profile")
        a, b, c = st.columns(3)

        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])

        with b:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract type",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with c:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            monthly = st.number_input(
                "Monthly charges ($)", min_value=0.0,
                max_value=200.0, value=70.0, step=1.0,
            )
            total = st.number_input(
                "Total charges ($)", min_value=0.0,
                max_value=100000.0, value=840.0, step=10.0,
            )

        st.divider()
        st.markdown("### 02 — Services")

        d, e, f = st.columns(3)

        with d:
            security = st.selectbox(
                "Online security",
                ["Yes", "No", "No internet service"],
            )
            backup = st.selectbox(
                "Online backup",
                ["Yes", "No", "No internet service"],
            )

        with e:
            protection = st.selectbox(
                "Device protection",
                ["Yes", "No", "No internet service"],
            )
            tech = st.selectbox(
                "Tech support",
                ["Yes", "No", "No internet service"],
            )

        with f:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                ["Yes", "No", "No internet service"],
            )

        submitted = st.form_submit_button(
            "⚡  Analyze customer risk",
            use_container_width=True,
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        X_customer = make_features(customer)

        try:
            probability = float(model.predict_proba(X_customer)[0, 1])
            prediction = int(model.predict(X_customer)[0])

            # Illustrative risk bands; model threshold remains unchanged.
            risk = (
                "High" if probability >= .60
                else "Medium" if probability >= .30
                else "Low"
            )

            st.divider()
            st.markdown("## Prediction results")

            result_left, result_right = st.columns([1, 1.2], gap="large")

            with result_left:
                st.plotly_chart(
                    risk_gauge(probability),
                    use_container_width=True,
                )

            with result_right:
                st.markdown("### Customer risk assessment")

                st.metric(
                    "Predicted class",
                    "Likely to churn" if prediction == 1 else "Likely to stay",
                )
                st.metric("Risk category", risk)
                st.progress(probability)
                st.caption(
                    f"Estimated churn probability: {probability:.1%}"
                )

                if prediction == 1:
                    st.warning(
                        "Consider reviewing this customer's experience, "
                        "billing concerns and retention options."
                    )
                else:
                    st.success(
                        "The model predicts this customer will stay. "
                        "Continue delivering reliable service."
                    )

            st.caption(
                "Probability is an estimate, not a guarantee. Risk bands "
                "are illustrative and are separate from the model's "
                "classification threshold."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =========================================================
# MODEL INTELLIGENCE
# =========================================================
else:
    header(
        "MACHINE LEARNING / EXPLAINABILITY",
        "Model Intelligence",
        "Inspect the saved classifier, feature importance and evaluation results.",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Test ROC AUC", "0.8453")
    m2.metric("Algorithm", "Gradient Boosting")
    m3.metric("Tuning", "RandomizedSearchCV")

    st.caption(
        "The AUC shown is from the recorded evaluation run; "
        "it is not classification accuracy."
    )

    st.markdown("### Features the model relies on")

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(15)

        fig = px.bar(
            importance_df.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#1caaa0", "#60a5fa", "#a78bfa"],
        )
        fig.update_layout(
            xaxis_title="Relative feature importance",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(chart_layout(fig, 500),
                        use_container_width=True)

    except Exception as exc:
        st.warning(f"Could not display feature importance: {exc}")

    st.markdown("### Evaluation snapshot")

    st.write(
        "- **Test ROC AUC:** 0.8453\n"
        "- **Validation strategy:** Stratified train/test split\n"
        "- **Model tuning:** RandomizedSearchCV\n"
        "- **Preprocessing:** StandardScaler and OneHotEncoder\n"
        "- **Feature engineering:** Tenure groups\n"
        "- **Interpretability:** SHAP analysis was performed during evaluation"
    )

    st.warning(
        "Feature importance describes how the fitted model uses features. "
        "It does not prove causation or guarantee that changing a feature "
        "will change a customer's outcome."
    )

st.markdown(
    '<div class="footer">CHURNSENSE • CUSTOMER RETENTION INTELLIGENCE • ML PROJECT</div>',
    unsafe_allow_html=True,
)
