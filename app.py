
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import date

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Bangalore Traffic Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111c30;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #60a5fa;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 30px;
}

.result-card {
    background: linear-gradient(135deg, #172554, #1e3a8a);
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #3b82f6;
    margin-top: 20px;
}

.result-value {
    font-size: 42px;
    font-weight: bold;
    color: #93c5fd;
}

.section-title {
    color: #60a5fa;
    font-size: 22px;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD MODEL AND DATASET
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "bangalore_traffic_model.pkl"
DATA_PATH = BASE_DIR / "Banglore_traffic_Dataset.csv"

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_data()

except FileNotFoundError as e:

    st.error(f"Required file not found: {e}")
    st.stop()

except Exception as e:

    st.error(f"Error loading project files: {e}")
    st.stop()


# --------------------------------
# FEATURE CONFIGURATION
# --------------------------------

feature_columns = list(model.feature_names_in_)

date_features = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek"
]

categorical_features = [
    col for col in feature_columns
    if col in df.columns
    and not pd.api.types.is_numeric_dtype(df[col])
]

numeric_features = [
    col for col in feature_columns
    if col in df.columns
    and col not in date_features
    and pd.api.types.is_numeric_dtype(df[col])
]

# Convert dataset date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Default date
valid_dates = df["Date"].dropna()

if len(valid_dates):
    default_date = valid_dates.sort_values().iloc[len(valid_dates)//2].date()
else:
    default_date = date.today()


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("🚦 Traffic Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader("About Project")

st.sidebar.write(
    """
    This Machine Learning application predicts
    Bangalore traffic congestion using a
    tuned Decision Tree Regressor.

    **Algorithms studied:**

    - KNN
    - Decision Tree
    - Linear Regression

    **Selected model:**
    Tuned Decision Tree
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.metric(
    "Testing R² Score",
    "96.66%"
)

st.sidebar.metric(
    "Mean Absolute Error",
    "2.82"
)

st.sidebar.caption(
    "Notebook results, pending final leakage verification."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed using Python, Scikit-learn and Streamlit."
)


# --------------------------------
# MAIN HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">🚦 Bangalore Traffic Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Smart Machine Learning Based Traffic Congestion Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# DASHBOARD METRICS
# --------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📊 Dataset Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "🧠 ML Algorithm",
        "Decision Tree"
    )

with col3:
    st.metric(
        "🎯 Model R²",
        "96.66%"
    )

st.divider()


# --------------------------------
# INPUT FORM
# --------------------------------

st.markdown(
    '<p class="section-title">📍 Enter Traffic Details</p>',
    unsafe_allow_html=True
)

st.write(
    "Provide road, traffic, weather and date information "
    "to predict congestion."
)

with st.form("traffic_prediction_form"):

    # DATE
    st.subheader("📅 Date Information")

    selected_date = st.date_input(
        "Select Date",
        value=default_date
    )

    st.divider()

    # CATEGORICAL FEATURES
    st.subheader("📍 Location and Conditions")
    input_values = {}
    categorical_columns = st.columns(2)
    for index, col in enumerate(categorical_features):
        options = sorted(df[col].dropna().unique())
        if not options:
            st.error(f"No valid options found for {col}.")
            st.stop()
        with categorical_columns[index % 2]:
            input_values[col] = st.selectbox(col, options)

    st.divider()

    # TRAFFIC INFORMATION
    st.subheader("🚗 Traffic Information")

    traffic_col1, traffic_col2 = st.columns(2)

    with traffic_col1:

        for col in numeric_features[:len(numeric_features)//2]:

            if col not in df.columns:
                continue

            minimum = float(df[col].min())
            maximum = float(df[col].max())
            default = float(df[col].median())

            if minimum == maximum:
                maximum = minimum + 1

            step = 1.0 if (
                pd.api.types.is_integer_dtype(df[col])
            ) else 0.1

            input_values[col] = st.number_input(
                col,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step
            )

    with traffic_col2:

        remaining_features = numeric_features[
            len(numeric_features)//2:
        ]

        for col in remaining_features:

            if col not in df.columns:
                continue

            minimum = float(df[col].min())
            maximum = float(df[col].max())
            default = float(df[col].median())

            if minimum == maximum:
                maximum = minimum + 1

            step = 1.0 if (
                pd.api.types.is_integer_dtype(df[col])
            ) else 0.1

            input_values[col] = st.number_input(
                col,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step
            )

    st.divider()

    # DATE FEATURE ENGINEERING
    input_values["Year"] = selected_date.year
    input_values["Month"] = selected_date.month
    input_values["Day"] = selected_date.day
    input_values["DayOfWeek"] = selected_date.weekday()

    # PREDICTION BUTTON
    st.markdown("---")

    predict_button = st.form_submit_button(
        "🚀 Predict Traffic Congestion"
    )


# --------------------------------
# PREDICTION
# --------------------------------

if predict_button:

    try:

        # Arrange columns in training order
        input_df = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        # Make prediction
        prediction = model.predict(input_df)[0]

        st.divider()

        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:20px;color:#bfdbfe;">
                    Predicted Congestion Level
                </div>
                <div class="result-value">
                    {prediction:.2f}
                </div>
                <div style="color:#bfdbfe;">
                    Model output
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("Traffic prediction completed successfully!")

        # INPUT SUMMARY
        st.subheader("📋 Input Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.write("**Location:**", input_values.get("Area Name", "N/A"))
            st.write(
                "**Road:**",
                input_values.get("Road/Intersection Name", "N/A")
            )
            st.write(
                "**Weather:**",
                input_values.get("Weather Conditions", "N/A")
            )

        with summary_col2:
            st.write("**Date:**", selected_date)
            st.write("**Traffic Volume:**",
                     input_values.get("Traffic Volume", "N/A"))

        with st.expander("View all model input values"):
            st.dataframe(input_df, width="stretch")

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
        Bangalore Traffic Prediction Project |
        Machine Learning | Python | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
