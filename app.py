
import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f4f7fb;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #123c69, #1e88e5);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #eaf4ff;
    font-size: 16px;
}

/* Section heading */
.section-title {
    color: #123c69;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Input card */
.input-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e1e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

/* Prediction result */
.result-card {
    background: linear-gradient(120deg, #123c69, #1976d2);
    padding: 28px;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-top: 20px;
}

.result-card h2 {
    color: white;
    font-size: 20px;
}

.result-card h1 {
    color: white;
    font-size: 38px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #1565c0, #2196f3);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #0d47a1;
    color: white;
    border: none;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/LinearRegressionModel.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Car_data.csv")


# Check files
model_path = "models/LinearRegressionModel.pkl"
data_path = "Cleaned_Car_data.csv"

if not os.path.exists(model_path) or not os.path.exists(data_path):
    st.error("Model or dataset file not found. Please check your project folder.")
    st.stop()

model = load_model()
car = load_data()

# ---------------- PREPARE DATA ----------------
car["year"] = pd.to_numeric(car["year"], errors="coerce")
car["kms_driven"] = pd.to_numeric(car["kms_driven"], errors="coerce")
car["Price"] = pd.to_numeric(car["Price"], errors="coerce")
car = car.dropna(subset=[
    "name", "company", "year", "kms_driven", "fuel_type", "Price"
])

companies = sorted(car["company"].astype(str).unique())
fuel_types = sorted(car["fuel_type"].astype(str).unique())


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🚗 Car Price Prediction System</h1>
    <p>Machine Learning Based Used Car Price Estimation</p>
    <p>Enter your car details below and get an estimated selling price.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.5, 1], gap="large")

with left:

    st.markdown('<div class="section-title">🚘 Enter Car Details</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    company = st.selectbox(
        "Select Car Company",
        companies
    )

    company_cars = sorted(
        car[car["company"].astype(str) == company]["name"].astype(str).unique()
    )

    car_name = st.selectbox(
        "Select Car Model",
        company_cars
    )

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Manufacturing Year",
            min_value=int(car["year"].min()),
            max_value=int(car["year"].max()),
            value=int(car["year"].max()),
            step=1
        )

    with col2:
        kms = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=30000,
            step=1000
        )

    fuel = st.selectbox(
        "Select Fuel Type",
        fuel_types
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    predict_button = st.button(
        "🔍 Predict Car Price",
        use_container_width=True
    )


# ---------------- RIGHT INFORMATION SECTION ----------------
with right:

    st.markdown('<div class="section-title">📋 Selected Car Details</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="input-card">
        <h3 style="color:#123c69;">Your Car Information</h3>
        <p style="color:#64748b;">
        Enter the details on the left side to estimate the price
        of your used car.
        </p>
        <hr>
        <p>🚗 <b>Company:</b> Selected above</p>
        <p>⚙️ <b>Model:</b> Selected above</p>
        <p>📅 <b>Year:</b> Manufacturing year</p>
        <p>🛣️ <b>Distance:</b> Kilometers driven</p>
        <p>⛽ <b>Fuel:</b> Petrol / Diesel / Other</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if predict_button:

    input_data = pd.DataFrame(
        [[car_name, company, year, kms, fuel]],
        columns=[
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type"
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">💰 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-card">
            <h2>Estimated Car Price</h2>
            <h1>₹ {prediction:,.0f}</h1>
            <p>Based on your selected car details</p>
        </div>
        """, unsafe_allow_html=True)

        st.success("Car price predicted successfully!")

    except Exception as e:
        st.error(f"Prediction error: {e}")


# ---------------- ABOUT PROJECT ----------------
st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.markdown("""
    ### Car Price Prediction Using Machine Learning

    This project predicts the estimated price of a used car
    based on its details.

    **Machine Learning Algorithm:** Linear Regression

    **Technologies Used:**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Joblib

    **Input Features:**
    - Car Name
    - Company
    - Manufacturing Year
    - Kilometers Driven
    - Fuel Type
    """)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Car Price Prediction System | Developed using Python & Machine Learning
</div>
""", unsafe_allow_html=True)
