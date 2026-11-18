## Car Price Prediction

# Car Price Prediction Using Machine Learning

## Project Overview
This project predicts the estimated selling price of used cars based on different features using Machine Learning.

A user can enter car details through an interactive Streamlit web application and get an estimated car price.

## Features
- Select car company and model.
- Enter manufacturing year.
- Enter kilometers driven.
- Select fuel type.
- Predict estimated car price.
- Interactive web application using Streamlit.

## Machine Learning Algorithm
- Linear Regression

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Structure
- `app.py` – Streamlit web application.
- `car_price_ML.ipynb` – Machine Learning implementation.
- `car_price_dataset.csv` – Original dataset.
- `Cleaned_Car_data.csv` – Cleaned dataset.
- `models/` – Trained ML model.
- `requirements.txt` – Required Python libraries.

## How to Run the Project

1. Clone this repository.
2. Install the required libraries:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

## Output
The application displays an estimated used car price based on the entered car details.

## Disclaimer
Predictions are educational estimates based on the available historical dataset and may not reflect current market prices.
