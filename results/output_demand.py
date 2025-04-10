# predict_demand_cli.py
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('lgbm_model.pkl')
print("✅ Model loaded.")

# Define required features (must match training)
features = ['Price', 'IsWeekend', 'Year', 'Week', 'IsHolidaySeason', 'CountryCode',
            'CategoryCluster', 'Quantity_lag_1', 'Quantity_roll_mean_2', 'Quantity_roll_mean_4']

# Prompt user for inputs
print("🧾 Enter values for prediction:")

def get_input(prompt, cast_type):
    while True:
        try:
            return cast_type(input(prompt))
        except ValueError:
            print("❌ Invalid input. Please try again.")

input_data = {
    'Price': get_input("Price: ", float),
    'IsWeekend': get_input("Number of transactions during the weekend: ", int),
    'Year': get_input("Year (e.g., 2010): ", int),
    'Week': get_input("Week number (1–52): ", int),
    'IsHolidaySeason': get_input("Holiday season? (0 = No, 1 = Yes): ", int),
    'CountryCode': get_input("Country code (numeric): ", int),
    'CategoryCluster': get_input("Product cluster (0–19): ", int),
    'Quantity_lag_1': get_input("Quantity last week: ", float),
    'Quantity_roll_mean_2': get_input("Rolling mean (2 weeks): ", float),
    'Quantity_roll_mean_4': get_input("Rolling mean (4 weeks): ", float)
}

# Convert to DataFrame
X_input = pd.DataFrame([input_data])

# Predict log quantity and inverse-transform
log_pred = model.predict(X_input)[0]
quantity_pred = np.expm1(log_pred)

print(f"\n📦 Predicted Weekly Demand: {quantity_pred:.2f} units")