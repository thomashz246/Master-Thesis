# demand.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation
import time
import joblib
import math

# Load engineered data with clusters
print("\U0001F4C2 Loading engineered dataset...")
df = pd.read_csv("../data/engineered_weekly_demand_with_lags.csv")
print(f"Dataset loaded with {len(df)} rows")

# Handle outliers
def remove_outliers(df, col='Quantity', threshold=3):
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    return df[z_scores < threshold]

# Handle missing values more intelligently
def handle_missing(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col.startswith('Quantity_lag') or col.startswith('Quantity_roll'):
                # Use appropriate filling for time-series features
                df[col] = df[col].fillna(df['Quantity'].mean())
            else:
                df[col] = df[col].fillna(df[col].median())
    return df

# Apply preprocessing
df_clean = remove_outliers(df)
df_clean = handle_missing(df_clean)

# Log-transform the target to reduce skew
print("🔄 Applying log-transform to Quantity...")
df_clean['LogQuantity'] = np.log1p(df_clean['Quantity'])

# Encode categorical variables
for col in ['StockCode']:
    df_clean[col] = df_clean[col].astype('category')

# Print all columns
print("📋 Columns in dataset:"
      f"\n{df_clean.columns.tolist()}")

print("🛠️ Engineering additional features...")

# Price-related features
df_clean['Price_log'] = np.log1p(df_clean['Price'])
df_clean['Price_squared'] = df_clean['Price'] ** 2

# Seasonal decomposition
df_clean['SinWeek'] = np.sin(2 * np.pi * df_clean['Week'] / 52)
df_clean['CosWeek'] = np.cos(2 * np.pi * df_clean['Week'] / 52)

# Monthly seasonality approximation
df_clean['MonthApprox'] = ((df_clean['Week'] - 1) // 4) + 1
df_clean['SinMonth'] = np.sin(2 * np.pi * df_clean['MonthApprox'] / 12)
df_clean['CosMonth'] = np.cos(2 * np.pi * df_clean['MonthApprox'] / 12)

# Interaction terms
df_clean['Price_Holiday'] = df_clean['Price'] * df_clean['IsHolidaySeason']
df_clean['Price_Category'] = df_clean['Price'] * df_clean['CategoryCluster']

# Trend indicators
df_clean['Trend'] = df_clean['Quantity_roll_mean_4'] - df_clean['Quantity_roll_mean_2']
df_clean['Acceleration'] = df_clean['Quantity_lag_1'] - df_clean['Quantity_roll_mean_2']
df_clean['Volatility'] = df_clean.groupby(['CategoryCluster'])['Quantity'].transform(lambda x: x.rolling(4).std())
df_clean['Volatility'].fillna(df_clean['Volatility'].median(), inplace=True)

# Price position metrics
df_clean['Price_vs_Category_Avg'] = df_clean['Price'] / df_clean.groupby('CategoryCluster')['Price'].transform('mean')

feature_cols = ['Price', 
                'IsWeekend', 
                'Year', 
                'Week', 
                'IsHolidaySeason', 
                'CountryCode', 
                'CategoryCluster', 
                'Quantity_lag_1', 
                'Quantity_roll_mean_2', 
                'Quantity_roll_mean_4']

# Enhance the feature columns list
extended_features = feature_cols + [
    'Price_log', 'Price_squared', 
    'SinWeek', 'CosWeek', 'SinMonth', 'CosMonth',
    'Price_Holiday', 'Price_Category', 
    'Trend', 'Acceleration', 'Volatility',
    'Price_vs_Category_Avg'
]

print(f"✅ Added {len(extended_features) - len(feature_cols)} new features")
print(f"📋 New feature set: {extended_features}")

X = df_clean[extended_features]
y = df_clean['LogQuantity']

print(f"🧪 Using full dataset of {len(X)} samples")

# Split data
print("✂️ Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(f"📚 Train: {len(X_train)}, Test: {len(X_test)}")

# Train LightGBM model
print("🚀 Training LightGBM model...")
model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=2048,
    learning_rate=0.03,
    num_leaves=256,
    max_bin=512,
    n_jobs=4,
    verbose=-1
)

start_time = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=10)
    ]
)

end_time = time.time()
print(f"✅ Training finished in {end_time - start_time:.2f} seconds")

# Save model
joblib.dump(model, '../models/lgbm_model.pkl')
print("📂 Model saved to lgbm_model.pkl")

# Evaluate model
print("📈 Evaluating model...")
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # inverse transform
y_test_exp = np.expm1(y_test)

rmse = math.sqrt(mean_squared_error(y_test_exp, y_pred))
mae = mean_absolute_error(y_test_exp, y_pred)
r2 = r2_score(y_test_exp, y_pred)

print(f"📉 RMSE: {rmse:.2f}")
print(f"🔏 MAE: {mae:.2f}")
print(f"🌟 R²: {r2:.2f}")

# Plot feature importances
print("📊 Plotting feature importances...")
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature Importances (Log Demand)")
plt.tight_layout()
plt.savefig("feature_importances_log.png")
print("📁 Feature importances saved to feature_importances_log.png")

# Price elasticity analysis
print("\n📊 Analyzing price elasticity...")
price_col_idx = np.where(X.columns == 'Price')[0][0]
price_multipliers = np.linspace(0.3, 2.5, 50)
baseline_predictions = np.expm1(model.predict(X_test))

results = []
for multiplier in price_multipliers:
    X_mod = X_test.copy()
    X_mod.iloc[:, price_col_idx] *= multiplier
    new_pred_log = model.predict(X_mod)
    new_pred = np.expm1(new_pred_log)
    avg_change = (new_pred.mean() - baseline_predictions.mean()) / baseline_predictions.mean()
    results.append({
        'Price_Multiplier': multiplier,
        'Pred_Quantity': new_pred.mean(),
        'Quantity_Change_%': avg_change * 100
    })

elasticity_df = pd.DataFrame(results)
print(elasticity_df)

# Elasticity calc
try:
    p_up = elasticity_df['Price_Multiplier'].max()
    p_down = elasticity_df['Price_Multiplier'].min()
    q_up = elasticity_df.iloc[(elasticity_df['Price_Multiplier'] - p_up).abs().argmin()]['Pred_Quantity']
    q_down = elasticity_df.iloc[(elasticity_df['Price_Multiplier'] - p_down).abs().argmin()]['Pred_Quantity']
    q_base = elasticity_df.iloc[(elasticity_df['Price_Multiplier'] - 1.0).abs().argmin()]['Pred_Quantity']

    elasticity = ((q_up - q_down) / q_base) / ((p_up - p_down) / 1.0)

    print(f"\n📉 Estimated Price Elasticity: {elasticity:.3f}")
    if elasticity < 0:
        if abs(elasticity) > 1:
            print("📈 Demand is ELASTIC - quantity changes more than price")
            print(f"💰 Revenue-maximizing price change: {(-1/(1 + elasticity) - 1)*100:.1f}%")
        else:
            print("📉 Demand is INELASTIC - quantity less sensitive to price")
    else:
        print("⚠️ Unusual elasticity (positive) - check your model")
except Exception as e:
    print("⚠️ Could not calculate elasticity:", e)

# Smooth the predicted quantity with a rolling average
elasticity_df['Smoothed_Quantity'] = elasticity_df['Pred_Quantity'].rolling(window=5, center=True).mean()

# Plot smoothed and raw curve
plt.figure(figsize=(10, 6))
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Pred_Quantity'], marker='o', label='Raw')
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Smoothed_Quantity'], color='orange', linewidth=2, label='Smoothed (rolling mean)')
plt.axvline(x=1.0, color='r', linestyle='--', label='Current Price')
plt.xlabel('Price Multiplier')
plt.ylabel('Predicted Quantity')
plt.title('Price Sensitivity Analysis (Smoothed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("price_sensitivity_smoothed.png")
print("📁 Smoothed price sensitivity analysis saved to price_sensitivity_smoothed.png")

# Predicted quantity by week
print("\n🕰️ Plotting predicted weekly quantity trend...")
df_clean['PredictedQuantity'] = np.expm1(model.predict(X))
weekly_avg = df_clean.groupby(['Year', 'Week'])['PredictedQuantity'].mean().reset_index()
weekly_avg['YearWeek'] = weekly_avg['Year'].astype(str) + "-W" + weekly_avg['Week'].astype(str)
weekly_avg = weekly_avg.sort_values(['Year', 'Week'])

print("\n Summing predicted demand across weeks...")
df_clean['Predicted_LogQuantity'] = model.predict(X)
df_clean['Predicted_Quantity'] = np.expm1(df_clean['Predicted_LogQuantity'])

weekly_sum = df_clean.groupby('Week')['Predicted_Quantity'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(weekly_sum['Week'], weekly_sum['Predicted_Quantity'], marker='o')
plt.title('Total Predicted Demand per Week (Aggregated Over All Years)')
plt.xlabel('Week Number')
plt.ylabel('Summed Predicted Quantity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("predicted_demand_summed_by_week.png")
print("📁 Weekly demand plot saved to predicted_demand_summed_by_week.png")
