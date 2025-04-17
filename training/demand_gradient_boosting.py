"""
Gradient Boosting implementation for demand prediction in the Online Retail dataset.
This implementation follows similar methodology as the LightGBM version.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import time
import joblib
import math
import warnings
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
warnings.filterwarnings("ignore", category=FutureWarning)

# Create image directory
os.makedirs("images", exist_ok=True)
os.makedirs("product_elasticity", exist_ok=True)

# Load engineered data with clusters
print("📂 Loading engineered dataset...")
df = pd.read_csv("../data/engineered_weekly_demand_with_lags.csv")
print(f"Dataset loaded with {len(df)} rows")

# Handle outliers
def remove_outliers(df, col='Quantity', threshold=3):
    """Remove outliers using z-score method."""
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    return df[z_scores < threshold]

# Handle missing values more intelligently
def handle_missing(df):
    """Handle missing values in a way appropriate for time series data."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col.startswith('Quantity_lag') or col.startswith('Quantity_roll'):
                df[col] = df[col].fillna(df['Quantity'].mean())
            else:
                df[col] = df[col].fillna(df[col].median())
    return df

# Apply preprocessing
df_clean = remove_outliers(df).copy()
df_clean = handle_missing(df_clean)

# Log-transform the target to reduce skew
print("🔄 Applying log-transform to Quantity...")
df_clean.loc[:, 'LogQuantity'] = np.log1p(df_clean['Quantity'])

# Encode categorical variables
for col in ['StockCode']:
    df_clean[col] = df_clean[col].astype('category')

# Print all columns
print("📋 Columns in dataset:"
      f"\n{df_clean.columns.tolist()}")

print("🛠️ Engineering additional features...")

# Feature engineering function
def apply_feature_engineering(df, train_only=None):
    """
    Apply feature engineering with optional prevention of data leakage.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The data to transform
    train_only : pandas DataFrame, optional
        If provided, compute group statistics only from this data
        and apply to df
    """
    # Create explicit copy
    result = df.copy()
    
    # If train_only is provided, use it for group statistics
    source_for_stats = train_only if train_only is not None else df
    
    # Apply all transformations using .loc
    result.loc[:, 'Price_log'] = np.log1p(result['Price'])
    result.loc[:, 'Price_squared'] = result['Price'] ** 2
    
    result.loc[:, 'SinWeek'] = np.sin(2 * np.pi * result['Week'] / 52)
    result.loc[:, 'CosWeek'] = np.cos(2 * np.pi * result['Week'] / 52)
    
    result.loc[:, 'MonthApprox'] = ((result['Week'] - 1) // 4) + 1
    result.loc[:, 'SinMonth'] = np.sin(2 * np.pi * result['MonthApprox'] / 12)
    result.loc[:, 'CosMonth'] = np.cos(2 * np.pi * result['MonthApprox'] / 12)
    
    result.loc[:, 'Price_Holiday'] = result['Price'] * result['IsHolidaySeason']
    result.loc[:, 'Price_Category'] = result['Price'] * result['CategoryCluster']
    
    result.loc[:, 'Trend'] = result['Quantity_roll_mean_4'] - result['Quantity_roll_mean_2']
    result.loc[:, 'Acceleration'] = result['Quantity_lag_1'] - result['Quantity_roll_mean_2']
    
    # Group operations with data leakage prevention
    # Calculate statistics from source_for_stats
    cat_volatility = source_for_stats.groupby(['CategoryCluster'])['Quantity_lag_1'].rolling(4).std().reset_index(level=0, drop=True)
    cat_price_avg = source_for_stats.groupby('CategoryCluster')['Price'].mean()
    cat_week_year_price = source_for_stats.groupby(['CategoryCluster', 'Week', 'Year'])['Price'].mean()
    
    # Map these values to result
    # For volatility, we'll use a different approach since it's more complex
    if train_only is not None:
        # Create mapping dictionary for category volatility
        cat_vol_dict = {}
        for cat in source_for_stats['CategoryCluster'].unique():
            cat_vol_dict[cat] = source_for_stats[source_for_stats['CategoryCluster'] == cat]['Quantity_lag_1'].rolling(4).std().mean()
        
        # Apply mapping
        result.loc[:, 'Volatility'] = result['CategoryCluster'].map(cat_vol_dict)
        result.loc[:, 'Price_vs_Category_Avg'] = result['Price'] / result['CategoryCluster'].map(cat_price_avg)
        
        # For CategoryPrice_ratio we need to create a lookup function
        def get_cat_week_year_price(row):
            try:
                return cat_week_year_price.loc[(row['CategoryCluster'], row['Week'], row['Year'])]
            except KeyError:
                # Fallback to category average if specific combination not found
                try:
                    return cat_price_avg[row['CategoryCluster']]
                except KeyError:
                    return result['Price'].mean()
                
        result.loc[:, 'CategoryPrice_ratio'] = result.apply(
            lambda row: row['Price'] / get_cat_week_year_price(row), axis=1
        )
    else:
        # Original code when using the whole dataset
        volatility = result.groupby(['CategoryCluster'])['Quantity_lag_1'].transform(lambda x: x.rolling(4).std())
        result.loc[:, 'Volatility'] = volatility
        
        price_vs_category = result['Price'] / result.groupby('CategoryCluster')['Price'].transform('mean')
        result.loc[:, 'Price_vs_Category_Avg'] = price_vs_category
        
        result.loc[:, 'CategoryPrice_ratio'] = result['Price'] / result.groupby(
            ['CategoryCluster', 'Week', 'Year'])['Price'].transform('mean')
    
    # Fill any NaN values
    result['Volatility'] = result['Volatility'].fillna(result['Volatility'].median())
    
    return result

# Apply feature engineering
df_clean = apply_feature_engineering(df_clean)

# Calculate the absolute week number for each row (Year*52 + Week)
df_clean['AbsoluteWeek'] = df_clean['Year']*52 + df_clean['Week']

# Find the first appearance week for each product
first_appearance = df_clean.groupby('StockCode')['AbsoluteWeek'].min()

# Calculate product age in weeks
df_clean['ProductAge'] = df_clean['AbsoluteWeek'] - df_clean['StockCode'].map(first_appearance)

# Drop the temporary column
df_clean = df_clean.drop(columns=['AbsoluteWeek'])

# Capture product popularity - total sales volume per product
df_clean = df_clean.sort_values(['StockCode', 'Year', 'Week'])
df_clean['StockCode_popularity'] = df_clean.groupby('StockCode')['Quantity'].transform(lambda x: x.shift(1).expanding().sum())

# Add lag differences (rate of change)
df_clean['Quantity_diff_1'] = df_clean.groupby('StockCode')['Quantity_lag_1'].diff()
df_clean['Price_diff_1'] = df_clean.groupby('StockCode')['Price'].diff()

# Add more lagged features with different time horizons
for i in [8, 12]:
    df_clean[f'Quantity_roll_mean_{i}'] = df_clean.groupby('StockCode')['Quantity'].transform(
        lambda x: x.rolling(window=i, min_periods=1).mean())

# Interaction between price and season
df_clean['Price_Week_sin'] = df_clean['Price'] * df_clean['SinWeek']
df_clean['Price_Week_cos'] = df_clean['Price'] * df_clean['CosWeek']

# Price elasticity might vary by product popularity
df_clean['Price_Popularity'] = df_clean['Price'] * np.log1p(df_clean['StockCode_popularity'])

# Handle missing values after feature engineering
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

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
    'Price_vs_Category_Avg',
    'ProductAge', 'Quantity_diff_1', 'Price_diff_1',
    'Quantity_roll_mean_8', 'Quantity_roll_mean_12',
    'Price_Week_sin', 'Price_Week_cos', 'Price_Popularity', 'CategoryPrice_ratio'
]

print(f"✅ Added {len(extended_features) - len(feature_cols)} new features")
print(f"📋 New feature set: {extended_features}")

X = df_clean[extended_features]
y = df_clean['LogQuantity']

print(f"🧪 Using full dataset of {len(X)} samples")

# Time series cross-validation
print("\n🔄 Performing time series cross-validation...")
n_splits = 8  # Same as in LightGBM implementation
tscv = TimeSeriesSplit(n_splits=n_splits)

rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n🔁 Fold {fold + 1}/{n_splits}")
    
    # Get the corresponding rows from the original dataframe
    df_train = df_clean.iloc[train_idx]
    df_val = df_clean.iloc[val_idx]
    
    # For training data
    df_train_featured = apply_feature_engineering(df_train)
    
    # For validation data (using train data for group statistics)
    df_val_featured = apply_feature_engineering(df_val, train_only=df_train)
    
    # Extract features and target from featured dataframes
    X_train = df_train_featured[extended_features]
    y_train = df_train_featured['LogQuantity']
    X_val = df_val_featured[extended_features]
    y_val = df_val_featured['LogQuantity']
    
    # Handle any remaining missing values
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_val.median())
    
    # Train Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred_log = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_log)
    y_val_true = np.expm1(y_val)
    
    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(y_val_true, y_val_pred))
    mae = mean_absolute_error(y_val_true, y_val_pred)
    r2 = r2_score(y_val_true, y_val_pred)
    
    print(f"📉 Fold RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

# Print summary of cross-validation results
print("\n📊 Cross-validation results summary:")
print(f"  RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
print(f"  MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
print(f"  R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# Plot cross-validation results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.bar(range(1, n_splits+1), rmse_scores)
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE by Fold')

plt.subplot(1, 3, 2)
plt.bar(range(1, n_splits+1), mae_scores)
plt.xlabel('Fold')
plt.ylabel('MAE')
plt.title('MAE by Fold')

plt.subplot(1, 3, 3)
plt.bar(range(1, n_splits+1), r2_scores)
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('R² by Fold')

plt.tight_layout()
plt.savefig("images/gradboost_cv_results.png")
print("📁 Cross-validation results saved to gradboost_cv_results.png")

# Use predefined hyperparameters (similar approach to the LightGBM implementation)
print("\n🔍 Using pre-defined hyperparameters...")
best_params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'subsample': 0.8
}

print("\n🚀 Training final model on train/test split...")
# Hold out the last 10% of weeks as test set
df_clean = df_clean.sort_values(['Year', 'Week'])
cutoff = int(len(df_clean) * 0.9)
X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
print(f"📚 Train: {len(X_train)}, Test: {len(X_test)}")

# Handle any remaining missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

print("🚀 Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=2048,
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    subsample=best_params['subsample'],
    random_state=42,
    verbose=1
)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"✅ Training finished in {end_time - start_time:.2f} seconds")

# Save model
joblib.dump(model, '../models/gradboost_model.pkl')
print("📂 Model saved to gradboost_model.pkl")

# Evaluate model
print("📈 Evaluating model...")
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
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
features = X_train.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Gradient Boosting Feature Importances")
plt.tight_layout()
plt.savefig("images/gradboost_feature_importances.png")
print("📁 Feature importances saved to gradboost_feature_importances.png")

# Plot actual vs predicted
print("📊 Plotting actual vs. predicted quantities...")

# Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Actual': y_test_exp,
    'Predicted': y_pred
})

# Sort by actual values for better visualization
comparison_df = comparison_df.sort_values('Actual')

# Plot actual vs predicted
plt.figure(figsize=(12, 8))

# Scatter plot
plt.scatter(comparison_df['Actual'], comparison_df['Predicted'], 
            alpha=0.4, color='blue', edgecolors='none')

# Perfect prediction line
max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

# Labels and title
plt.xlabel('Actual Quantity')
plt.ylabel('Predicted Quantity')
plt.title(f'Gradient Boosting: Actual vs. Predicted Quantities (R² = {r2:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text with metrics
plt.annotate(f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
             verticalalignment='top')

plt.tight_layout()
plt.savefig("images/gradboost_actual_vs_predicted.png")
print("📁 Actual vs. predicted plot saved to gradboost_actual_vs_predicted.png")

# Plot residuals
plt.figure(figsize=(12, 6))
comparison_df['Residuals'] = comparison_df['Actual'] - comparison_df['Predicted']

# Plot residuals
plt.scatter(comparison_df['Actual'], comparison_df['Residuals'], 
            alpha=0.4, color='green', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Actual Quantity')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Gradient Boosting: Residual Plot')
plt.grid(True, alpha=0.3)

# Add a histogram of residuals as an inset
axins = inset_axes(plt.gca(), width="30%", height="30%", loc=2)
axins.hist(comparison_df['Residuals'], bins=30, color='skyblue', edgecolor='black')
axins.set_title('Residuals Distribution')
axins.set_xticks([])
axins.set_yticks([])

plt.tight_layout()
plt.savefig("images/gradboost_residuals.png")
print("📁 Residual plot saved to gradboost_residuals.png")

# Price elasticity analysis
print("\n📊 Analyzing price elasticity...")
price_multipliers = np.linspace(0.5, 2.5, 50)
baseline_predictions = np.expm1(model.predict(X_test))

results = []
for multiplier in price_multipliers:
    X_mod = X_test.copy()
    X_mod['Price'] *= multiplier
    
    # Recompute all price-dependent features
    X_mod = apply_feature_engineering(X_mod)
    X_mod = X_mod[extended_features]
    
    # Handle any missing values
    X_mod = X_mod.fillna(X_mod.median())
    
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

# Calculate elasticity
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

# Plot elasticity curve
plt.figure(figsize=(10, 6))
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Pred_Quantity'], marker='o', label='Demand')
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Smoothed_Quantity'], color='orange', linewidth=2, label='Smoothed')
plt.axvline(x=1.0, color='r', linestyle='--', label='Current Price')

# Calculate and plot revenue curve
revenues = elasticity_df['Price_Multiplier'] * elasticity_df['Pred_Quantity']
revenue_scale = elasticity_df['Pred_Quantity'].max() / revenues.max()
plt.plot(elasticity_df['Price_Multiplier'], revenues * revenue_scale, 
         marker='s', color='green', label='Revenue (scaled)')

# Calculate the price that maximizes revenue
max_revenue_idx = revenues.idxmax()
plt.axvline(x=elasticity_df.loc[max_revenue_idx, 'Price_Multiplier'], color='green', 
            linestyle=':', label=f'Max Revenue Price (x{elasticity_df.loc[max_revenue_idx, "Price_Multiplier"]:.2f})')

plt.title('Gradient Boosting: Price Sensitivity Analysis')
plt.xlabel('Price Multiplier')
plt.ylabel('Predicted Quantity / Scaled Revenue')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("images/gradboost_price_sensitivity.png")
print("📁 Price sensitivity plot saved to gradboost_price_sensitivity.png")

# If LightGBM model exists, compare with it
try:
    lgbm_model = joblib.load('../models/lgbm_model.pkl')
    
    print("\n🔍 Comparing Gradient Boosting vs LightGBM performance...")
    
    # Get LightGBM predictions
    lgbm_pred_log = lgbm_model.predict(X_test)
    lgbm_pred = np.expm1(lgbm_pred_log)
    
    # Calculate metrics for LightGBM
    lgbm_rmse = math.sqrt(mean_squared_error(y_test_exp, lgbm_pred))
    lgbm_mae = mean_absolute_error(y_test_exp, lgbm_pred)
    lgbm_r2 = r2_score(y_test_exp, lgbm_pred)
    
    # Print comparison
    print("\n📊 Model Comparison:")
    print(f"{'Metric':<10} {'LightGBM':>10} {'GradBoost':>10} {'Difference':>12}")
    print(f"{'-'*10:<10} {'-'*10:>10} {'-'*10:>10} {'-'*12:>12}")
    print(f"{'RMSE':<10} {lgbm_rmse:>10.2f} {rmse:>10.2f} {rmse-lgbm_rmse:>+12.2f}")
    print(f"{'MAE':<10} {lgbm_mae:>10.2f} {mae:>10.2f} {mae-lgbm_mae:>+12.2f}")
    print(f"{'R²':<10} {lgbm_r2:>10.2f} {r2:>10.2f} {r2-lgbm_r2:>+12.2f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    metrics = ['RMSE', 'MAE', '1-R²']
    lgbm_values = [lgbm_rmse, lgbm_mae, 1-lgbm_r2]
    gb_values = [rmse, mae, 1-r2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, lgbm_values, width, label='LightGBM')
    plt.bar(x + width/2, gb_values, width, label='Gradient Boosting')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('LightGBM vs Gradient Boosting (Lower is Better)')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("images/model_comparison.png")
    print("📁 Model comparison saved to model_comparison.png")
    
except Exception as e:
    print(f"\n⚠️ Could not compare with LightGBM model: {e}")

print("\n✅ Gradient Boosting demand prediction complete!")