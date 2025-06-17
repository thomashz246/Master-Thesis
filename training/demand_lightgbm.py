# demand.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation
import time
import joblib
import math
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore", category=FutureWarning)
# Import necessary libraries

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

# Add this function to your script
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

# Then replace your feature engineering section with:
df_clean = apply_feature_engineering(df_clean)

# Add these features before extended_features list
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

# Price elasticity might vary by product popularity (now StockCode_popularity exists)
df_clean['Price_Popularity'] = df_clean['Price'] * np.log1p(df_clean['StockCode_popularity'])

# Category-specific price effects
df_clean['CategoryPrice_ratio'] = df_clean['Price'] / df_clean.groupby(
    ['CategoryCluster', 'Week', 'Year'])['Price'].transform('mean')

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
n_splits = 8  # or 10 if you have many weeks
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
    
    model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=2048,
    learning_rate=0.03,
    num_leaves=256,
    max_bin=512,
    min_data_in_leaf=20, 
    n_jobs=4,
    verbose=-1
    )

    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[early_stopping(stopping_rounds=100)]
    )

    y_val_pred_log = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_log)
    y_val_true = np.expm1(y_val)

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
plt.savefig("images/cv_results.png")
print("📁 Cross-validation results saved to cv_results.png")

# First, move the cache loading to be the first thing after cross-validation
try:
    # Check if cache exists
    import os
    if os.path.exists('../models/preprocessing_cache.pkl'):
        print("Loading cached data and CV results...")
        data_cache = joblib.load('../models/preprocessing_cache.pkl')
        X = data_cache['X']
        y = data_cache['y']
        best_params = data_cache['best_params']
        # Now you can continue with train/test split
except:
    print("No cache found, continuing with normal processing")
    # Define best params here since they weren't loaded from cache
    print("\n🔍 Using pre-defined hyperparameters...")
    best_params = {
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_data_in_leaf': 30,
        'num_leaves': 128,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }

# Later you could load this instead of rerunning everything
try:
    # Check if cache exists
    import os
    if os.path.exists('../models/preprocessing_cache.pkl'):
        print("Loading cached data and CV results...")
        data_cache = joblib.load('../models/preprocessing_cache.pkl')
        X = data_cache['X']
        y = data_cache['y']
        best_params = data_cache['best_params']
        # Now you can continue with train/test split
except:
    print("No cache found, continuing with normal processing")

# Replace the current param_grid with this smaller version
param_grid = {
    'learning_rate': [0.03, 0.1],
    'num_leaves': [64, 256],
    'max_depth': [8, -1],
    'min_data_in_leaf': [20, 50],
    'reg_alpha': [0, 0.5],
    'reg_lambda': [0, 0.5]
}

# Use a smaller subset for hyperparameter tuning (for speed)
X_sample = X.sample(min(5000, len(X)))
y_sample = y.loc[X_sample.index]

# Import TimeoutError
from concurrent.futures import TimeoutError

# Modify your GridSearchCV with a timeout parameter
from sklearn.model_selection import GridSearchCV

print("\n🔍 Using pre-defined hyperparameters instead of grid search...")
best_params = {
    'learning_rate': 0.03,
    'max_depth': 8,
    'min_data_in_leaf': 30,
    'num_leaves': 128,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

print("\n⏱️ Starting feature selection and final model training...")
# Feature selection starts here
print("\n🚀 Training final model on train/test split...")
# Hold out the last 10% of weeks as test set
df_clean = df_clean.sort_values(['Year', 'Week'])  # already done earlier
cutoff = int(len(df_clean) * 0.9)
X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
print(f"📚 Train: {len(X_train)}, Test: {len(X_test)}")

# Just use all features if the specific feature selection is causing issues
X_train_selected = X_train
X_test_selected = X_test
print("\nShow one data row used as a base:")
print(X_test_selected.iloc[0])  # Show one data row used as a base
print(f"📋 Using all {X_train.shape[1]} features without selection")

print("✅ Feature selection completed")
print("\n⏱️ Training final model...")
start_time = time.time()
# Optimize for speed in the final model (replace lines ~290-302)
print("🚀 Training LightGBM model...")
model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=2000,
    learning_rate=best_params['learning_rate'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    min_data_in_leaf=best_params['min_data_in_leaf'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    n_jobs=4,
    verbose=-1,
    # Speed optimizations
    subsample=0.8,
    feature_fraction=0.8,
    force_row_wise=True
)

start_time = time.time()

model.fit(
    X_train_selected, y_train,
    eval_set=[(X_test_selected, y_test)],
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
y_pred_log = model.predict(X_test_selected)
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
features = X_train_selected.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature Importances (Log Demand)")
plt.tight_layout()
plt.savefig("images/feature_importances_log.png")
print("📁 Feature importances saved to feature_importances_log.png")

# Add this code after the feature importance plot (around line 395)
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
plt.title(f'Actual vs. Predicted Quantities (R² = {r2:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text with metrics
plt.annotate(f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
             verticalalignment='top')

# Add a second plot showing residuals
plt.figure(figsize=(12, 6))
comparison_df['Residuals'] = comparison_df['Actual'] - comparison_df['Predicted']

# Plot residuals
plt.scatter(comparison_df['Actual'], comparison_df['Residuals'], 
            alpha=0.4, color='green', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Actual Quantity')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Add a histogram of residuals as an inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(plt.gca(), width="30%", height="30%", loc=2)
axins.hist(comparison_df['Residuals'], bins=30, color='skyblue', edgecolor='black')
axins.set_title('Residuals Distribution')
axins.set_xticks([])
axins.set_yticks([])

plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png")
print("📁 Actual vs. predicted plot saved to actual_vs_predicted.png")

# Add a third plot showing prediction error distribution by quantity range
plt.figure(figsize=(12, 6))

# Create quantity bins
comparison_df['Quantity_Bin'] = pd.cut(comparison_df['Actual'], 10)
bin_errors = comparison_df.groupby('Quantity_Bin').apply(
    lambda x: np.sqrt(mean_squared_error(x['Actual'], x['Predicted']))
).reset_index(name='Bin_RMSE')

# Plot errors by bin
plt.bar(range(len(bin_errors)), bin_errors['Bin_RMSE'], 
        width=0.8, color='purple', alpha=0.7)
plt.xticks(range(len(bin_errors)), 
           [f"{b.left:.0f}-{b.right:.0f}" for b in bin_errors['Quantity_Bin']], 
           rotation=45)
plt.xlabel('Actual Quantity Range')
plt.ylabel('RMSE within Range')
plt.title('Prediction Error by Quantity Range')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("images/prediction_error_by_range.png")
print("📁 Prediction error plot saved to prediction_error_by_range.png")

# Price elasticity analysis
print("\n📊 Analyzing price elasticity...")
price_col_idx = np.where(X_train_selected.columns == 'Price')[0][0]
price_multipliers = np.linspace(0.5, 2.5, 50)
baseline_predictions = np.expm1(model.predict(X_test_selected))

results = []
for multiplier in price_multipliers:
    X_mod = X_test_selected.copy()
    X_mod['Price'] *= multiplier
    
    # IMPORTANT: Recompute all price-dependent features
    X_mod = apply_feature_engineering(X_mod)
    X_mod = X_mod[extended_features]  # Restrict to original feature set
    
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

# Increase font sizes globally
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16
})

# Smooth the predicted quantity with a rolling average
elasticity_df['Smoothed_Quantity'] = elasticity_df['Pred_Quantity'].rolling(window=5, center=True).mean()

# Find the exact baseline at multiplier=1.0 by interpolation
# First find the closest points around 1.0
closest_idx = (elasticity_df['Price_Multiplier'] - 1.0).abs().idxmin()
if elasticity_df.loc[closest_idx, 'Price_Multiplier'] < 1.0 and closest_idx < len(elasticity_df) - 1:
    lower_idx, upper_idx = closest_idx, closest_idx + 1
elif elasticity_df.loc[closest_idx, 'Price_Multiplier'] > 1.0 and closest_idx > 0:
    lower_idx, upper_idx = closest_idx - 1, closest_idx
else:
    lower_idx = upper_idx = closest_idx  # Exact match found or at boundary

# Interpolate to get exact smoothed quantity at multiplier=1.0
if lower_idx != upper_idx:
    lower_price = elasticity_df.loc[lower_idx, 'Price_Multiplier']
    upper_price = elasticity_df.loc[upper_idx, 'Price_Multiplier']
    lower_qty = elasticity_df.loc[lower_idx, 'Smoothed_Quantity']
    upper_qty = elasticity_df.loc[upper_idx, 'Smoothed_Quantity']
    
    # Linear interpolation
    baseline_quantity = lower_qty + (upper_qty - lower_qty) * (1.0 - lower_price) / (upper_price - lower_price)
else:
    # Exact match or boundary case
    baseline_quantity = elasticity_df.loc[closest_idx, 'Smoothed_Quantity']

print(f"Using reference point at price multiplier 1.0 with smoothed quantity {baseline_quantity:.2f}")

# Add quantity multiplier columns
elasticity_df['Quantity_Multiplier'] = elasticity_df['Pred_Quantity'] / baseline_quantity
elasticity_df['Smoothed_Quantity_Multiplier'] = elasticity_df['Smoothed_Quantity'] / baseline_quantity

# Plot with quantity multipliers
plt.figure(figsize=(10, 6))
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Quantity_Multiplier'], 
         marker='o', label='Raw')
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Smoothed_Quantity_Multiplier'], 
         color='orange', linewidth=4, label='Smoothed (rolling mean)')
# Add reference lines without including them in the legend
plt.axvline(x=1.0, color='r', linestyle='--')
plt.axhline(y=1.0, color='r', linestyle=':')
plt.xlabel('Price Multiplier')
plt.ylabel('Quantity Multiplier')
plt.title('Price Sensitivity Analysis')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/price_sensitivity_normalized.png")
print("📁 Normalized price sensitivity analysis saved to price_sensitivity_normalized.png")

# Original plot code with the same reference point
plt.figure(figsize=(10, 6))
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Pred_Quantity'], marker='o', label='Raw')
plt.plot(elasticity_df['Price_Multiplier'], elasticity_df['Smoothed_Quantity'], color='orange', linewidth=2, label='Smoothed (rolling mean)')
# Add reference lines without including them in the legend
plt.axvline(x=1.0, color='r', linestyle='--')
plt.axhline(y=baseline_quantity, color='r', linestyle=':')
plt.xlabel('Price Multiplier', fontsize=14)
plt.ylabel('Predicted Quantity', fontsize=14)
plt.title('Price Sensitivity Analysis (Smoothed)', fontsize=16)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/price_sensitivity_smoothed.png")
print("📁 Smoothed price sensitivity analysis saved to price_sensitivity_smoothed.png")

# Predicted quantity by week
print("\n🕰️ Plotting predicted weekly quantity trend...")
predicted_quantities = np.expm1(model.predict(X))
df_clean = df_clean.copy()  # Create an explicit copy
df_clean.loc[:, 'PredictedQuantity'] = predicted_quantities
weekly_avg = df_clean.groupby(['Year', 'Week'])['PredictedQuantity'].mean().reset_index()
weekly_avg['YearWeek'] = weekly_avg['Year'].astype(str) + "-W" + weekly_avg['Week'].astype(str)
weekly_avg = weekly_avg.sort_values(['Year', 'Week'])

print("\n Summing predicted demand across weeks...")
predicted_log = model.predict(X)
df_clean = df_clean.copy()  # Create another explicit copy if needed
df_clean.loc[:, 'Predicted_LogQuantity'] = predicted_log
df_clean.loc[:, 'Predicted_Quantity'] = np.expm1(df_clean['Predicted_LogQuantity'])

weekly_sum = df_clean.groupby('Week')['Predicted_Quantity'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(weekly_sum['Week'], weekly_sum['Predicted_Quantity'], marker='o')
plt.title('Total Predicted Demand per Week (Aggregated Over All Years)')
plt.xlabel('Week Number')
plt.ylabel('Summed Predicted Quantity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/predicted_demand_summed_by_week.png")
print("📁 Weekly demand plot saved to predicted_demand_summed_by_week.png")

# Replace the problematic code around line 623 with this:

print("\n📊 Analyzing price elasticity for specific products...")

# Select a random product for elasticity analysis
product_id = df_clean['StockCode'].sample(1).values[0]
print(f"🔍 Analyzing price elasticity for random product: {product_id}")

# Filter data for this product
product_mask = df_clean['StockCode'] == product_id

# Check if we have enough data for this product
if sum(product_mask) < 10:  # Require at least 10 records for meaningful analysis
    print(f"⚠️ Not enough data for product {product_id}. Finding another product...")
    # Get products with sufficient data
    product_counts = df_clean['StockCode'].value_counts()
    valid_products = product_counts[product_counts >= 10].index
    if len(valid_products) > 0:
        product_id = np.random.choice(valid_products)
        print(f"🔄 Using product {product_id} instead")
        product_mask = df_clean['StockCode'] == product_id
    else:
        print("⚠️ No products with sufficient data found")

# Check if we have data for this product
if sum(product_mask) < 10:
    print("⚠️ Not enough data available for meaningful analysis")
else:
    print(f"📊 Found {sum(product_mask)} records for product {product_id}")
    
    # Get the complete product data
    product_df = df_clean[product_mask].copy()
    
    # Process product data for prediction (without using df_train)
    # We'll apply feature engineering to the product data directly
    product_featured = apply_feature_engineering(product_df)
    
    # Get the features for prediction
    X_product = product_featured[extended_features]
    
    # Get baseline prediction
    baseline = np.expm1(model.predict(X_product))
    baseline_mean = baseline.mean()
    print(f"🔍 Baseline average demand: {baseline_mean:.2f} units")
    
    # Calculate elasticities for this specific product
    product_elasticities = []
    price_values = []
    
    # Get the original price of this product for reference
    original_price = X_product['Price'].mean()
    print(f"💰 Original average price: £{original_price:.2f}")
    
    for multiplier in price_multipliers:
        # Make a copy of the product data
        product_modified = product_df.copy()
        
        # Record the actual price value (not just the multiplier)
        price_values.append(original_price * multiplier)
        
        # Modify the price
        product_modified['Price'] *= multiplier
        
        # Apply feature engineering to update all price-dependent features
        product_modified = apply_feature_engineering(product_modified)
        
        # Get the features needed for prediction
        X_mod = product_modified[extended_features]
        
        # Make predictions with the modified data
        preds = np.expm1(model.predict(X_mod))
        product_elasticities.append(preds.mean())
    
    # Calculate elasticity
    p_up = price_multipliers[-1]
    p_down = price_multipliers[0]
    q_up = product_elasticities[-1]
    q_down = product_elasticities[0]
    q_base = product_elasticities[np.abs(np.array(price_multipliers) - 1.0).argmin()]
    
    product_elasticity = ((q_up - q_down) / q_base) / ((p_up - p_down) / 1.0)
    print(f"📉 Product Elasticity: {product_elasticity:.3f}")
    
    if product_elasticity < 0:
        if abs(product_elasticity) > 1:
            print("📈 Product is ELASTIC - quantity changes more than price")
            opt_multiplier = -1/(1 + product_elasticity)
            print(f"💰 Revenue-maximizing price change: {(opt_multiplier - 1)*100:.1f}%")
        else:
            print("📉 Product is INELASTIC - quantity less sensitive to price")
    else:
        print("⚠️ Unusual elasticity (positive) - check your model")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(price_multipliers, product_elasticities, marker='o', color='blue', label='Demand')
    # Reference line without legend entry
    plt.axvline(x=1.0, color='red', linestyle='--')

    # Calculate and plot revenue curve
    revenues = [p * q for p, q in zip(price_values, product_elasticities)]
    revenue_scale = max(product_elasticities) / max(revenues)
    plt.plot(price_multipliers, [r * revenue_scale for r in revenues], 
             marker='s', color='green', label='Revenue (scaled)')

    # Calculate the price that maximizes revenue
    max_revenue_idx = np.argmax(revenues)
    plt.axvline(x=price_multipliers[max_revenue_idx], color='green', 
                linestyle=':', label=f'Max Revenue Price (x{price_multipliers[max_revenue_idx]:.2f})')

    plt.title(f'Price Sensitivity for Product {product_id}', fontsize=16)
    plt.xlabel('Price Multiplier', fontsize=14)
    plt.ylabel('Predicted Quantity / Scaled Revenue', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"product_elasticity/product_{product_id}_elasticity.png")
    print(f"📁 Product elasticity plot saved to product_{product_id}_elasticity.png")
