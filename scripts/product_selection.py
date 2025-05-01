import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Load and prepare data
print("Loading data...")
df = pd.read_csv("../data/online_retail_II.csv", encoding='latin1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Filter for valid data, FOCUS ON CUSTOMERS WITH ID
df = df[df['Quantity'] > 0]      # Filter out returns
df = df[df['Price'] > 0]         # Filter out free items
df = df[df['Customer ID'].notna()]  # ONLY customers with ID

print(f"Filtered data shape: {df.shape}")

# Group by product
product_stats = df.groupby('StockCode').agg({
    'Invoice': 'count',
    'Price': pd.Series.nunique,
    'Description': 'first'
}).reset_index()

# Rename columns
product_stats.columns = ['StockCode', 'NumSales', 'NumUniquePrices', 'Description']

# Filter for active and dynamic products
filtered = product_stats[
    (product_stats['NumSales'] > 100) &
    (product_stats['NumUniquePrices'] > 3)
].sort_values(by='NumSales', ascending=False)

# Pick top N
top_products = filtered.head(10)  # Getting more products since we have a narrower focus
print("\n===== Top Products with Dynamic Pricing (Registered Customers) =====")
print(top_products)

# Function to analyze and plot demand curve for a given product
def analyze_product_demand(stockcode, description):
    print(f"\nAnalyzing demand curve for {description} (StockCode: {stockcode})")
    
    # Select product data
    product_df = df[df['StockCode'] == stockcode].copy()
    
    # Print basic stats
    print(f"Total sales: {product_df['Quantity'].sum()}")
    print(f"Price range: £{product_df['Price'].min():.2f} to £{product_df['Price'].max():.2f}")
    print(f"Unique price points: {product_df['Price'].nunique()}")
    
    # Create a daily time series of average price and total quantity sold
    daily = product_df.groupby(product_df['InvoiceDate'].dt.date).agg({
        'Quantity': 'sum',
        'Price': 'mean'  # average unit price that day
    }).reset_index().dropna()
    
    print(f"Days with sales data: {len(daily)}")
    
    if len(daily) < 10:
        print("Not enough data points for analysis")
        return
    
    # Remove outliers for better curve fitting
    z_scores = stats.zscore(daily[['Price', 'Quantity']])
    daily_filtered = daily[(abs(z_scores) < 3).all(axis=1)]
    
    print(f"Days after removing outliers: {len(daily_filtered)}")
    
    if len(daily_filtered) < 10:
        print("Warning: Not enough data points after filtering")
        return
    
    # Apply LOWESS with robust error handling
    try:
        # Sort by price for better smoothing
        daily_filtered = daily_filtered.sort_values('Price')
        
        # Apply LOWESS
        lowess_result = sm.nonparametric.lowess(
            endog=daily_filtered['Quantity'], 
            exog=daily_filtered['Price'], 
            frac=0.5,
            it=3,
            return_sorted=True
        )
        
        smoothed_prices = lowess_result[:, 0]
        smoothed_quantities = lowess_result[:, 1]
        
        # Plot demand curve
        plt.figure(figsize=(12, 7))
        
        # Plot the raw data points
        plt.scatter(daily_filtered['Price'], daily_filtered['Quantity'], 
                   alpha=0.5, 
                   color='gray',
                   label='Daily Sales')
        
        # Plot LOWESS curve
        plt.plot(smoothed_prices, smoothed_quantities, 
               color='blue', 
               linewidth=2, 
               label='LOWESS')
        
        # Add linear regression
        x = daily_filtered['Price'].values.reshape(-1, 1)
        y = daily_filtered['Quantity'].values
        
        # Linear regression
        linear_model = sm.OLS(y, sm.add_constant(x)).fit()
        price_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        linear_pred = linear_model.predict(sm.add_constant(price_range))
        
        plt.plot(price_range, linear_pred, 
                color='green', 
                linestyle='--', 
                linewidth=1.5, 
                label=f'Linear (R²:{linear_model.rsquared:.2f})')
        
        # Polynomial regression (degree=2)
        poly_features = np.column_stack((x, x**2))
        poly_model = sm.OLS(y, sm.add_constant(poly_features)).fit()
        
        poly_range = np.column_stack((price_range, price_range**2))
        poly_pred = poly_model.predict(sm.add_constant(poly_range))
        
        plt.plot(price_range, poly_pred, 
                color='red', 
                linestyle='-.', 
                linewidth=1.5, 
                label=f'Polynomial (n=2, R²:{poly_model.rsquared:.2f})')
        
        # Calculate elasticity if possible
        avg_price = daily_filtered['Price'].mean()
        avg_quantity = daily_filtered['Quantity'].mean()
        
        # Find two points on the curve to estimate elasticity
        idx_above = np.searchsorted(smoothed_prices, avg_price * 1.1)
        idx_below = np.searchsorted(smoothed_prices, avg_price * 0.9)
        
        if idx_above < len(smoothed_prices) and idx_below > 0:
            price_diff = smoothed_prices[idx_above] - smoothed_prices[idx_below]
            qty_diff = smoothed_quantities[idx_above] - smoothed_quantities[idx_below]
            
            if price_diff != 0 and avg_price > 0 and avg_quantity > 0:
                elasticity = (qty_diff / avg_quantity) / (price_diff / avg_price)
                plt.annotate(f"Est. Price Elasticity: {elasticity:.2f}", 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                # Calculate optimal price if we have elasticity
                if elasticity < 0:
                    optimal_markup = -1/elasticity
                    optimal_price = avg_price * (1 + optimal_markup)
                    plt.annotate(f"Estimated Optimal Price: £{optimal_price:.2f}", 
                                xy=(0.05, 0.88), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Get model elasticities for comparison
        try:
            # Linear model elasticity at mean price
            linear_slope = linear_model.params[1]
            linear_elasticity = linear_slope * (avg_price / avg_quantity)
            
            # Polynomial model elasticity at mean price
            # For polynomial y = a + bx + cx², the derivative is b + 2cx
            poly_b = poly_model.params[1]
            poly_c = poly_model.params[2]
            poly_slope = poly_b + 2 * poly_c * avg_price
            poly_elasticity = poly_slope * (avg_price / avg_quantity)
            
            plt.annotate(f"Linear Model Elasticity: {linear_elasticity:.2f}\nPoly Model Elasticity: {poly_elasticity:.2f}", 
                        xy=(0.05, 0.80), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except:
            # If calculation fails, just continue without model elasticities
            pass
        
        plt.title(f'Demand Curve for {description} (Registered Customers)\n(StockCode: {stockcode})')
        plt.xlabel('Price (£)')
        plt.ylabel('Quantity Sold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'images/demand_estimation_{stockcode}_registered.png')
        print(f"Demand curve saved to images/demand_estimation_{stockcode}_registered.png")
        
        # Additional plot: Price over time
        plt.figure(figsize=(12, 7))
        timeplot = product_df.groupby(product_df['InvoiceDate'].dt.date).agg({
            'Price': ['mean', 'min', 'max'],
            'Quantity': 'sum'
        })
        timeplot.columns = ['AvgPrice', 'MinPrice', 'MaxPrice', 'Quantity']
        timeplot = timeplot.reset_index()
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (£)', color=color)
        ax1.plot(timeplot['InvoiceDate'], timeplot['AvgPrice'], color=color, label='Average Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Quantity', color=color)
        ax2.plot(timeplot['InvoiceDate'], timeplot['Quantity'], color=color, alpha=0.7, label='Quantity')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f'Price and Quantity Over Time - Registered Customers\n{description} (StockCode: {stockcode})')
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.savefig(f'images/price_time_{stockcode}_registered.png')
        print(f"Price time series saved to images/price_time_{stockcode}_registered.png")
        
    except Exception as e:
        print(f"Error in analysis: {e}")

# Analyze each top product
for _, row in top_products.iterrows():
    analyze_product_demand(row['StockCode'], row['Description'])

print("\nAnalysis complete!")