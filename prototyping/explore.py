import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('../data/online_retail_II.csv', encoding='latin1')

# Filter for stockcodes that start with exactly 5 digits
df = df[df['StockCode'].str.match(r'^\d{5}')]

print(f"\nAfter filtering, {len(df)} records remain")
print(f"Number of unique products: {df['StockCode'].nunique()}")

# Find number of unique prices per StockCode
unique_prices = df.groupby('StockCode')['Price'].nunique()

# Create price analysis DataFrame
price_analysis = df.groupby('StockCode').agg({
    'Price': lambda x: x.nunique(),
    'Description': 'first',
    'Invoice': 'count'
}).reset_index()

# Rename columns
price_analysis.columns = ['StockCode', 'UniquePrice', 'Description', 'Transactions']

# Sort by number of unique prices
price_analysis = price_analysis.sort_values(by='UniquePrice', ascending=False)

# Get the product with most price variations
target_stockcode = price_analysis.iloc[0]['StockCode']
target_description = price_analysis.iloc[0]['Description']

# Filter data for this product
product_df = df[df['StockCode'] == target_stockcode].copy()
product_df['InvoiceDate'] = pd.to_datetime(product_df['InvoiceDate'])
product_df['HasCustomerID'] = ~product_df['Customer ID'].isna()

# Print summary stats grouped by whether customer ID exists
customer_id_stats = product_df.groupby('HasCustomerID')['Price'].agg(['mean', 'median', 'min', 'max', 'count'])
print("\n===== Price Analysis by Customer ID Presence =====")
print(customer_id_stats)

# Calculate percentage difference
with_id_mean = customer_id_stats.loc[True, 'mean']
without_id_mean = customer_id_stats.loc[False, 'mean'] if False in customer_id_stats.index else 0
if without_id_mean > 0:
    pct_diff = (without_id_mean - with_id_mean) / with_id_mean * 100
    print(f"\nPrices without Customer ID are {pct_diff:.1f}% {'higher' if pct_diff > 0 else 'lower'} on average")

# Plot price comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='HasCustomerID', y='Price', data=product_df)
plt.title(f'Price Distribution With/Without Customer ID\n{target_description} (StockCode: {target_stockcode})')
plt.xlabel('Has Customer ID')
plt.ylabel('Price (£)')
plt.xticks([0, 1], ['No (Retail)', 'Yes (Registered)'])
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('price_by_customer_id.png')

# Plot price over time, colored by customer ID status
plt.figure(figsize=(12, 6))
for has_id, group in product_df.groupby('HasCustomerID'):
    label = 'With Customer ID (Registered)' if has_id else 'Without Customer ID (Retail)'
    color = 'blue' if has_id else 'red'
    plt.scatter(group['InvoiceDate'], group['Price'], 
                label=label, color=color, alpha=0.6, s=30)

plt.title(f'Price Changes Over Time by Customer Type\n{target_description} (StockCode: {target_stockcode})')
plt.xlabel('Date')
plt.ylabel('Price (£)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('price_time_by_customer_type.png')

# Let's also check if this pattern holds across all products
print("\n===== Testing if pattern holds across all products =====")
df['HasCustomerID'] = ~df['Customer ID'].isna()
overall_price_diff = df.groupby('HasCustomerID')['Price'].agg(['mean', 'median', 'count'])
print(overall_price_diff)

if False in overall_price_diff.index and True in overall_price_diff.index:
    overall_pct_diff = (overall_price_diff.loc[False, 'mean'] - overall_price_diff.loc[True, 'mean']) / overall_price_diff.loc[True, 'mean'] * 100
    print(f"Across ALL products, prices without Customer ID are {overall_pct_diff:.1f}% {'higher' if overall_pct_diff > 0 else 'lower'}")

# Let's check the top 10 products with the biggest price difference
product_price_by_id = df.groupby(['StockCode', 'HasCustomerID'])['Price'].mean().reset_index()
product_price_pivot = product_price_by_id.pivot(index='StockCode', columns='HasCustomerID', values='Price')
product_price_pivot.columns = ['WithoutID', 'WithID']
product_price_pivot = product_price_pivot.dropna()  # Only keep products with both customer types
product_price_pivot['PriceDiff'] = product_price_pivot['WithoutID'] / product_price_pivot['WithID']
product_price_pivot['PercentDiff'] = (product_price_pivot['WithoutID'] - product_price_pivot['WithID']) / product_price_pivot['WithID'] * 100

# Add description and transaction count
product_price_pivot = product_price_pivot.reset_index()
product_info = df.groupby('StockCode').agg({
    'Description': 'first',
    'Invoice': 'count'
}).reset_index()
product_price_pivot = product_price_pivot.merge(product_info, on='StockCode')

# Sort by biggest price difference (both directions)
highest_markup = product_price_pivot.sort_values('PercentDiff', ascending=False).head(10)
print("\n===== Products with HIGHEST price markup for non-registered customers =====")
print(highest_markup[['StockCode', 'Description', 'WithID', 'WithoutID', 'PercentDiff', 'Invoice']])

highest_discount = product_price_pivot.sort_values('PercentDiff', ascending=True).head(10)
print("\n===== Products with LOWEST price markup (discounts) for non-registered customers =====")
print(highest_discount[['StockCode', 'Description', 'WithID', 'WithoutID', 'PercentDiff', 'Invoice']])

# Group by both customer ID status and quantity ranges
df['QtyBucket'] = pd.cut(df['Quantity'], bins=[0, 10, 50, 100, 500, float('inf')])
price_by_id_qty = df.groupby(['HasCustomerID', 'QtyBucket'])['Price'].mean()
print(price_by_id_qty)

# Analyze monthly price differences
df['Month'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('M')
monthly_diff = df.groupby(['Month', 'HasCustomerID'])['Price'].mean().unstack()
monthly_diff['Ratio'] = monthly_diff[False] / monthly_diff[True]

# Classify products into categories based on descriptions
df['Category'] = df['Description'].str.extract('(CANDLE|MUG|CUSHION|LIGHT|BOX|BAG)')
category_diff = df.groupby(['Category', 'HasCustomerID'])['Price'].mean().unstack()
category_diff['PriceDiff'] = (category_diff[False] / category_diff[True] - 1) * 100

# Print category price differences
print("\n===== Category Price Differences =====")
print(category_diff)

# Plot category price differences - CORRECTED VERSION
plt.figure(figsize=(12, 8))
# Access MultiIndex columns by their names
cat_plot_data = pd.DataFrame({
    'Without CustomerID': category_diff[False],
    'With CustomerID': category_diff[True]
})
cat_plot_data.plot(kind='bar', figsize=(12, 6))
plt.title('Average Price by Category and Customer Type')
plt.xlabel('Product Category')
plt.ylabel('Average Price (£)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('category_price_comparison.png')

# Plot price ratio by category
plt.figure(figsize=(10, 6))
category_diff['PriceDiff'].plot(kind='barh', color='teal')
plt.title('Price Difference by Category (% Higher for Non-Registered Customers)')
plt.xlabel('% Price Difference')
plt.ylabel('Product Category')
plt.axvline(x=0, color='black', linestyle='-')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('category_price_difference.png')

# Plot quantity vs price analysis - CORRECTED VERSION
plt.figure(figsize=(12, 8))
# Create a better format for the quantity price data
qty_price_data = pd.DataFrame({
    'Without CustomerID': price_by_id_qty.loc[False],
    'With CustomerID': price_by_id_qty.loc[True]
})
qty_price_data.plot(kind='bar')
plt.title('Average Price by Quantity Range and Customer Type')
plt.xlabel('Quantity Range')
plt.ylabel('Average Price (£)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('quantity_price_analysis.png')

# Analyze monthly price differences
df['Month'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('M')
monthly_diff = df.groupby(['Month', 'HasCustomerID'])['Price'].mean().unstack()
monthly_diff['Ratio'] = monthly_diff[False] / monthly_diff[True]

# Print monthly price differences
print("\n===== Monthly Price Differences =====")
print(monthly_diff)

# Plot monthly price trends
plt.figure(figsize=(14, 7))
ax1 = plt.subplot(111)
monthly_diff[True].plot(kind='line', ax=ax1, color='blue', marker='o', linestyle='-', label='With Customer ID')
monthly_diff[False].plot(kind='line', ax=ax1, color='red', marker='x', linestyle='-', label='Without Customer ID')
ax2 = ax1.twinx()
monthly_diff['Ratio'].plot(kind='line', ax=ax2, color='green', marker='s', linestyle='--', label='Price Ratio')

ax1.set_xlabel('Month')
ax1.set_ylabel('Average Price (£)')
ax2.set_ylabel('Price Ratio (No ID / With ID)')
plt.title('Average Monthly Prices by Customer Type')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_price_trends.png')

# Add some additional insights based on the data we've seen
print("\n===== KEY INSIGHTS =====")
print(f"1. Across all products, non-registered customers pay {category_diff['PriceDiff'].mean():.1f}% more on average")
print(f"2. MUG category has the highest price difference at {category_diff.loc['MUG', 'PriceDiff']:.1f}%")
print(f"3. The price difference is most pronounced for smaller quantities ({qty_price_data.iloc[0,0]/qty_price_data.iloc[0,1]-1:.1%} higher)")
print(f"4. The price gap has {'increased' if monthly_diff['Ratio'].iloc[-1] > monthly_diff['Ratio'].iloc[0] else 'decreased'} over time")
print(f"5. Month with highest price difference: {monthly_diff['Ratio'].idxmax()} ({monthly_diff['Ratio'].max():.2f}x)")