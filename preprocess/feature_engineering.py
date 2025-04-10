import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load dataset
print("\U0001F4C2 Loading data...")
df = pd.read_csv("online_retail_II.csv", encoding='latin1', parse_dates=['InvoiceDate'])

# Filter valid data
df = df[(df['Quantity'] > 0) & (df['Price'] > 0) & (df['Customer ID'].notna())].copy()
print(f"✅ Filtered dataset: {len(df)} rows")

# Filter to most common products to reduce noise
top_products = df['StockCode'].value_counts().head(100).index
df = df[df['StockCode'].isin(top_products)].copy()
print(f"✔️ Filtered to top {len(top_products)} products")

# Add datetime features
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Week'] = df['InvoiceDate'].dt.isocalendar().week
df['IsHolidaySeason'] = df['Week'].isin([47, 48, 49, 50, 51, 52])
df['Weekday'] = df['InvoiceDate'].dt.weekday
df['IsWeekend'] = df['Weekday'] >= 5
df['Hour'] = df['InvoiceDate'].dt.hour

# Encode country
df['Country'] = df['Country'].astype('category')
df['CountryCode'] = df['Country'].cat.codes

# Cluster descriptions with Sentence-BERT
print("\U0001F9E0 Embedding descriptions with Sentence-BERT...")
unique_desc = df[['Description']].drop_duplicates().dropna().copy()
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(unique_desc['Description'].tolist(), show_progress_bar=True)

print("\U0001F500 Clustering descriptions...")
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
unique_desc['CategoryCluster'] = kmeans.fit_predict(embeddings)

# Merge clusters back
df = df.merge(unique_desc, on='Description', how='left')

# Aggregate weekly demand per product
df_weekly = df.groupby(['StockCode', 'Year', 'Week']).agg({
    'Quantity': 'sum',
    'Price': 'mean',
    'IsWeekend': 'sum',
    'Hour': 'mean',
    'CountryCode': 'first',
    'CategoryCluster': 'first',
    'Month': 'first',
    'IsHolidaySeason': 'first'
}).reset_index()

# Sort for lag feature generation
df_weekly.sort_values(by=['StockCode', 'Year', 'Week'], inplace=True)

# Lag and rolling mean features
df_weekly['Quantity_lag_1'] = df_weekly.groupby('StockCode')['Quantity'].shift(1)
df_weekly['Quantity_roll_mean_2'] = df_weekly.groupby('StockCode')['Quantity'].shift(1).rolling(window=2).mean().reset_index(level=0, drop=True)
df_weekly['Quantity_roll_mean_4'] = df_weekly.groupby('StockCode')['Quantity'].shift(1).rolling(window=4).mean().reset_index(level=0, drop=True)

# Drop rows with missing lags (from rolling computations)
df_weekly.dropna(inplace=True)

# Save to CSV
output_path = "engineered_weekly_demand_with_lags.csv"
df_weekly.to_csv(output_path, index=False)
print(f"📁 Saved to {output_path}")

# Optional: show top descriptions in each cluster
example_clusters = df[['Description', 'CategoryCluster']].drop_duplicates().dropna()
for cid in sorted(example_clusters['CategoryCluster'].unique()):
    print(f"\n\U0001F9E0 Cluster {cid}")
    print("-" * 30)
    for desc in example_clusters[example_clusters['CategoryCluster'] == cid]['Description'].head(5):
        print(f"• {desc}")