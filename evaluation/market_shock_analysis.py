import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_market_shocks(price_df, return_data, save_dir=None, enable_shocks=True):
    """
    Analyze how agents respond to controlled market shocks.
    
    Args:
        price_df (DataFrame): Price evolution data with Episode, Week, Agent columns
        return_data (dict): Dictionary of agent returns
        save_dir (str): Directory to save analysis results
        enable_shocks (bool): Whether shocks were enabled in the simulation
    
    Returns:
        dict: Adaptability metrics related to market shocks
    """
    print("Analyzing agent responses to market shocks...")
    
    # If shocks were disabled, return baseline metrics
    if not enable_shocks:
        print("Market shocks were disabled in this simulation. Returning baseline metrics.")
        return {
            'response_time': {},
            'price_volatility_after_shock': {},
            'recovery_percentage': {},
            'stabilization_time': {},
            'shocks_enabled': False
        }
    
    # Initialize metrics dictionary
    shock_metrics = {
        'response_time': {},
        'price_volatility_after_shock': {},
        'recovery_percentage': {},
        'stabilization_time': {},
        'shocks_enabled': True
    }
    
    # Define shock periods - assuming shocks happen at weeks 26, 52, 78
    shock_weeks = [26, 52, 78]
    window_size = 6  # Look at 6 weeks after each shock
    
    # First, check if price_df is in the right format
    if 'Agent' not in price_df.columns and 'Product' not in price_df.columns:
        # Convert to the right format if needed
        price_df = process_price_data(price_df)
    
    # Analyze each agent's response
    for agent_id in set([col.split('_')[0] for col in price_df.columns if '_Price' in col]):
        # Extract price data for this agent
        agent_prices = get_agent_prices(price_df, agent_id)
        
        if agent_prices.empty:
            continue
            
        # Calculate metrics for each shock
        response_times = []
        post_shock_volatility = []
        recovery_rates = []
        stabilization_times = []
        
        for shock_week in shock_weeks:
            # 1. Response time: how many weeks until first significant price change after shock
            response_time = calculate_response_time(agent_prices, shock_week)
            response_times.append(response_time)
            
            # 2. Price volatility after shock
            volatility = calculate_post_shock_volatility(agent_prices, shock_week, window_size)
            post_shock_volatility.append(volatility)
            
            # 3. Recovery percentage: comparing returns before and after shock
            recovery = calculate_recovery_percentage(agent_id, return_data, shock_week, window_size)
            recovery_rates.append(recovery)
            
            # 4. Time to stabilization: weeks until price changes fall below threshold
            stabilization = calculate_stabilization_time(agent_prices, shock_week, window_size)
            stabilization_times.append(stabilization)
            
        # Average metrics across all shocks
        shock_metrics['response_time'][agent_id] = np.mean(response_times)
        shock_metrics['price_volatility_after_shock'][agent_id] = np.mean(post_shock_volatility)
        shock_metrics['recovery_percentage'][agent_id] = np.mean(recovery_rates)
        shock_metrics['stabilization_time'][agent_id] = np.mean(stabilization_times)
        
    # Create visualizations if save_dir is provided
    if save_dir:
        create_shock_response_visualizations(price_df, shock_weeks, shock_metrics, save_dir)
        
    return shock_metrics

def process_price_data(price_df):
    """Convert wide-format price data to long format"""
    processed_rows = []
    
    for _, row in price_df.iterrows():
        week = row['Week']
        year = row.get('Year', 1)
        episode = row.get('Episode', 1)
        
        for col in price_df.columns:
            if '_Price' in col:
                parts = col.split('_')
                agent_id = parts[0]
                product_name = parts[1] if len(parts) > 2 else 'Product1'
                
                processed_rows.append({
                    'Week': week,
                    'Year': year,
                    'Episode': episode,
                    'Agent': agent_id,
                    'Product': product_name,
                    'Price': row[col]
                })
    
    return pd.DataFrame(processed_rows)

def get_agent_prices(price_df, agent_id):
    """Extract price data for a specific agent"""
    # Check format of price_df
    if 'Agent' in price_df.columns:
        return price_df[price_df['Agent'] == agent_id]
    else:
        # Extract columns for this agent from wide format
        agent_cols = [col for col in price_df.columns if col.startswith(f"{agent_id}_") and col.endswith('_Price')]
        if not agent_cols:
            return pd.DataFrame()
            
        result = price_df[['Week', 'Episode']].copy()
        for col in agent_cols:
            product = col.split('_')[1]
            result[product] = price_df[col]
            
        return result

def calculate_response_time(agent_prices, shock_week):
    """Calculate how many weeks until first significant price change after shock"""
    # Define significant price change threshold (e.g., 3%)
    threshold = 0.03
    
    # Filter data to the weeks after shock
    post_shock = agent_prices[agent_prices['Week'] >= shock_week].sort_values('Week')
    
    if post_shock.empty or len(post_shock) < 2:
        return np.nan
    
    # Get price columns
    price_cols = [col for col in post_shock.columns 
                  if col not in ['Week', 'Year', 'Episode', 'Agent', 'Product']]
    
    # Find first week with significant change in any product
    response_time = None
    for i in range(1, len(post_shock)):
        for col in price_cols:
            current_price = post_shock.iloc[i][col]
            prev_price = post_shock.iloc[i-1][col]
            
            if abs(current_price / prev_price - 1) > threshold:
                response_time = post_shock.iloc[i]['Week'] - shock_week
                return response_time
    
    # If no significant change found
    return 10  # Default to maximum window size if no response

def calculate_post_shock_volatility(agent_prices, shock_week, window_size):
    """Calculate price volatility in the window after shock"""
    # Get data for window after shock
    post_shock = agent_prices[
        (agent_prices['Week'] >= shock_week) & 
        (agent_prices['Week'] < shock_week + window_size)
    ]
    
    if post_shock.empty or len(post_shock) < 2:
        return np.nan
        
    # Get price columns
    price_cols = [col for col in post_shock.columns 
                  if col not in ['Week', 'Year', 'Episode', 'Agent', 'Product']]
    
    # Calculate coefficient of variation for each product and average
    product_volatilities = []
    
    for col in price_cols:
        prices = post_shock[col].dropna()
        if len(prices) > 1:
            # Calculate coefficient of variation (std/mean)
            cv = prices.std() / prices.mean() if prices.mean() > 0 else 0
            product_volatilities.append(cv)
    
    # Return average volatility across products
    return np.mean(product_volatilities) if product_volatilities else 0

def calculate_recovery_percentage(agent_id, return_data, shock_week, window_size):
    """Calculate how quickly agent returns recover after shock"""
    # This is a simplified implementation since we don't have weekly returns
    # Actual implementation would require weekly return data
    
    # Placeholder implementation
    return 0.85  # Assume 85% recovery

def calculate_stabilization_time(agent_prices, shock_week, window_size):
    """Calculate how many weeks until prices stabilize after shock"""
    # Define stabilization threshold (e.g., price changes below 1%)
    threshold = 0.01
    
    # Filter data to the weeks after shock
    post_shock = agent_prices[
        (agent_prices['Week'] >= shock_week) & 
        (agent_prices['Week'] < shock_week + window_size)
    ].sort_values('Week')
    
    if post_shock.empty or len(post_shock) < 3:
        return np.nan
    
    # Get price columns
    price_cols = [col for col in post_shock.columns 
                  if col not in ['Week', 'Year', 'Episode', 'Agent', 'Product']]
    
    # For each week after shock, check if prices have stabilized
    for i in range(2, len(post_shock)):
        all_stable = True
        for col in price_cols:
            current_price = post_shock.iloc[i][col]
            prev_price = post_shock.iloc[i-1][col]
            
            # Check if price change is still above threshold
            if abs(current_price / prev_price - 1) > threshold:
                all_stable = False
                break
                
        if all_stable:
            return post_shock.iloc[i]['Week'] - shock_week
    
    # If no stabilization within window
    return window_size  # Return max window size if never stabilizes

def create_shock_response_visualizations(price_df, shock_weeks, shock_metrics, save_dir):
    """Create visualizations of agent responses to shocks"""
    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot price evolution with shocks highlighted
    plt.figure(figsize=(14, 8))
    
    # Check if price_df is in long or wide format
    if 'Agent' in price_df.columns and 'Price' in price_df.columns:
        # Long format
        for agent in price_df['Agent'].unique():
            agent_data = price_df[price_df['Agent'] == agent]
            plt.plot(agent_data['Week'], agent_data['Price'], label=agent)
    else:
        # Wide format
        price_cols = [col for col in price_df.columns if '_Price' in col]
        for col in price_cols:
            plt.plot(price_df['Week'], price_df[col], label=col)
    
    # Add vertical lines for shock weeks
    for week in shock_weeks:
        plt.axvline(x=week, color='red', linestyle='--', alpha=0.7, 
                    label='Market Shock' if week == shock_weeks[0] else '')
    
    plt.xlabel('Week')
    plt.ylabel('Price')
    plt.title('Price Evolution with Market Shocks')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/price_evolution_with_shocks.png")
    
    # 2. Bar chart of response metrics by agent
    metrics_to_plot = ['response_time', 'stabilization_time']
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 2, i+1)
        agents = list(shock_metrics[metric].keys())
        values = list(shock_metrics[metric].values())
        
        bars = plt.bar(agents, values)
        
        plt.xlabel('Agent')
        plt.ylabel('Weeks')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shock_response_metrics.png")
    
    # 3. Heatmap of all metrics
    plt.figure(figsize=(10, 8))
    
    # Create a DataFrame from the metrics
    heatmap_data = pd.DataFrame(index=list(shock_metrics['response_time'].keys()))
    
    for metric, values in shock_metrics.items():
        if isinstance(values, dict):  # Skip non-dict entries like 'shocks_enabled'
            heatmap_data[metric] = pd.Series(values)
    
    # Normalize the data for better visualization
    normalized_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) 
                                        if x.max() != x.min() else 0)
    
    sns.heatmap(normalized_data, cmap='viridis', annot=heatmap_data.round(2), fmt='.2f')
    plt.title('Shock Response Metrics by Agent (Normalized Colors, Actual Values)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shock_metrics_heatmap.png")
    
    # Save metrics to CSV
    heatmap_data.to_csv(f"{save_dir}/shock_response_metrics.csv")
    print(f"Saved shock response metrics to {save_dir}/shock_response_metrics.csv")

def inject_market_shocks(env, shock_weeks=[26, 52, 78], shock_magnitude=0.3, enable_shocks=True):
    """
    Function to inject market shocks into simulation environment.
    This would be called from the simulation code.
    
    Args:
        env: Market environment
        shock_weeks: List of weeks when shocks should occur
        shock_magnitude: Size of demand shock (e.g., 0.3 = 30% decrease)
        enable_shocks: Whether to enable market shocks
    
    Returns:
        bool: Whether a shock was injected
    """
    if not enable_shocks:
        return False
        
    current_week = env.current_week
    
    # Check if current week is a shock week
    if current_week in shock_weeks:
        print(f"🌊 MARKET SHOCK at Week {current_week}: Demand decreased by {shock_magnitude*100:.0f}%")
        # Apply temporary demand multiplier
        env.demand_shock_multiplier = 1.0 - shock_magnitude
        return True
    else:
        # Reset demand multiplier to normal
        env.demand_shock_multiplier = 1.0
        return False

# When integrated with the main framework, you would add this to the environment's
# demand calculation logic:
#
# def calculate_demand(self, ...):
#     original_demand = ... # Original demand calculation
#     
#     # Apply any active demand shocks
#     if hasattr(self, 'demand_shock_multiplier'):
#         final_demand = original_demand * self.demand_shock_multiplier
#     else:
#         final_demand = original_demand
#         
#     return final_demand