import numpy as np
import pandas as pd

def calculate_price_stability(price_df, episode_num, window=10):
    """Calculate price stability metrics for the final episode"""
    price_columns = [col for col in price_df.columns if 'Price' in col]
    stability_metrics = {}
    
    for col in price_columns:
        # Calculate rolling variance of price changes
        price_changes = price_df[col].pct_change().dropna()
        if len(price_changes) > window:
            # Use exponentially weighted variance to emphasize recent stability
            rolling_var = price_changes.ewm(span=window).var().iloc[-1]
        else:
            rolling_var = price_changes.var()
        
        # Lower values indicate more stability (scale to [0,1] range)
        stability = 1 - min(1, rolling_var * 100)
        stability_metrics[col] = stability
        
    return stability_metrics

def nash_equilibrium_proximity(price_df, last_n_weeks=20):
    """Measure how close agents are to a Nash equilibrium
    In Nash equilibrium, no agent can improve by changing only their strategy"""
    
    # Get only the final n weeks of data
    final_data = price_df.tail(min(last_n_weeks, len(price_df)))
    price_columns = [col for col in final_data.columns if 'Price' in col]
    
    # Group columns by agent
    agents = {}
    for col in price_columns:
        agent_id = col.split('_')[0]
        if agent_id not in agents:
            agents[agent_id] = []
        agents[agent_id].append(col)
    
    # Calculate price change volatility by agent
    nash_scores = {}
    for agent_id, cols in agents.items():
        # Calculate average week-to-week price changes
        changes = []
        for col in cols:
            weekly_changes = final_data[col].pct_change().abs().dropna()
            changes.extend(weekly_changes.tolist())
        
        # Lower score means closer to equilibrium (less price changes)
        nash_score = sum(changes) / max(1, len(changes))
        nash_scores[agent_id] = 1 - min(1, nash_score * 10)  # Scale to [0,1]
    
    overall_score = sum(nash_scores.values()) / max(1, len(nash_scores))
    return nash_scores, overall_score

def revenue_optimality_gap(returns, window=1):
    """Calculate how far each agent is from their peak revenue"""
    optimality_gaps = {}
    
    for agent_id, revenues in returns.items():
        # Find max revenue for this agent
        max_revenue = max(revenues) if revenues else 0
        
        # Calculate the average of last window episodes
        window_size = min(window, len(revenues))
        recent_avg = sum(revenues[-window_size:]) / window_size if revenues else 0
        
        # Calculate gap as percentage of maximum
        gap = (max_revenue - recent_avg) / max_revenue if max_revenue > 0 else 0
        optimality_gaps[agent_id] = gap
    
    return optimality_gaps

def price_revenue_elasticity(price_df, returns):
    """Analyze how price changes correlate with revenue changes"""
    results = {}
    
    # Group price columns by agent
    agents = {}
    price_columns = [col for col in price_df.columns if 'Price' in col]
    for col in price_columns:
        agent_id = col.split('_')[0]
        if agent_id not in agents:
            agents[agent_id] = []
        agents[agent_id].append(col)
    
    for agent_id, cols in agents.items():
        if agent_id not in returns or not returns[agent_id]:
            results[agent_id] = 0
            continue
        
        # Calculate average price per agent per time point
        avg_prices = []
        for i in range(len(price_df)):
            week_prices = [price_df.iloc[i][col] for col in cols]
            avg_prices.append(sum(week_prices) / len(week_prices))
        
        # For single episode case, use weekly price changes vs revenue
        if len(returns[agent_id]) == 1:
            # Use price trend correlation with revenue as elasticity
            first_price = avg_prices[0]
            last_price = avg_prices[-1]
            # Price elasticity: % change in price from start to end
            price_change = (last_price - first_price) / first_price if first_price > 0 else 0
            results[agent_id] = abs(price_change)
        else:
            # Multiple episodes - use episode data
            price_changes = []
            revenue_changes = []
            
            for ep in range(1, len(returns[agent_id])):
                if returns[agent_id][ep-1] > 0 and avg_prices[ep-1] > 0:
                    revenue_pct_change = (returns[agent_id][ep] - returns[agent_id][ep-1]) / returns[agent_id][ep-1]
                    price_pct_change = (avg_prices[ep] - avg_prices[ep-1]) / avg_prices[ep-1]
                    
                    price_changes.append(price_pct_change)
                    revenue_changes.append(revenue_pct_change)
            
            # Calculate correlation if enough data points
            if len(price_changes) > 1:
                correlation = np.corrcoef(price_changes, revenue_changes)[0,1]
                results[agent_id] = correlation
            else:
                results[agent_id] = 0
    
    return results

def global_vs_individual_optimality(price_df, returns, last_n_episodes=1):
    """Compare agents' individual revenue vs total market revenue"""
    
    # Calculate total market revenue in final episodes
    total_revenue = 0
    for agent_id in returns.keys():
        # Handle case with fewer episodes than requested
        n_episodes = min(last_n_episodes, len(returns[agent_id]))
        if n_episodes > 0:
            total_revenue += sum(returns[agent_id][-n_episodes:])
    
    # Calculate individual revenue share
    revenue_shares = {}
    for agent_id in returns.keys():
        n_episodes = min(last_n_episodes, len(returns[agent_id]))
        if n_episodes > 0:
            agent_revenue = sum(returns[agent_id][-n_episodes:])
            revenue_shares[agent_id] = agent_revenue / total_revenue if total_revenue > 0 else 0
        else:
            revenue_shares[agent_id] = 0
    
    # Calculate price similarity between agents
    agent_avg_prices = {}
    
    # Get average price per agent
    for col in [c for c in price_df.columns if 'Price' in c]:
        agent_id = col.split('_')[0]
        if agent_id not in agent_avg_prices:
            agent_avg_prices[agent_id] = []
        agent_avg_prices[agent_id].append(price_df[col].mean())
    
    for agent_id in agent_avg_prices:
        agent_avg_prices[agent_id] = sum(agent_avg_prices[agent_id]) / len(agent_avg_prices[agent_id]) if agent_avg_prices[agent_id] else 0
    
    # Calculate coefficient of variation of average prices
    prices = list(agent_avg_prices.values())
    price_cv = np.std(prices) / np.mean(prices) if prices and np.mean(prices) > 0 else 0
    
    # Return simple values rather than nested dictionaries
    return {
        'Agent1_revenue_share': float(revenue_shares.get('Agent1', 0)),
        'Agent2_revenue_share': float(revenue_shares.get('Agent2', 0)),
        'Agent3_revenue_share': float(revenue_shares.get('Agent3', 0)),
        'Agent4_revenue_share': float(revenue_shares.get('Agent4', 0)),
        'price_coefficient_variation': float(price_cv),
        'price_convergence': float(1 - min(1, price_cv))  # Higher means more convergence
    }

def social_welfare_metric(price_df, returns, last_n_episodes=1):
    """Calculate a social welfare metric that combines total revenue with fairness"""
    # Get final episode data
    final_revenues = {}
    for agent_id in returns:
        # Handle case with fewer episodes than requested
        n_episodes = min(last_n_episodes, len(returns[agent_id]))
        if n_episodes > 0:
            final_revenues[agent_id] = returns[agent_id][-n_episodes:]
        else:
            final_revenues[agent_id] = [0]
    
    # Calculate average revenue per agent
    avg_revenues = {agent_id: sum(revs)/len(revs) if revs else 0 
                   for agent_id, revs in final_revenues.items()}
    
    # Calculate total welfare
    total_welfare = sum(avg_revenues.values())
    
    # Calculate fairness using Gini coefficient (lower is more fair)
    values = list(avg_revenues.values())
    if values and sum(values) > 0:
        n = len(values)
        diff_sum = sum(abs(x - y) for x in values for y in values)
        gini = diff_sum / (2 * n * sum(values))
    else:
        gini = 0
    
    # Social welfare = total welfare × (1 - gini)
    # Higher value means both high total revenue and fair distribution
    social_welfare = total_welfare * (1 - gini)
    
    return {
        'total_revenue': total_welfare,
        'gini_coefficient': gini,
        'fairness': 1 - gini,  # Higher is more fair
        'social_welfare': social_welfare
    }