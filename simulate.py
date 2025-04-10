import os
import sys
import tensorflow as tf
from evaluation.eval_metrics import *

# Redirect stderr to nowhere (suppresses all stderr warnings)
old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')          # Suppress TensorFlow warning messages

# Now import everything else
from env.market_env import MarketEnv
from env.product import Product
from agents.rl_agent import RLAgent 
from agents.maddpg_agent import MADDPGAgent
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

def run_simulation(weeks=52, episodes=3, num_agents=4, use_maddpg=False):
    """Run multiple episodes of simulation for learning"""
    print("Starting simulation...")
    # Track agent learning performance across episodes
    episode_returns = {f'Agent{i}': [] for i in range(1, num_agents+1)}
    
    for episode in range(episodes):
        print(f"\n=== Episode {episode+1}/{episodes} ===")
        # print("Creating product portfolios...")
        
        # Create multiple product portfolios
        product_portfolios = [
            # Agent 1 
            [
                Product("PremiumA1", 15.0, category_cluster=1),
                Product("PremiumA2", 25.0, category_cluster=1),
                Product("PremiumA3", 35.0, category_cluster=2)
            ],
            # Agent 2
            [
                Product("BudgetB1", 10.0, category_cluster=1),
                Product("BudgetB2", 20.0, category_cluster=1),
                Product("BudgetB3", 30.0, category_cluster=2)
            ],
            # Agent 3
            [
                Product("MidC1", 10.0, category_cluster=1),
                Product("MidC2", 20.0, category_cluster=1),
                Product("MidC4", 30.0, category_cluster=2)
            ],
            # Agent 4
            [
                Product("PremiumA1", 8.0, category_cluster=1),
                Product("PremiumA2", 18.0, category_cluster=1),
                Product("PremiumA3", 28.0, category_cluster=2)
            ]
        ]
        
        # Create agents with different strategies/parameters
        agents = []
        for i in range(num_agents):
            if i < len(product_portfolios):  # Make sure we have a portfolio for this agent
                if use_maddpg:
                    agents.append(
                        MADDPGAgent(
                            f"Agent{i+1}", 
                            product_portfolios[i],
                            actor_lr=0.0005,        # Reduced from 0.001
                            critic_lr=0.001,        # Reduced from 0.002
                            discount_factor=0.98,   # Higher discount factor for longer-term focus
                            tau=0.005,              # Slower target network updates
                            exploration_noise=max(0.1, 0.3 * np.exp(-episode/15)),  # Exponential decay
                        )
                    )
                else:
                    agents.append(
                        RLAgent(
                            f"Agent{i+1}", 
                            product_portfolios[i],
                            learning_rate=0.05,
                            exploration_rate=max(0.2, 1.0 - episode*0.3),  # Decrease exploration over episodes
                            discount_factor=0.90 + i * 0.02  # Different time horizons
                        )
                    )
        
        print(f"Created {len(agents)} agents")
        
        # Create market environment
        print("Initializing market environment...")
        model_path = os.path.join(os.path.dirname(__file__), "models/lgbm_model.pkl")
        market = MarketEnv(agents=agents, model_path=model_path)
        
        # Run simulation
        print(f"Starting simulation for {weeks} weeks")
        weekly_results = []
        price_tracking = []  # To track price changes over time
        price_changes = []  # To track price changes
        
        try:
            for week in range(weeks):
                print(f"Episode {episode+1}/{episodes}, Week {week+1}/{weeks}", end='\r')
                # Store previous week prices for comparison
                previous_prices = {}
                for agent in agents:
                    for product_name, product in agent.products.items():
                        previous_prices[f"{agent.agent_id}_{product_name}"] = product.price
                
                demands, revenues = market.step()
                
                # Print current state (less frequently to reduce output volume)
                if week % 1 == 0:
                    print(f"Week {market.current_week}, Year {market.current_year}")
                    for agent_id in episode_returns.keys():
                        print(f"{agent_id} Revenue: ${revenues[agent_id]:.2f}")
                
                # Track prices for competing products
                price_data = {
                    'Week': market.current_week,
                    'Year': market.current_year
                }
                for agent in agents:
                    for product_name, product in agent.products.items():
                        price_data[f'{agent.agent_id}_{product_name}_Price'] = product.price
                price_tracking.append(price_data)
                
                # Track price changes
                for agent in agents:
                    for product_name, product in agent.products.items():
                        product_key = f"{agent.agent_id}_{product_name}"
                        if product_key in previous_prices:
                            pct_change = (product.price / previous_prices[product_key] - 1.0) * 100
                            price_changes.append({
                                'Week': market.current_week,
                                'Year': market.current_year, 
                                'Agent': agent.agent_id,
                                'Product': product_name,
                                'PriceChange': pct_change,
                                'Category': product.category_cluster
                            })
                
                # Collect data for analysis
                for agent_id, agent_demands in demands.items():
                    for product, demand in agent_demands.items():
                        agent = next(a for a in agents if a.agent_id == agent_id)
                        price = agent.products[product].price
                        weekly_results.append({
                            'Week': market.current_week,
                            'Year': market.current_year,
                            'Agent': agent_id,
                            'Product': product,
                            'Demand': demand,
                            'Price': price,
                            'Revenue': demand * price,
                            'Episode': episode + 1
                        })
        
        except Exception as e:
            print(f"\nERROR in episode {episode+1}, week {week+1}:")
            import traceback
            traceback.print_exc()
            break  # Exit the simulation loop but continue with the rest
        
        # Store overall episode performance
        for agent in agents:
            episode_returns[agent.agent_id].append(sum(agent.revenue_history))
        
        # Only save plots for final episode
        if episode == episodes - 1:
            # Create DataFrames for analysis
            results_df = pd.DataFrame(weekly_results)
            price_df = pd.DataFrame(price_tracking)
            changes_df = pd.DataFrame(price_changes)
            
            # Plot revenue over time
            plt.figure(figsize=(12, 6))
            for agent_id in episode_returns.keys():
                agent_data = results_df[results_df['Agent'] == agent_id]
                weekly_revenue = agent_data.groupby(['Week', 'Year'])['Revenue'].sum().reset_index()
                plt.plot(range(len(weekly_revenue)), weekly_revenue['Revenue'], label=agent_id)
            
            plt.xlabel('Week')
            plt.ylabel('Revenue')
            plt.title(f'Weekly Revenue by Agent (Episode {episode+1})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"simulation_results_revenue_ep{episode+1}.png")
            # Plot price competition for products in the same category
            plt.figure(figsize=(12, 6))
            price_columns = [col for col in price_df.columns if 'Price' in col]
            for column in price_columns:
                plt.plot(price_df['Week'], price_df[column], label=column)
            plt.xlabel('Week')
            plt.ylabel('Price')
            plt.title(f'Price Competition (Episode {episode+1})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"simulation_results_prices_ep{episode+1}.png")
            plt.savefig(f"simulation_results_prices_ep{episode+1}.png")
            
            # Create visualization for price changes
            # Option 1: Heatmap of price changes
            plt.figure(figsize=(14, 8))
            pivoted = changes_df.pivot_table(
                index=['Agent', 'Product'], 
                columns='Week', 
                values='PriceChange'
            )
            sns.heatmap(pivoted, cmap="RdBu_r", center=0, vmin=-10, vmax=10)
            plt.title('Price Changes by Week (%)')
            plt.tight_layout()
            plt.savefig(f"price_changes_heatmap_ep{episode+1}.png")
            
            # Option 2: Significant price changes as scatter plot
            plt.figure(figsize=(14, 6))
            # Filter for significant changes
            significant = changes_df[abs(changes_df['PriceChange']) > 2.0]
            
            for agent_id in changes_df['Agent'].unique():
                agent_data = significant[significant['Agent'] == agent_id]
                plt.scatter(agent_data['Week'], agent_data['PriceChange'], 
                            alpha=0.7, s=30, label=agent_id)
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Week')
            plt.ylabel('Price Change (%)')
            plt.title('Significant Price Changes (>2%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"significant_price_changes_ep{episode+1}.png")
    
    # Save the trained MADDPG models if using them
    if use_maddpg:
        print("Saving MADDPG models...")
        for agent in agents:
            agent.save()
    
    # Plot learning progress across episodes
    plt.figure(figsize=(10, 6))
    episodes_range = list(range(1, episodes+1))
    for agent_id in episode_returns.keys():
        plt.plot(episodes_range, episode_returns[agent_id], label=agent_id)
    plt.xlabel('Episode')
    plt.ylabel('Total Revenue')
    plt.title('Learning Progress Across Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("learning_progress.png")

    # At the end of simulation
    # Create final metrics for evaluation
    final_metrics = {}
    try:
        nash_scores, nash_overall = nash_equilibrium_proximity(price_df)
        
        final_metrics = {
            'price_stability': calculate_price_stability(price_df, episode),
            'nash_equilibrium': nash_scores,
            'nash_overall': nash_overall,
            'optimality_gap': revenue_optimality_gap(episode_returns),
            'price_revenue_elasticity': price_revenue_elasticity(price_df, episode_returns),
            'global_vs_individual': global_vs_individual_optimality(price_df, episode_returns),
            'social_welfare': social_welfare_metric(price_df, episode_returns)
        }
        
        # Print or save metrics
        print("\nConvergence & Optimality Metrics:")
        for metric_name, values in final_metrics.items():
            if isinstance(values, dict):
                print(f"\n{metric_name.replace('_', ' ').title()}:")
                for k, v in values.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"{metric_name.replace('_', ' ').title()}: {values:.4f}")
                
        # Save metrics to CSV
        metrics_df = pd.DataFrame()
        for metric_name, values in final_metrics.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    metrics_df.loc[metric_name, k] = v
            else:
                metrics_df.loc['overall', metric_name] = values
        
        metrics_df.to_csv(f"logs/optimization_metrics_ep{episode+1}.csv")
    except Exception as e:
        print(f"\nError calculating evaluation metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return episode_returns, final_metrics

if __name__ == "__main__":
    # Choose agent type: False for Q-Learning, True for MADDPG
    use_maddpg = True
    episode_returns, metrics = run_simulation(weeks=104, episodes=1, num_agents=4, use_maddpg=use_maddpg)
    print("\nSimulation complete!")
    print("\nTotal revenue by episode:")
    for episode in range(len(episode_returns['Agent1'])):
        print(f"Episode {episode+1}: ", end="")
        for agent_id in episode_returns.keys():
            print(f"{agent_id}: ${episode_returns[agent_id][episode]:.2f} ", end="")
        print()