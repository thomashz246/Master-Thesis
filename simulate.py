import os
import sys
import tensorflow as tf
from evaluation.eval_metrics import *
from agents.maddpg_coordinator import MADDPGCoordinator, JointReplayBuffer

old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
tf.get_logger().setLevel('ERROR')          

from env.market_env import MarketEnv
from env.product import Product
from agents.rl_agent import RLAgent 
from agents.maddpg_agent import MADDPGAgent
from agents.madqn_agent import MADQNAgent
from agents.qmix_agent import QMIXAgent
from agents.random_agent import RandomPricingAgent 
from agents.rule_agent import RuleBasedAgent
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

def run_simulation(weeks=52, episodes=3, num_agents=4, agent_type="maddpg", 
                  agent_types=None, rule_strategy="competitor_match", 
                  rule_strategies=None, save_dir=None, return_price_df=False,
                  enable_shocks=True):
    """Run multiple episodes of simulation for learning"""
    print("Starting simulation...")
    print(f"Market shocks: {'ENABLED' if enable_shocks else 'DISABLED'}")
    episode_returns = {f'Agent{i}': [] for i in range(1, num_agents+1)}
    
    all_price_data = []
    
    if agent_types and agent_type == "custom":
        print(f"Using custom agent configuration: {agent_types}")
    
    if rule_strategies:
        print(f"Using multiple rule strategies: {rule_strategies}")
    else:
        rule_strategies = [rule_strategy] * num_agents
        
    for episode in range(episodes):
        print(f"\n=== Episode {episode+1}/{episodes} ===")
        print("Creating product portfolios...")
        
        product_portfolios = [
            [
                Product("PremiumA1", 50.0, category_cluster=1),
                Product("PremiumA2", 32.0, category_cluster=2),
                Product("PremiumA3", 28.0, category_cluster=3),
                Product("PremiumA4", 40.0, category_cluster=5),
                Product("LuxuryA5", 55.4, category_cluster=10)
            ],
            [
                Product("PremiumA1", 50.0, category_cluster=1),
                Product("PremiumA2", 32.0, category_cluster=2),
                Product("PremiumA3", 28.0, category_cluster=3),
                Product("PremiumA4", 40.0, category_cluster=5),
                Product("LuxuryA5", 55.4, category_cluster=10)
            ],
            [
                Product("PremiumA1", 50.0, category_cluster=1),
                Product("PremiumA2", 32.0, category_cluster=2),
                Product("PremiumA3", 28.0, category_cluster=3),
                Product("PremiumA4", 40.0, category_cluster=5),
                Product("LuxuryA5", 55.4, category_cluster=10)
            ],
            [
                Product("PremiumA1", 50.0, category_cluster=1),
                Product("PremiumA2", 32.0, category_cluster=2),
                Product("PremiumA3", 28.0, category_cluster=3),
                Product("PremiumA4", 40.0, category_cluster=5),
                Product("LuxuryA5", 55.4, category_cluster=10)
            ]
        ]
        
        agents = []
        try:
            for i in range(min(num_agents, len(product_portfolios))):
                print(f"Creating agent {i+1}...")
                current_agent_type = agent_types[i] if agent_types else agent_type if i == 0 else "rule"
                
                if current_agent_type == "maddpg":
                    print("Creating MADDPG agent...")
                    maddpg_agent = MADDPGAgent(
                        f"Agent{i+1}", 
                        product_portfolios[i],
                        actor_lr=0.0001,
                        critic_lr=0.0005,  
                        discount_factor=0.95,
                        tau=0.001,
                        exploration_noise=max(0.1, 0.3 * np.exp(-episode/15)),
                    )
                    maddpg_agent.debug_mode = True
                    agents.append(maddpg_agent)
                elif current_agent_type == "madqn":
                    print("Creating MADQN agent...")
                    madqn_agent = MADQNAgent(
                        f"Agent{i+1}", 
                        product_portfolios[i],
                        learning_rate=0.001,
                        discount_factor=0.95,
                        exploration_rate=max(0.1, 0.5 * np.exp(-episode/15)),
                        exploration_decay=0.995,
                        min_exploration=0.05,
                        batch_size=64,
                        update_target_every=5
                    )
                    agents.append(madqn_agent)
                elif current_agent_type == "qmix":
                    print("Creating QMIX agent...")
                    qmix_agent = QMIXAgent(
                        f"Agent{i+1}", 
                        product_portfolios[i],
                        learning_rate=0.001,
                        discount_factor=0.95,
                        exploration_rate=max(0.1, 0.5 * np.exp(-episode/15)),
                        exploration_decay=0.995,
                        min_exploration=0.05,
                        batch_size=64,
                        update_target_every=5,
                        num_agents=4
                    )
                    agents.append(qmix_agent)
                elif current_agent_type == "random":
                    print("Creating random agent...")
                    agents.append(RandomPricingAgent(f"Agent{i+1}", product_portfolios[i]))
                else:  
                    current_rule_strategy = rule_strategies[i] if i < len(rule_strategies) else rule_strategy
                    print(f"Creating rule-based agent with strategy: {current_rule_strategy}")
                    agents.append(
                        RuleBasedAgent(
                            f"Agent{i+1}", 
                            product_portfolios[i],
                            strategy=current_rule_strategy,
                            markup_pct=0.20,
                            undercut_pct=0.05,
                            demand_threshold=0.10,
                            seasonal_boost=0.15
                        )
                    )
        except Exception as e:
            print(f"ERROR creating agents: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"Created {len(agents)} agents")
        
        maddpg_agents = [agent for agent in agents if isinstance(agent, MADDPGAgent)]
        coordinator = None
        if maddpg_agents and agent_type == "maddpg":
            coordinator = MADDPGCoordinator(maddpg_agents)
            print(f"Created MADDPG coordinator for {len(maddpg_agents)} agents")
        
        print("Initializing market environment...")
        model_path = os.path.join(os.path.dirname(__file__), "models/lgbm_model.pkl")
        market = MarketEnv(agents=agents, model_path=model_path)
        market.enable_shocks = enable_shocks
        
        print(f"Starting simulation for {weeks} weeks")
        weekly_results = []
        price_tracking = [] 
        price_changes = []  
      
        try:
            for week in range(weeks):
                print(f"Episode {episode+1}/{episodes}, Week {week+1}/{weeks}", end='\r')
                
                previous_prices = {}
                for agent in agents:
                    for product_name, product in agent.products.items():
                        previous_prices[f"{agent.agent_id}_{product_name}"] = product.price
                
                market_observations = market.get_market_observations()
                
                if coordinator:
                    joint_states = []
                    for agent in maddpg_agents:
                        state = agent.get_state_representation(list(agent.products.keys())[0], market_observations)
                        joint_states.append(state)
                    
                    current_joint_states = np.array(joint_states)
                
                for agent in agents:
                    agent.act(market.current_week, market.current_year, market.is_holiday_season, market_observations)
                
                if coordinator:
                    joint_actions = []
                    for agent in maddpg_agents:
                        if hasattr(agent, 'previous_action'):
                            joint_actions.append(agent.previous_action)
                        else:
                            joint_actions.append(np.zeros((1,), dtype=np.float32))
                    
                    current_joint_actions = np.array(joint_actions)
                
                demands, revenues = market.step()
                
                if week % 1 == 0:
                    print(f"Week {week+1}")
                    for agent_id, revenue in revenues.items():
                        print(f"  {agent_id}: ${revenue:.2f}")

                if coordinator:
                    next_market_observations = market.get_market_observations()       
                    next_joint_states = []
                    joint_rewards = []
                    
                    for agent in maddpg_agents:
                        next_state = agent.get_state_representation(list(agent.products.keys())[0], next_market_observations)
                        next_joint_states.append(next_state)
                        
                        reward = revenues.get(agent.agent_id, 0) / 1000.0 
                        joint_rewards.append(reward)
                    
                    joint_rewards_array = np.array(joint_rewards).reshape(-1, 1)
                    coordinator.store_transition(
                        current_joint_states,
                        current_joint_actions,
                        joint_rewards_array,
                        np.array(next_joint_states),
                        False 
                    )
                    coordinator.learn()
                
                price_data = {
                    'Week': market.current_week,
                    'Year': market.current_year,
                    'Episode': episode + 1  
                }
                for agent in agents:
                    for product_name, product in agent.products.items():
                        price_data[f'{agent.agent_id}_{product_name}_Price'] = product.price
                price_tracking.append(price_data)
                
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
            break   
        for agent in agents:
            episode_returns[agent.agent_id].append(sum(agent.revenue_history))
        
        if episode == episodes - 1:
            results_df = pd.DataFrame(weekly_results)
            price_df = pd.DataFrame(price_tracking)
            changes_df = pd.DataFrame(price_changes)
            
            price_df['ContinuousWeek'] = price_df['Week'] + (price_df['Year'] - price_df['Year'].min()) * 52
            
            plt.figure(figsize=(12, 6))
            for agent_id in episode_returns.keys():
                agent_data = results_df[results_df['Agent'] == agent_id]
                agent_data['ContinuousWeek'] = agent_data['Week'] + (agent_data['Year'] - agent_data['Year'].min()) * 52
                weekly_revenue = agent_data.groupby('ContinuousWeek')['Revenue'].sum().reset_index()
                plt.plot(weekly_revenue['ContinuousWeek'], weekly_revenue['Revenue'], label=agent_id)
            
            plt.xlabel('Week')
            plt.ylabel('Revenue')
            plt.title(f'Weekly Revenue by Agent (Episode {episode+1})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 104)
            plt.xticks(np.arange(0, 105, 13))  
            if save_dir:
                plt.savefig(f"{save_dir}/simulation_results_revenue_ep{episode+1}.png")
            else:
                plt.savefig(f"simulation_results_revenue_ep{episode+1}.png")
            
            plt.figure(figsize=(12, 6))
            price_columns = [col for col in price_df.columns if 'Price' in col]
            for column in price_columns:
                plt.plot(price_df['ContinuousWeek'], price_df[column], label=column)
            plt.xlabel('Week')
            plt.ylabel('Price')
            plt.title(f'Price Competition (Episode {episode+1})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 104)  
            plt.xticks(np.arange(0, 105, 13))  
            if save_dir:
                plt.savefig(f"{save_dir}/simulation_results_prices_ep{episode+1}.png")
            else:
                plt.savefig(f"simulation_results_prices_ep{episode+1}.png")
            
            plt.figure(figsize=(14, 8))
            pivoted = changes_df.pivot_table(
                index=['Agent', 'Product'], 
                columns='Week', 
                values='PriceChange'
            )
            sns.heatmap(pivoted, cmap="RdBu_r", center=0, vmin=-10, vmax=10)
            plt.title('Price Changes by Week (%)')
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/price_changes_heatmap_ep{episode+1}.png")
            else:
                plt.savefig(f"price_changes_heatmap_ep{episode+1}.png")
            
            plt.figure(figsize=(14, 6))
            significant = changes_df[abs(changes_df['PriceChange']) > 2.0]
            significant['ContinuousWeek'] = significant['Week'] + (significant['Year'] - significant['Year'].min()) * 52

            for agent_id in significant['Agent'].unique():
                agent_data = significant[significant['Agent'] == agent_id]
                plt.scatter(agent_data['ContinuousWeek'], agent_data['PriceChange'], 
                            alpha=0.7, s=30, label=agent_id)

            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Week (Continuous)')
            plt.ylabel('Price Change (%)')
            plt.title('Significant Price Changes (>2%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if all(agent_type == "rule" for agent_type in (agent_types if agent_types else ["rule"] * num_agents)) and rule_strategies and len(set(rule_strategies)) > 1:
                strategy_performance = []
                for i, agent in enumerate(agents):
                    if i < len(rule_strategies):
                        strategy = rule_strategies[i]
                    else:
                        strategy = rule_strategy
                        
                    total_revenue = sum(agent.revenue_history)
                    average_weekly_revenue = total_revenue / weeks
                    
                    strategy_performance.append({
                        'Agent': agent.agent_id,
                        'Strategy': strategy,
                        'Total Revenue': total_revenue,
                        'Average Weekly Revenue': average_weekly_revenue
                    })
                
                strategy_df = pd.DataFrame(strategy_performance)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Agent', y='Total Revenue', hue='Strategy', data=strategy_df)
                plt.title('Revenue by Rule-Based Pricing Strategy')
                plt.xlabel('Agent')
                plt.ylabel('Total Revenue')
                plt.grid(True, alpha=0.3)
                if save_dir:
                    plt.savefig(f"{save_dir}/rule_strategy_comparison_ep{episode+1}.png")
                else:
                    plt.savefig(f"rule_strategy_comparison_ep{episode+1}.png")
    
        all_episode_prices = pd.DataFrame(price_tracking)
        all_price_data.append(all_episode_prices)
    
    if agent_type in ["maddpg", "madqn", "qmix"]:
        print(f"Saving {agent_type.upper()} models...")
        try:
            for idx, agent in enumerate(agents):
                if isinstance(agent, MADDPGAgent): 
                    print(f"Saving model for {agent.agent_id}...")
                    agent.save()
            print("All models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
            traceback.print_exc()
    
    plt.figure(figsize=(10, 6))
    episodes_range = list(range(1, episodes+1))
    for agent_id in episode_returns.keys():
        plt.plot(episodes_range, episode_returns[agent_id], label=agent_id)
    plt.xlabel('Episode')
    plt.ylabel('Total Revenue')
    plt.title('Learning Progress Across Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir:
        plt.savefig(f"{save_dir}/learning_progress.png")
    else:
        plt.savefig("learning_progress.png")

    final_metrics = {}
    try:
        print("\nCalculating evaluation metrics...")
        
        if 'price_df' not in locals() or price_df.empty:
            print("ERROR: price_df is not defined or is empty")
            results_df = pd.DataFrame(weekly_results)
            price_df = pd.DataFrame(price_tracking)
            changes_df = pd.DataFrame(price_changes)
            print(f"Recreated price_df with shape: {price_df.shape}")
        
        print(f"Price dataframe shape: {price_df.shape}")
        print(f"Price dataframe columns: {price_df.columns.tolist()}")
        print(f"Episode returns: {episode_returns}")
        
        print("Checking if required functions are available...")
        print(f"nash_equilibrium_proximity available: {'nash_equilibrium_proximity' in globals()}")
        print(f"calculate_price_stability available: {'calculate_price_stability' in globals()}")
        
        print("Calculating Nash equilibrium proximity...")
        nash_scores, nash_overall = nash_equilibrium_proximity(price_df)
        print("Nash equilibrium metrics calculated successfully")
        
        print("Calculating price stability...")
        price_stability = calculate_price_stability(price_df, episode)
        print("Price stability calculated successfully")
        
        print("Calculating optimality gap...")
        opt_gap = revenue_optimality_gap(episode_returns, window=5)
        print("Optimality gap calculated successfully")
        
        print("Calculating price-revenue elasticity...")
        elasticity = price_revenue_elasticity(price_df, episode_returns)
        print("Elasticity calculated successfully")
        
        print("Calculating global vs individual optimality...")
        global_vs_ind = global_vs_individual_optimality(price_df, episode_returns)
        print("Global vs individual optimality calculated successfully")
        
        print("Calculating social welfare metric...")
        welfare = social_welfare_metric(price_df, episode_returns)
        print("Social welfare calculated successfully")
        
        final_metrics = {
            'price_stability': price_stability,
            'nash_equilibrium': nash_scores,
            'nash_overall': nash_overall,
            'optimality_gap': opt_gap,
            'price_revenue_elasticity': elasticity,
            'global_vs_individual': global_vs_ind,
            'social_welfare': welfare
        }
        
        metrics_df = pd.DataFrame()
        for metric_name, values in final_metrics.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    metrics_df.loc[metric_name, k] = v
            else:
                metrics_df.loc['overall', metric_name] = values
        
        os.makedirs("logs", exist_ok=True)
        metrics_df.to_csv(f"logs/optimization_metrics_ep{episode+1}.csv")
        print(f"Saved metrics to logs/optimization_metrics_ep{episode+1}.csv")
        
    except Exception as e:
        print(f"\nError calculating evaluation metrics: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()
    
    print(f"Agent type: {agent_type}")
    print(f"Agent has debug_mode attribute: {hasattr(agents[0], 'debug_mode')}")
    print(f"Agent has actor_losses attribute: {hasattr(agents[0], 'actor_losses')}")
    
    if hasattr(agents[0], 'actor_losses'):
        print(f"Length of actor_losses: {len(agents[0].actor_losses)}")
    else:
        print("WARNING: MADDPG agent doesn't have actor_losses attribute")
        agents[0].actor_losses = []
        agents[0].critic_losses = []
        agents[0].actions_before_noise = []
        agents[0].actions_after_noise = []
        print("Added missing attributes to agent")

    if agent_type == "maddpg" and hasattr(agents[0], 'actor_losses'):
        print(f"Preparing to plot learning metrics...")
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(agents[0].actor_losses if len(agents[0].actor_losses) > 0 else [0])
        plt.title('Actor Loss (empty if no data)')
        plt.xlabel('Training Step')
        plt.grid(True, alpha=0.3)
    
    if agent_type == "maddpg" and hasattr(agents[0], 'actor_losses') and len(agents[0].actor_losses) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(agents[0].actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Training Step')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(agents[0].critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Training Step')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        if len(agents[0].actions_before_noise) > 0:
            plt.plot(agents[0].actions_before_noise[-100:], label='Before Noise')
            plt.plot(agents[0].actions_after_noise[-100:], label='After Noise', alpha=0.7)
            plt.title('Recent Actions (Last 100)')
            plt.xlabel('Step')
            plt.ylabel('Action Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        episodes_x = np.arange(1, episodes+1)
        exploration_values = [max(0.1, 0.3 * np.exp(-ep/15)) for ep in range(episodes)]
        plt.plot(episodes_x, exploration_values)
        plt.title('Exploration Noise Decay')
        plt.xlabel('Episode')
        plt.ylabel('Noise Magnitude')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/maddpg_learning_metrics.png")
        else:
            plt.savefig("maddpg_learning_metrics.png")
        print("Saved MADDPG learning metrics visualization to maddpg_learning_metrics.png")
    
    if return_price_df and all_price_data:
        combined_price_df = pd.concat(all_price_data, ignore_index=True)
        return episode_returns, final_metrics, combined_price_df
    else:
        return episode_returns, final_metrics

if __name__ == "__main__":
    try:
        episode_returns, metrics = run_simulation(weeks=52, episodes=3)
        
        episode = 2 
        print(f"Final Episode Results: ", end="")
        for agent_id in episode_returns.keys():
            print(f"{agent_id}: ${episode_returns[agent_id][episode]:.2f} ", end="")
        print()
            
        print("\nConvergence & Optimality Metrics:")
        if metrics:
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    print(f"\n{metric_name.replace('_', ' ').title()}:")
                    for k, v in values.items():
                        print(f"  {k}: {v:.4f}")
                else:
                    print(f"{metric_name.replace('_', ' ').title()}: {values:.4f}")
        else:
            print("No metrics were calculated successfully.")
    except Exception as e:
        print("\nCRITICAL ERROR in main execution:")
        import traceback
        traceback.print_exc()
