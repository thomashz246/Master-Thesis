import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from simulate import run_simulation
import traceback

def run_experiment_batch():
    """Run all experimental configurations and save results"""
    
    # Define the experimental configurations
    configurations = [
    {
        "name": "Config A - All Rule-Based",
        "description": "All 4 agents using rule-based strategy",
        "agent_types": ["rule", "rule", "rule", "rule"],
        "rule_strategies": ["competitor_match", "historical_anchor", "demand_responsive", "seasonal_pricing"]
    },
    {
        "name": "Config B - All MADDPG",
        "description": "All 4 agents using MADDPG",
        "agent_types": ["maddpg", "maddpg", "maddpg", "maddpg"],
        "rule_strategy": "competitor_match"
    },
    {
        "name": "Config C - All MADQN",
        "description": "All 4 agents using MADQN",
        "agent_types": ["madqn", "madqn", "madqn", "madqn"],
        "rule_strategy": "competitor_match"
    },
    {
        "name": "Config D - MADDPG vs MADQN",
        "description": "Mixed MARL setup: 2 MADDPG agents, 2 MADQN agents",
        "agent_types": ["maddpg", "maddpg", "madqn", "madqn"],
        "rule_strategy": "competitor_match"
    },
    {
        "name": "Config E - MADQN vs Rule-Based",
        "description": "1 MADQN agent with 3 rule-based agents",
        "agent_types": ["madqn", "rule", "rule", "rule"],
        "rule_strategy": "competitor_match"
    },
    {
        "name": "Config F - All QMIX",
        "description": "All 4 agents using QMIX",
        "agent_types": ["qmix", "qmix", "qmix", "qmix"],
        "rule_strategy": "competitor_match"
    },
    {
        "name": "Config G - One MADDPG",
        "description": "1 MADDPG agent with 3 rule-based agents",
        "agent_types": ["maddpg", "rule", "rule", "rule"],
        "rule_strategy": "competitor_match"
    },
    {
        "name": "Config H - MADDPG vs QMIX",
        "description": "Mixed MARL setup: 2 MADDPG agents, 2 QMIX agents",
        "agent_types": ["maddpg", "maddpg", "qmix", "qmix"],
        "rule_strategy": "competitor_match"
    }
]
    
    # Create experiment results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_results/experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{results_dir}/experiment_config.json", "w") as f:
        json.dump(configurations, f, indent=4)
    
    # Run each configuration
    all_results = {}
    for idx, config in enumerate(configurations):
        print(f"\n{'='*80}")
        print(f"Running Experiment {idx+1}/{len(configurations)}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")
        
        config_dir = f"{results_dir}/config_{chr(65+idx)}"
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            # Run the simulation with this configuration
            rule_strategies = config.get("rule_strategies")
            
            episode_returns, metrics = run_experiment(
                config["agent_types"],
                config.get("rule_strategy", "competitor_match"),
                rule_strategies=rule_strategies,
                weeks=104,       # Adjust as needed
                episodes=10,     # Adjust as needed
                save_dir=config_dir
            )
            
            # Save results
            all_results[config["name"]] = {
                "returns": episode_returns,
                "metrics": metrics
            }
            
            # Save individual results as CSV
            returns_df = pd.DataFrame(episode_returns)
            returns_df.to_csv(f"{config_dir}/episode_returns.csv")
            
            # Save metrics
            if metrics:
                metrics_df = pd.DataFrame()
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):
                        for k, v in values.items():
                            metrics_df.loc[metric_name, k] = v
                    else:
                        metrics_df.loc['overall', metric_name] = values
                metrics_df.to_csv(f"{config_dir}/metrics.csv")
            
            print(f"Configuration {config['name']} completed successfully")
            
        except Exception as e:
            print(f"Error running configuration {config['name']}: {e}")
            traceback.print_exc()
            print("Moving to next configuration...")
    
    # Create a consolidated returns dataset across all configs and episodes
    all_returns_data = []
    for config_name, result in all_results.items():
        config_key = config_name.split(" - ")[0]
        for agent_id, returns in result["returns"].items():
            for episode_idx, return_val in enumerate(returns):
                all_returns_data.append({
                    "Configuration": config_name,
                    "ConfigKey": config_key,
                    "Agent": agent_id,
                    "Episode": episode_idx + 1,
                    "Return": return_val
                })
    
    if all_returns_data:
        all_returns_df = pd.DataFrame(all_returns_data)
        all_returns_df.to_csv(f"{results_dir}/all_returns_by_episode.csv", index=False)
        print(f"Saved all returns data to {results_dir}/all_returns_by_episode.csv")
    
    # Generate comparative visualizations
    try:
        create_comparative_visuals(all_results, results_dir)
    except Exception as e:
        print(f"Error generating comparative visuals: {e}")
        traceback.print_exc()
    
    return all_results, results_dir

def run_experiment(agent_types, rule_strategy="competitor_match", 
                  rule_strategies=None, weeks=52, episodes=5, save_dir=None):
    """Run a single experiment with specified agent configuration"""
    
    from simulate import run_simulation
    from evaluation.eval_metrics import generate_evaluation_plots
    
    # Run the simulation as before
    episode_returns, metrics, price_df = run_simulation(
        weeks=weeks,
        episodes=episodes,
        num_agents=len(agent_types),
        agent_type="custom",
        agent_types=agent_types,
        rule_strategy=rule_strategy,
        rule_strategies=rule_strategies,
        save_dir=save_dir,
        return_price_df=True  # Add this parameter to return price data
    )
    
    # Generate the new evaluation plots
    if save_dir and price_df is not None:
        additional_metrics = generate_evaluation_plots(price_df, episode_returns, save_dir)
        
        # Merge the additional metrics into the existing metrics
        if metrics is None:
            metrics = {}
        metrics.update(additional_metrics)
    
    # Save detailed time series data
    if save_dir:
        # Save returns per episode as time series
        episode_returns_df = pd.DataFrame(episode_returns)
        episode_returns_df.to_csv(f"{save_dir}/episode_returns.csv")
        
        # If available, save market shares evolution
        if metrics and 'market_shares' in metrics:
            market_shares = metrics['market_shares']
            market_shares_df = pd.DataFrame({
                agent_id: shares for agent_id, shares in market_shares.items()
            })
            market_shares_df.to_csv(f"{save_dir}/market_shares_evolution.csv")
        
        # If available, save fairness evolution
        if metrics and 'fairness_over_time' in metrics:
            fairness_df = pd.DataFrame({
                'Episode': range(1, len(metrics['fairness_over_time']) + 1),
                'Fairness_Index': metrics['fairness_over_time']
            })
            fairness_df.to_csv(f"{save_dir}/fairness_evolution.csv")
        
        # Save the full price dataframe for detailed analysis
        if price_df is not None:
            price_df.to_csv(f"{save_dir}/price_evolution.csv")
            
            # Also create a summary of price stats by episode
            price_stats = []
            for episode in price_df['Episode'].unique():
                episode_data = price_df[price_df['Episode'] == episode]
                for agent_product in episode_data.columns:
                    if agent_product not in ['Week', 'Episode']:
                        price_series = episode_data[agent_product].dropna()
                        if not price_series.empty:
                            price_stats.append({
                                'Episode': episode,
                                'AgentProduct': agent_product,
                                'Mean': price_series.mean(),
                                'Min': price_series.min(),
                                'Max': price_series.max(),
                                'StdDev': price_series.std(),
                                'CoeffVar': price_series.std() / price_series.mean() if price_series.mean() != 0 else None
                            })
            
            if price_stats:
                pd.DataFrame(price_stats).to_csv(f"{save_dir}/price_statistics.csv", index=False)
    
    return episode_returns, metrics

def create_comparative_visuals(all_results, results_dir):
    """Create comparative visualizations across configurations and save all metrics to CSV"""
    
    # Compare final episode performance
    plt.figure(figsize=(12, 8))
    
    # Extract last episode returns for all configs and agents
    bar_data = []
    for config_name, result in all_results.items():
        for agent_id, returns in result["returns"].items():
            if returns:  # Make sure there's data
                bar_data.append({
                    "Config": config_name.split(" - ")[0],
                    "Agent": agent_id,
                    "Final Return": returns[-1]
                })
    
    # Create DataFrame for easier plotting
    bar_df = pd.DataFrame(bar_data)
    
    # Plot grouped bar chart
    if not bar_df.empty:
        pivot_df = bar_df.pivot(index="Config", columns="Agent", values="Final Return")
        pivot_df.plot(kind="bar", figsize=(12, 8))
        plt.title("Final Episode Returns by Configuration")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/comparative_returns.png")
    
    # Compare key metrics across configurations
    if all(result.get("metrics") for result in all_results.values()):
        # Extract common metrics
        metrics_data = []
        for config_name, result in all_results.items():
            if "metrics" in result and result["metrics"]:
                # For scalar metrics
                for metric_name, value in result["metrics"].items():
                    if not isinstance(value, dict):
                        metrics_data.append({
                            "Config": config_name.split(" - ")[0],
                            "Metric": metric_name,
                            "Value": value
                        })
        
        # Create DataFrame and plot
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            plt.figure(figsize=(14, 10))
            for i, metric in enumerate(metrics_df["Metric"].unique()):
                plt.subplot(2, 2, i+1)
                metric_data = metrics_df[metrics_df["Metric"] == metric]
                plt.bar(metric_data["Config"], metric_data["Value"])
                plt.title(metric.replace("_", " ").title())
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/comparative_metrics.png")
    
    # Add new visualization - Compare agent types within each configuration
    agent_type_data = []
    
    for config_name, result in all_results.items():
        # Extract the agent types from the configuration name
        config_key = config_name.split(" - ")[0]
        
        # Create mapping of agents to their types based on configuration
        agent_types = {}
        if "One MADDPG" in config_name:
            agent_types = {"Agent1": "MADDPG", "Agent2": "Rule-Based", "Agent3": "Rule-Based", "Agent4": "Rule-Based"}
        elif "Two MADDPG" in config_name:
            agent_types = {"Agent1": "MADDPG", "Agent2": "MADDPG", "Agent3": "Rule-Based", "Agent4": "Rule-Based"}
        elif "Three MADDPG" in config_name:
            agent_types = {"Agent1": "MADDPG", "Agent2": "MADDPG", "Agent3": "MADDPG", "Agent4": "Rule-Based"}
        elif "All MADDPG" in config_name:
            agent_types = {"Agent1": "MADDPG", "Agent2": "MADDPG", "Agent3": "MADDPG", "Agent4": "MADDPG"}
        elif "One MADQN" in config_name:
            agent_types = {"Agent1": "MADQN", "Agent2": "Rule-Based", "Agent3": "Rule-Based", "Agent4": "Rule-Based"}
        elif "All MADQN" in config_name:
            agent_types = {"Agent1": "MADQN", "Agent2": "MADQN", "Agent3": "MADQN", "Agent4": "MADQN"}
        elif "One QMIX" in config_name:
            agent_types = {"Agent1": "QMIX", "Agent2": "Rule-Based", "Agent3": "Rule-Based", "Agent4": "Rule-Based"}
        elif "All QMIX" in config_name:
            agent_types = {"Agent1": "QMIX", "Agent2": "QMIX", "Agent3": "QMIX", "Agent4": "QMIX"}
        elif "RL Competition" in config_name:
            agent_types = {"Agent1": "MADDPG", "Agent2": "MADQN", "Agent3": "QMIX", "Agent4": "Rule-Based"}
        elif "All Rule-Based" in config_name:
            agent_types = {"Agent1": "Rule-Based", "Agent2": "Rule-Based", "Agent3": "Rule-Based", "Agent4": "Rule-Based"}
        elif "All Random" in config_name:
            agent_types = {"Agent1": "Random", "Agent2": "Random", "Agent3": "Random", "Agent4": "Random"}
        
        # Group by agent type
        type_returns = {"MADDPG": [], "MADQN": [], "QMIX": [], "Rule-Based": [], "Random": []}
        
        for agent_id, returns in result["returns"].items():
            if returns:  # Make sure there's data
                agent_type = agent_types.get(agent_id, "Unknown")
                if agent_type in type_returns:
                    # Use the last episode return
                    type_returns[agent_type].append(returns[-1])
        
        # Calculate averages for each agent type (if data exists)
        for agent_type, values in type_returns.items():
            if values:  # Only add if we have data for this type
                avg_return = sum(values) / len(values)
                agent_type_data.append({
                    "Config": config_key,
                    "Agent Type": agent_type,
                    "Average Return": avg_return,
                    "Count": len(values)
                })
    
    # Create DataFrame and plot
    if agent_type_data:
        type_df = pd.DataFrame(agent_type_data)
        
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        pivot_type_df = type_df.pivot(index="Config", columns="Agent Type", values="Average Return")
        
        # Fill missing values with 0
        pivot_type_df = pivot_type_df.fillna(0)
        
        # Plot with a more visually distinct color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
        ax = pivot_type_df.plot(kind="bar", figsize=(14, 8), color=colors)
        
        # Add count labels above the bars
        for i, config in enumerate(pivot_type_df.index):
            for j, agent_type in enumerate(pivot_type_df.columns):
                if pivot_type_df.loc[config, agent_type] > 0:
                    # Get count for this config and agent type
                    count = type_df[(type_df["Config"] == config) & 
                                    (type_df["Agent Type"] == agent_type)]["Count"].values[0]
                    # Add label for count
                    ax.text(i + (j-len(pivot_type_df.columns)/2+0.5)*0.25, 
                            pivot_type_df.loc[config, agent_type] + 50,  # Offset above bar
                            f"n={count}", 
                            ha='center', va='bottom',
                            fontweight='bold')
        
        plt.title("Average Return by Agent Type in Each Configuration")
        plt.ylabel("Average Revenue")
        plt.xlabel("Configuration")
        plt.legend(title="Agent Type")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/comparative_returns_per_agent_type.png")
        
        # Also add a stacked version to see total market revenue
        plt.figure(figsize=(14, 8))
        pivot_type_df.plot(kind="bar", figsize=(14, 8), stacked=True)
        plt.title("Total Market Revenue by Agent Type in Each Configuration")
        plt.ylabel("Revenue")
        plt.xlabel("Configuration")
        plt.legend(title="Agent Type")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/stacked_returns_per_agent_type.png")
    
    # Save ALL metrics in a comprehensive DataFrame
    all_metrics_rows = []
    
    for config_name, result in all_results.items():
        if "metrics" in result and result["metrics"]:
            config_key = config_name.split(" - ")[0]
            base_row = {"Configuration": config_name, "ConfigKey": config_key}
            
            # Add final return values by agent
            for agent_id, returns in result["returns"].items():
                if returns:
                    base_row[f"FinalReturn_{agent_id}"] = returns[-1]
            
            # Add all metrics (both top-level and nested)
            for metric_name, value in result["metrics"].items():
                if isinstance(value, dict):
                    for sub_metric_name, sub_value in value.items():
                        # Create a unique key for sub-metrics
                        base_row[f"{metric_name}_{sub_metric_name}"] = sub_value
                else:
                    base_row[metric_name] = value
            
            all_metrics_rows.append(base_row)
    
    if all_metrics_rows:
        # Create comprehensive DataFrame with all metrics for all configurations
        comprehensive_metrics_df = pd.DataFrame(all_metrics_rows)
        
        # Save to CSV
        comprehensive_metrics_df.to_csv(f"{results_dir}/all_configurations_metrics.csv", index=False)
        print(f"Saved comprehensive metrics to {results_dir}/all_configurations_metrics.csv")
        
        # Also save a pivot table with metrics by configuration for easier analysis
        # This creates a table with configurations as rows and metrics as columns
        pivot_metrics = pd.DataFrame(index=[row["ConfigKey"] for row in all_metrics_rows])
        
        # For each possible metric column, create a pivot column
        all_metric_cols = set()
        for row in all_metrics_rows:
            all_metric_cols.update(key for key in row.keys() if key not in ["Configuration", "ConfigKey"])
            
        for metric_col in all_metric_cols:
            for row in all_metrics_rows:
                if metric_col in row:
                    pivot_metrics.loc[row["ConfigKey"], metric_col] = row[metric_col]
        
        pivot_metrics.to_csv(f"{results_dir}/pivot_metrics_by_config.csv")
        print(f"Saved pivoted metrics to {results_dir}/pivot_metrics_by_config.csv")
    
    # Create a summary report
    with open(f"{results_dir}/experiment_summary.txt", "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=================\n\n")
        
        for config_name, result in all_results.items():
            f.write(f"{config_name}\n")
            f.write("-" * len(config_name) + "\n")
            
            # Write returns
            f.write("Final Episode Returns:\n")
            for agent_id, returns in result["returns"].items():
                if returns:
                    f.write(f"  {agent_id}: {returns[-1]:.2f}\n")
            
            # Write metrics
            if "metrics" in result and result["metrics"]:
                f.write("\nMetrics:\n")
                for metric_name, value in result["metrics"].items():
                    if isinstance(value, dict):
                        f.write(f"  {metric_name.replace('_', ' ').title()}:\n")
                        for k, v in value.items():
                            f.write(f"    {k}: {v:.4f}\n")
                    else:
                        f.write(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}\n")
            
            f.write("\n\n")

if __name__ == "__main__":
    # Run all experiments
    results, output_dir = run_experiment_batch()
    print(f"\nAll experiments completed. Results saved to {output_dir}")