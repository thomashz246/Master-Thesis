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
            "name": "Config A - One MADDPG",
            "description": "1 MADDPG agent with 3 rule-based agents",
            "agent_types": ["maddpg", "rule", "rule", "rule"],
            "rule_strategy": "competitor_match"
        },
        {
            "name": "Config B - Two MADDPG",
            "description": "2 MADDPG agents with 2 rule-based agents",
            "agent_types": ["maddpg", "maddpg", "rule", "rule"],
            "rule_strategy": "competitor_match"
        },
        {
            "name": "Config C - Three MADDPG",
            "description": "3 MADDPG agents with 1 rule-based agent",
            "agent_types": ["maddpg", "maddpg", "maddpg", "rule"],
            "rule_strategy": "competitor_match"
        },
        {
            "name": "Config D - All MADDPG",
            "description": "All 4 agents using MADDPG",
            "agent_types": ["maddpg", "maddpg", "maddpg", "maddpg"],
            "rule_strategy": "competitor_match"  # Not used but included for consistency
        },
        {
            "name": "Config E - All Rule-Based",
            "description": "All 4 agents using rule-based strategy",
            "agent_types": ["rule", "rule", "rule", "rule"],
            "rule_strategy": "competitor_match"
        },
        {
            "name": "Config F - All Random",
            "description": "All 4 agents using random pricing",
            "agent_types": ["random", "random", "random", "random"],
            "rule_strategy": "competitor_match"  # Not used but included for consistency
        }
    ]
    
    # Create experiment results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_results_{timestamp}"
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
            episode_returns, metrics = run_experiment(
                config["agent_types"],
                config["rule_strategy"],
                weeks=52,       # Adjust as needed
                episodes=5,     # Adjust as needed
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
    
    # Generate comparative visualizations
    try:
        create_comparative_visuals(all_results, results_dir)
    except Exception as e:
        print(f"Error generating comparative visuals: {e}")
        traceback.print_exc()
    
    return all_results, results_dir

def run_experiment(agent_types, rule_strategy, weeks=52, episodes=5, save_dir=None):
    """Run a single experiment with specified agent configuration"""
    
    from simulate import run_simulation
    
    # Modify the run_simulation function to handle our agent_types configuration
    return run_simulation(
        weeks=weeks,
        episodes=episodes,
        num_agents=len(agent_types),
        agent_type="custom",  # We'll handle agent creation in the modified function
        agent_types=agent_types,
        rule_strategy=rule_strategy,
        save_dir=save_dir
    )

def create_comparative_visuals(all_results, results_dir):
    """Create comparative visualizations across configurations"""
    
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