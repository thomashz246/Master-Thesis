import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import json
from datetime import datetime
from scipy import stats  
from simulate import run_simulation
import traceback

def run_experiment_batch(enable_shocks=True):
    """Run all experimental configurations and save results"""
    
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
    },
    {
        "name": "Config I - MADDPG vs QMIX vs MADQN vs Rule-Based",
        "description": "Mixed MARL setup: 1 MADDPG, 1 MADQN, 1 QMIX, 1 Rule-Based",
        "agent_types": ["maddpg", "madqn", "qmix", "rule"],
        "rule_strategy": "competitor_match"
    }
]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_results/experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/experiment_config.json", "w") as f:
        json.dump(configurations, f, indent=4)
    
    all_results = {}
    for idx, config in enumerate(configurations):
        print(f"\n{'='*80}")
        print(f"Running Experiment {idx+1}/{len(configurations)}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")
        
        config_dir = f"{results_dir}/config_{chr(65+idx)}"
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            rule_strategies = config.get("rule_strategies")
            
            episode_returns, metrics = run_experiment(
                config["agent_types"],
                config.get("rule_strategy", "competitor_match"),
                rule_strategies=rule_strategies,
                weeks=104,
                episodes=20,
                save_dir=config_dir,
                enable_shocks=enable_shocks  
            )
            
            all_results[config["name"]] = {
                "returns": episode_returns,
                "metrics": metrics
            }
            
            print(f"Successfully completed configuration: {config['name']}")
            print(f"Returns: {episode_returns}")
            print(f"Metrics: {metrics}")
            
        except Exception as e:
            print(f"Error running configuration {config['name']}: {e}")
            traceback.print_exc()
            print("Moving to next configuration...")
            
            all_results[config["name"]] = {
                "returns": {},
                "metrics": {"error": str(e)}
            }
    
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
    
    print(f"All configurations completed. Starting comparative visualizations...")
    
    try:
        create_comparative_visuals(all_results, results_dir)
        compare_algorithm_pricing_behavior(all_results, results_dir)
        
        statistical_results = analyze_agent_revenue_statistics(all_results, results_dir)
        
        significance_results = test_adaptability_differences(all_results)
        if significance_results:
            with open(f"{results_dir}/statistical_significance.txt", "w") as f:
                for test_name, p_value in significance_results.items():
                    f.write(f"{test_name}: p={p_value:.4f} (significant: {p_value < 0.05})\n")
    except Exception as e:
        print(f"Error generating comparative visuals: {e}")
        traceback.print_exc()
        print("Continuing with partial results...")
    
    return all_results, results_dir

def run_experiment(agent_types, rule_strategy="competitor_match", 
                  rule_strategies=None, weeks=52, episodes=5, save_dir=None,
                  enable_shocks=True):
    """Run a single experiment with specified agent configuration"""
    
    try:
        from simulate import run_simulation
        from evaluation.eval_metrics import generate_evaluation_plots
        from evaluation.market_shock_analysis import analyze_market_shocks
        
        print(f"Starting experiment with agent types: {agent_types}")
        print(f"Market shocks enabled: {enable_shocks}")
        
        episode_returns, metrics, price_df = run_simulation(
            weeks=weeks,
            episodes=episodes,
            num_agents=len(agent_types),
            agent_type="custom",
            agent_types=agent_types,
            rule_strategy=rule_strategy,
            rule_strategies=rule_strategies,
            save_dir=save_dir,
            return_price_df=True, 
            enable_shocks=enable_shocks 
        )
        
        print(f"Simulation completed successfully")
        print(f"Price DF shape: {price_df.shape if price_df is not None else 'None'}")
        print(f"Episode returns: {episode_returns}")
        
        if save_dir and price_df is not None:
            if 'Episode' not in price_df.columns:
                price_df['Episode'] = 1
            
            additional_metrics = generate_evaluation_plots(price_df, episode_returns, save_dir) 
            adaptability_metrics = calculate_adaptability_metrics(price_df, episode_returns)
            
            if metrics is None:
                metrics = {}
            metrics.update(additional_metrics)
            metrics["adaptability"] = adaptability_metrics
            
            shock_response_metrics = analyze_market_shocks(
                price_df, 
                episode_returns, 
                save_dir, 
                enable_shocks=enable_shocks
            )
            
            metrics["shock_response"] = shock_response_metrics
            
            print("Shock response metrics:")
            for metric, values in shock_response_metrics.items():
                if isinstance(values, dict):
                    print(f"  {metric}: {values}")
                else:
                    print(f"  {metric}: {values}")
        
        if save_dir:
            episode_returns_df = pd.DataFrame(episode_returns)
            episode_returns_df.to_csv(f"{save_dir}/episode_returns.csv")
            
            if metrics and 'market_shares' in metrics:
                market_shares = metrics['market_shares']
                market_shares_df = pd.DataFrame({
                    agent_id: shares for agent_id, shares in market_shares.items()
                })
                market_shares_df.to_csv(f"{save_dir}/market_shares_evolution.csv")
            
            if metrics and 'fairness_over_time' in metrics:
                fairness_df = pd.DataFrame({
                    'Episode': range(1, len(metrics['fairness_over_time']) + 1),
                    'Fairness_Index': metrics['fairness_over_time']
                })
                fairness_df.to_csv(f"{save_dir}/fairness_evolution.csv")
            
            if price_df is not None:
                price_df.to_csv(f"{save_dir}/price_evolution_original.csv")
                
                processed_price_df = process_price_df_for_analysis(price_df)
                processed_price_df.to_csv(f"{save_dir}/price_evolution.csv")
                
                adaptability_metrics = calculate_adaptability_metrics(processed_price_df, episode_returns)
                if "adaptability" not in metrics:
                    metrics["adaptability"] = {}
                metrics["adaptability"].update(adaptability_metrics)
                
                price_stats = []
                
                if isinstance(price_df, pd.DataFrame) and not price_df.empty:
                    # Safely determine episode values
                    if 'Episode' in price_df.columns:
                        episode_values = sorted(price_df['Episode'].unique())
                    else:
                        episode_values = [1]
                        price_df['Episode'] = 1
                    
                    for episode in episode_values:
                        try:
                            episode_data = price_df[price_df['Episode'] == episode]
                            
                            for agent_product in episode_data.columns:
                                if agent_product not in ['Week', 'Episode', 'Year', 'ContinuousWeek']:
                                    try:
                                        price_series = pd.to_numeric(episode_data[agent_product], errors='coerce').dropna()
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
                                    except Exception as stat_error:
                                        print(f"Warning: Could not calculate statistics for {agent_product}: {stat_error}")
                        except Exception as e:
                            print(f"Warning: Error processing episode {episode}: {e}")
                
                if price_stats:
                    pd.DataFrame(price_stats).to_csv(f"{save_dir}/price_statistics.csv", index=False)
                
                price_stats = []
                
                if 'Episode' in price_df.columns:
                    episode_values = price_df['Episode'].unique()
                else:
                    episode_values = [1]
                
                for episode in episode_values:
                    episode_data = price_df[price_df['Episode'] == episode] if 'Episode' in price_df.columns else price_df
                    for agent_product in episode_data.columns:
                        if agent_product not in ['Week', 'Episode', 'Year']:
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
    
    except Exception as e:
        print(f"Error in run_experiment: {e}")
        traceback.print_exc()
        raise  
    
    return episode_returns, metrics

def create_comparative_visuals(all_results, results_dir):
    """Create comparative visualizations across configurations and save all metrics to CSV"""
    
    print(f"Creating comparative visualizations for {len(all_results)} configurations...")
    
    plt.figure(figsize=(12, 8))
    
    bar_data = []
    for config_name, result in all_results.items():
        if "error" in result.get("metrics", {}):
            print(f"Skipping config {config_name} due to error: {result['metrics']['error']}")
            continue
            
        for agent_id, returns in result["returns"].items():
            if returns: 
                bar_data.append({
                    "Config": config_name.split(" - ")[0],
                    "Agent": agent_id,
                    "Final Return": returns[-1]
                })
    
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        
        if not bar_df.empty:
            pivot_df = bar_df.pivot(index="Config", columns="Agent", values="Final Return")
            pivot_df.plot(kind="bar", figsize=(12, 8))
            plt.title("Final Episode Returns by Configuration")
            plt.ylabel("Revenue")
            plt.tight_layout()
            plt.savefig(f"{results_dir}/comparative_returns.png")
    else:
        print("Warning: No valid return data for bar chart")
    
    metrics_data = []
    for config_name, result in all_results.items():
        if "error" in result.get("metrics", {}):
            continue
            
        if "metrics" in result and result["metrics"]:
            for metric_name, value in result["metrics"].items():
                if not isinstance(value, dict) and not isinstance(value, list):
                    metrics_data.append({
                        "Config": config_name.split(" - ")[0],
                        "Metric": metric_name,
                        "Value": value
                    })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        unique_metrics = metrics_df["Metric"].unique()
        num_metrics = len(unique_metrics)
        
        if num_metrics > 0:
            n_rows = (num_metrics + 1) // 2
            n_cols = min(2, num_metrics)
            
            plt.figure(figsize=(14, 6 * n_rows))
            for i, metric in enumerate(unique_metrics[:4]): 
                plt.subplot(n_rows, n_cols, i+1)
                metric_data = metrics_df[metrics_df["Metric"] == metric]
                plt.bar(metric_data["Config"], metric_data["Value"])
                plt.title(metric.replace("_", " ").title())
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/comparative_metrics.png")
            print(f"Saved comparative metrics visualization to {results_dir}/comparative_metrics.png")
    else:
        print("Warning: No valid metrics data for comparative visualization")
    
    all_metrics_rows = []
    
    for config_name, result in all_results.items():
        config_key = config_name.split(" - ")[0]
        base_row = {"Configuration": config_name, "ConfigKey": config_key}
        
        for agent_id, returns in result["returns"].items():
            if returns:
                base_row[f"FinalReturn_{agent_id}"] = returns[-1]
        
        if "error" in result.get("metrics", {}):
            base_row["error"] = str(result["metrics"]["error"])
            all_metrics_rows.append(base_row)
            continue
            
        if "metrics" in result and result["metrics"]:
            for metric_name, value in result["metrics"].items():
                if isinstance(value, dict):
                    for sub_metric_name, sub_value in value.items():
                        base_row[f"{metric_name}_{sub_metric_name}"] = sub_value
                elif not isinstance(value, list): 
                    base_row[metric_name] = value
            
            all_metrics_rows.append(base_row)
    
    if all_metrics_rows:
        comprehensive_metrics_df = pd.DataFrame(all_metrics_rows)
        
        comprehensive_metrics_df.to_csv(f"{results_dir}/all_configurations_metrics.csv", index=False)
        print(f"Saved comprehensive metrics to {results_dir}/all_configurations_metrics.csv")     
        pivot_metrics = pd.DataFrame(index=[row["ConfigKey"] for row in all_metrics_rows])
        
        all_metric_cols = set()
        for row in all_metrics_rows:
            all_metric_cols.update(key for key in row.keys() if key not in ["Configuration", "ConfigKey"])
            
        for metric_col in all_metric_cols:
            for row in all_metrics_rows:
                config_key = row["ConfigKey"]
                if metric_col in row:
                    pivot_metrics.loc[config_key, metric_col] = row[metric_col]
        
        pivot_metrics.to_csv(f"{results_dir}/pivot_metrics_by_config.csv")
        print(f"Saved pivoted metrics to {results_dir}/pivot_metrics_by_config.csv")
    else:
        print("Warning: No metrics data to save")

def calculate_adaptability_metrics(price_df, episode_returns):
    """Calculate metrics specifically for pricing adaptability research question"""
    adaptability_metrics = {}
    
    if 'Agent' in price_df.columns and 'Product' in price_df.columns and 'Price' in price_df.columns:
        adjustment_sizes = []
        for agent in price_df['Agent'].unique():
            agent_data = price_df[price_df['Agent'] == agent]
            for product in agent_data['Product'].unique():
                product_data = agent_data[agent_data['Product'] == product]
                product_data['PriceChange'] = product_data['Price'].pct_change().abs()
                adjustment_sizes.extend(product_data['PriceChange'].dropna().tolist())
        
        adaptability_metrics['mean_adjustment_magnitude'] = np.mean(adjustment_sizes) if adjustment_sizes else 0
        
        threshold = 0.01 
        total_changes = 0
        total_opportunities = 0
        
        for agent in price_df['Agent'].unique():
            agent_data = price_df[price_df['Agent'] == agent]
            for product in agent_data['Product'].unique():
                product_data = agent_data[agent_data['Product'] == product]
                changes = (product_data['Price'].pct_change().abs() > threshold).sum()
                opportunities = len(product_data) - 1  
                total_changes += changes
                total_opportunities += opportunities
        
    else:
        price_columns = [col for col in price_df.columns if '_Price' in col]    
        adjustment_sizes = []
        for col in price_columns:
            price_changes = price_df[col].pct_change().abs().dropna()
            adjustment_sizes.extend(price_changes.tolist())
        
        adaptability_metrics['mean_adjustment_magnitude'] = np.mean(adjustment_sizes) if adjustment_sizes else 0
        
        threshold = 0.01  
        total_changes = 0
        total_opportunities = 0
        
        for col in price_columns:
            changes = (price_df[col].pct_change().abs() > threshold).sum()
            opportunities = len(price_df) - 1 
            total_changes += changes
            total_opportunities += opportunities
    
    adaptability_metrics['price_change_frequency'] = (
        total_changes / total_opportunities if total_opportunities > 0 else 0
    )
    
    return adaptability_metrics

def compare_algorithm_pricing_behavior(all_results, results_dir):
    """Compare pricing behavior across different algorithms"""
    print(f"Comparing algorithm pricing behavior across {len(all_results)} configurations...")
    
    algorithm_prices = {}
    algorithm_mapping = {}
    
    for idx, (config_name, result) in enumerate(all_results.items()):
        config_letter = chr(65 + idx)
        config_dir = f"{results_dir}/config_{config_letter}"
        price_file = f"{config_dir}/price_evolution.csv"
        
        if os.path.exists(price_file):
            prices_df = pd.read_csv(price_file)
            print(f"Found price file at {price_file}")
            
            if "MADDPG" in config_name:
                algorithm = "MADDPG"
            elif "MADQN" in config_name:
                algorithm = "MADQN"
            elif "QMIX" in config_name:
                algorithm = "QMIX"
            else:
                algorithm = "Rule-Based"
                
            agents = set()
            price_data = {}
            
            for col in prices_df.columns:
                if "_Price" in col:
                    parts = col.split("_")
                    agent_id = parts[0] 
                    agents.add(agent_id)
            
            processed_rows = []
            for _, row in prices_df.iterrows():
                week = row['Week']
                for agent_id in agents:
                    agent_cols = [c for c in prices_df.columns if c.startswith(agent_id) and c.endswith("_Price")]
                    for col in agent_cols:
                        product = col.split("_")[1] 
                        processed_rows.append({
                            'Week': week,
                            'Agent': agent_id,
                            'Product': product,
                            'Price': row[col],
                            'Algorithm': algorithm
                        })
            
            if processed_rows:
                algorithm_prices[algorithm] = pd.DataFrame(processed_rows)
                
        else:
            print(f"Warning: Price file not found at {price_file}")

def analyze_seasonal_adaptability(price_df, config_name):
    """Analyze how well agents adapt to seasonal demand patterns"""
    holiday_weeks = list(range(44, 53))
    
    seasonal_adaptability = {}   
    return seasonal_adaptability

def process_price_df_for_analysis(price_df):
    """Convert column-based price data to row-based format with Agent and Product columns"""
    processed_rows = []
    
    if price_df is None or price_df.empty:
        print("Warning: Empty price dataframe provided to process_price_df_for_analysis")
        return pd.DataFrame(columns=['Week', 'Agent', 'Product', 'Price'])
    
    for _, row in price_df.iterrows():
        week = row['Week']
        year = row['Year'] if 'Year' in price_df.columns else None
        episode = row['Episode'] if 'Episode' in price_df.columns else 1
        
        for col in price_df.columns:
            if '_Price' in col:
                parts = col.split('_')
                agent_id = parts[0] 
                product = parts[1] if len(parts) > 2 else "Unknown" 
                
                new_row = {
                    'Week': week,
                    'Agent': agent_id,
                    'Product': product,
                    'Price': row[col],
                    'Episode': episode
                }
                
                if year is not None:
                    new_row['Year'] = year
                
                processed_rows.append(new_row)
    
    if not processed_rows:
        print("Warning: No price data could be extracted from dataframe")
        return pd.DataFrame(columns=['Week', 'Agent', 'Product', 'Price', 'Episode'])
        
    return pd.DataFrame(processed_rows)

def analyze_agent_revenue_statistics(all_results, results_dir):
    """
    Analyze revenues across agent types with statistical tests
    Uses Wilcoxon rank test to compare revenues between different agent types
    """
    print("Analyzing agent revenue statistics with statistical tests...")
    
    agent_type_revenues = {}
    
    agent_type_mapping = {}
    
    for config_name, result in all_results.items():
        if "error" in result.get("metrics", {}):
            continue
        
        if "MADDPG" in config_name:
            agent_type = "MADDPG"
        elif "MADQN" in config_name:
            agent_type = "MADQN"
        elif "QMIX" in config_name:
            agent_type = "QMIX"
        elif "Rule-Based" in config_name:
            agent_type = "Rule-Based"
        else:
            agent_type = config_name.split(" - ")[0]
        
        for agent_id, returns in result["returns"].items():
            if agent_id not in agent_type_revenues:
                agent_type_revenues[agent_id] = []
            agent_type_revenues[agent_id].extend(returns)
    
    statistical_results = []
    agent_types = list(set(agent_type_revenues.keys()))
    
    for i in range(len(agent_types)):
        for j in range(i+1, len(agent_types)):
            type1 = agent_types[i]
            type2 = agent_types[j]
            
            try:
                revenues1 = agent_type_revenues[type1]
                revenues2 = agent_type_revenues[type2]
                
                if len(revenues1) < 3 or len(revenues2) < 3:
                    continue
                
                stat, p_value = stats.wilcoxon(revenues1, revenues2)
                
                statistical_results.append({
                    'Agent Type 1': type1,
                    'Agent Type 2': type2,
                    'Mean Revenue 1': np.mean(revenues1),
                    'Mean Revenue 2': np.mean(revenues2),
                    'Wilcoxon Statistic': stat,
                    'p-value': p_value,
                    'Significant Difference': p_value < 0.05
                })
            except Exception as e:
                print(f"Error in statistical test for {type1} vs {type2}: {e}")
    
    if statistical_results:
        results_df = pd.DataFrame(statistical_results)
        results_df.to_csv(f"{results_dir}/agent_revenue_statistics.csv", index=False)
        print(f"Saved agent revenue statistical comparison to {results_dir}/agent_revenue_statistics.csv")
        
        plt.figure(figsize=(12, 8))
        
        agent_means = {agent: np.mean(revenues) for agent, revenues in agent_type_revenues.items()}
        agents = list(agent_means.keys())
        means = [agent_means[a] for a in agents]
        
        bars = plt.bar(agents, means, alpha=0.7)
        
        sig_pairs = [(r['Agent Type 1'], r['Agent Type 2']) 
                    for r in statistical_results if r['Significant Difference']]
        
        y_max = max(means) * 1.1
        y_step = max(means) * 0.05
        for i, (a1, a2) in enumerate(sig_pairs):
            idx1 = agents.index(a1)
            idx2 = agents.index(a2)
            x1, x2 = idx1, idx2
            
            plt.plot([x1, x2], [y_max + i*y_step, y_max + i*y_step], 'k-')
            plt.text((x1 + x2) / 2, y_max + i*y_step, '*', ha='center', fontsize=16)
        
        plt.title('Mean Revenue by Agent Type with Statistical Significance')
        plt.ylabel('Mean Revenue')
        plt.xlabel('Agent Type')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/agent_revenue_statistical_comparison.png")
        plt.close()
    
    return statistical_results
if __name__ == "__main__":
    try:
        import os
        import json
        import pandas as pd
        import numpy as np
        import traceback
        import matplotlib.pyplot as plt
        import argparse
        
        parser = argparse.ArgumentParser(description='Run retail pricing experiments')
        parser.add_argument('--no-shocks', dest='enable_shocks', action='store_false',
                           help='Disable market shocks in simulation')
        parser.set_defaults(enable_shocks=True)
        args = parser.parse_args()
        
        print(f"Starting experiment batch (Market shocks: {'ENABLED' if args.enable_shocks else 'DISABLED'})...")
        results, output_dir = run_experiment_batch(enable_shocks=args.enable_shocks)
        print(f"\nAll experiments completed. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Critical error in experiment batch: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck your simulation code and environment configuration.")
