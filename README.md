# Multi-Agent Reinforcement Learning for Dynamic Pricing in ERP-Integrated Supply Chains

## Overview

This repository contains the complete implementation of a simulation framework designed to benchmark multi-agent reinforcement learning (MARL) algorithms for dynamic pricing in ERP-integrated supply chains. It addresses the limitations of static pricing strategies used in current ERP systems by exploring how autonomous pricing agents can learn and adapt in complex, interdependent market environments.

Despite ERP systems efficiently handling logistics, procurement, and financial planning, their pricing modules often rely on static rules or historical averages. These strategies fail to respond to real-time changes in demand, competition, and inventory levels, leading to suboptimal revenue, poor adaptability, and missed opportunities.

This project proposes and evaluates MARL-based pricing agents that learn in a shared environment shaped by competitor actions, demand patterns, and stochastic shocks. The environment simulates real-world market complexity based on actual retail data, enabling fine-grained comparisons between traditional pricing methods and learning-based approaches.

## Research Questions

This study aims to answer the following main research question:

> **How can Multi-Agent Reinforcement Learning (MARL) improve dynamic pricing strategies in ERP-integrated supply chains compared to traditional pricing models, while also capturing the relationships among agents with different pricing strategies and learning behaviours?**

To investigate this, the following sub-questions are examined:

1. **How do different MARL algorithms perform in terms of pricing adaptability and stability?**  
2. **How do MARL-based pricing strategies compare to traditional rule-based pricing agents?**  
3. **What is the impact of market dynamics on the effectiveness of MARL agents?**

---

## Framework Architecture

The framework consists of a simulation engine modeling:

- Multiple pricing agents competing in a shared environment  
- Product-level weekly demand, affected by price, competition, and seasonality  
- Stochastic market shocks altering demand elasticity or volume  
- Centralized logging of prices, revenues, demand, and policy behavior  

Each agent receives local observations and learns to set prices to optimize revenue under changing market conditions.

### Agent Types

- **MADDPG** – Continuous action MARL algorithm with centralized critic  
- **MADQN** – Discrete deep Q-learning variant adapted for multi-agent scenarios  
- **QMIX** – Value factorization algorithm supporting decentralized execution with joint utility modeling  
- **Rule-Based Agents** – Static strategies including:
  - Fixed markup (cost-plus pricing)
  - Competitor matching
  - Demand-forecast responsive
  - Seasonal indexing

---

## Demand Modeling

- **Dataset**: [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) (UCI Machine Learning Repository), filtered to include only valid B2B transactions
- **Model**: LightGBM regression trained to predict weekly product demand
- **Feature Engineering**:
  - Lagged demand trends (rolling averages, volatility)
  - Time features (week of year, holiday flags)
  - Semantic product clusters via S-BERT + KMeans
  - Relative price vs category average

---

## Evaluation Metrics

The following metrics are used to evaluate agent performance:

- **Revenue** (cumulative and per-agent)  
- **Nash equilibrium proximity**  
- **Price convergence and volatility**  
- **Fairness** (Jain’s Index, Gini Coefficient)  
- **Social welfare** (revenue adjusted by fairness)  
- **Adjustment magnitude and frequency**  
- **Market share evolution**

---

## Key Findings

- MARL agents consistently outperform static rule-based agents in revenue and adaptability.
- MADDPG strikes a strong balance between competitive behavior, fairness, and price stability.
- QMIX achieves better coordination when agents are homogeneous.
- Rule-based agents offer stable but inflexible pricing and perform poorly under dynamic conditions.
- MARL agents exploit the inelastic nature of the dataset, emphasizing the need for future work on more elastic market scenarios.

---

## Repository Structure

```plaintext
agents/             # MARL and rule-based agent implementations
env/                # Custom multi-agent pricing environment
evaluation/         # Metric logging and performance evaluation scripts
preprocess/         # Feature engineering and data transformations
scripts/            # Utility scripts for product selection and visualization
run_experiments.py  # Main simulation runner for batch configurations
simulate.py         # Environment-agent simulation interface
requirements.txt    # Python dependencies
README.md           # This documentation file
