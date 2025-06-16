# Multi-Agent Reinforcement Learning for Dynamic Pricing in Online Retail

## Overview

This repository contains the full implementation of a multi-agent reinforcement learning (MARL) framework for dynamic pricing in ERP-integrated online retail environments. The research investigates how different MARL strategies adapt to demand fluctuations, competitor behavior, and market shocks, with a focus on revenue optimization, fairness, and price stability.

The simulation framework is built around realistic demand modeling and custom agent-environment interactions. It enables controlled experimentation with heterogeneous agent populations under configurable market conditions.

## Research Questions

This project explores the following research questions:

1. **Pricing Effectiveness**: How do MARL agents compare to rule-based strategies in competitive retail markets in terms of revenue and adaptability?
2. **Adaptation to Change**: How do MARL agents respond to market shocks, such as sudden changes in demand or competition?
3. **Emergent Dynamics**: What market behaviors arise when different agent types (learning-based and rule-based) interact in shared environments?

## Framework Architecture

The simulation environment models a multi-agent pricing scenario with:

- Multiple agents selling competing or substitute products
- Weekly demand cycles influenced by product price, seasonality, and agent behavior
- Market shocks (optional) that alter demand elasticity or volume mid-simulation
- Centralized logging of agent decisions, market outcomes, and performance metrics

### Agent Types

- **MADDPG** – Multi-Agent Deep Deterministic Policy Gradient  
  Continuous pricing, centralized training, decentralized execution  
- **MADQN** – Multi-Agent Deep Q-Network  
  Discrete pricing, decentralized learning  
- **QMIX** – Centralized value factorization for coordinated multi-agent learning  
- **Rule-Based Agents**, including:  
  - Fixed markup / static pricing  
  - Competitor-matching  
  - Demand-forecast responsive  
  - Seasonal indexing  

Each agent interacts with the environment through a defined interface, receives individualized observations, and learns policies to optimize revenue.

## Demand Modeling

- **Dataset**: Online Retail II from the UCI repository (filtered for B2B transactions)
- **Model**: LightGBM regression trained to predict weekly demand per product
- **Features**:
  - Lag-based sales trends
  - Price, time, and semantic product clusters (via S-BERT + KMeans)
  - Calendar effects (week of year, holiday flags)
  - Price positioning relative to category and historical context

## Evaluation Metrics

The framework includes built-in metrics for economic and strategic performance:

- Cumulative revenue  
- Nash equilibrium proximity  
- Price volatility and convergence  
- Fairness (Gini, Jain’s Index)  
- Social welfare (adjusted revenue × fairness)  
- Adjustment frequency and magnitude  
- Market share evolution  

## Key Findings

- **MARL agents outperform** rule-based agents in most revenue scenarios, especially under volatile demand.
- **MADDPG is most robust** to market shocks, but QMIX achieves better coordination in homogeneous setups.
- **Heterogeneous agent mixes** lead to diverse strategic equilibria and varying levels of fairness.
- **MARL agents exploit inelastic demand** in the dataset, underscoring the need for realistic elasticity modeling in future work.

## Repository Structure

```plaintext
agents/           # Agent classes (MADDPG, MADQN, QMIX, Rule-Based)
env/              # Retail market simulation environment
evaluation/       # Custom evaluation metrics and performance scripts
training/         # Demand model training (LightGBM, XGBoost, CatBoost)
scripts/          # Product selection, image generation, utilities
preprocess/       # Feature engineering scripts for demand modeling
run_experiments.py  # Main entry point for running simulation experiments
simulate.py         # Lower-level environment-agent interface
requirements.txt    # All necessary Python dependencies