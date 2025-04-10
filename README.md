# Master-Thesis

## Online Retail Dynamic Pricing Simulation
### Overview
This project implements a multi-agent reinforcement learning framework for dynamic pricing in an online retail environment. It simulates a marketplace where multiple retail agents compete by adjusting their product prices to maximize revenue while responding to market conditions and competitor behaviors.

The simulation uses two types of reinforcement learning algorithms:

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient): A deep reinforcement learning approach using actor-critic networks
Q-Learning: A more traditional reinforcement learning method using state-action value tables
Project Structure
Online Retail/
├── agents/                    # Agent implementations
│   ├── learning_agent.py      # Rule-based competitive learning agent
│   ├── maddpg_agent.py        # MADDPG implementation
│   ├── rl_agent.py            # Q-learning implementation  
├── env/                       # Environment classes
│   ├── market_env.py          # Market simulation environment
│   ├── pricing_agent.py       # Base pricing agent class
│   ├── product.py             # Product class definition
├── models/                    # Trained demand models
│   └── lgbm_model.pkl         # LightGBM demand prediction model
├── data/                      # Dataset files
│   ├── online_retail_II.csv             # Original dataset
│   └── engineered_weekly_demand_with_lags.csv  # Processed dataset with features
├── scripts/                   # Data processing scripts
│   └── product_selection.py   # Product selection and analysis
├── maddpg_models/             # Saved model checkpoints
├── simulate.py                # Main simulation script
├── simulate.job               # Batch job submission script
├── README.md                  # Project documentation

### Key Features
- Multi-Agent Environment: Simulates multiple retail agents competing in the same market
- Dynamic Pricing Strategy: Agents can adjust prices based on market observations
- Market Demand Model: Uses a trained ML model to predict demand based on price and other features
- Seasonality Effects: Incorporates weekly and holiday seasonality effects
- Performance Visualization: Creates plots of revenue, price competition, and learning progress

### Configuration Options
The simulation can be configured by modifying parameters in `simulate.py`:
`weeks`: Number of weeks to simulate in each episode (default: 52)
`episodes`: Number of episodes to run for training (default: 50)
`num_agents`: Number of competing agents (default: 4)
`use_maddpg`: Use MADDPG (True) or Q-learning (False)

#### Agent Parameters
**MADDPG Agent**
- `actor_lr`: Learning rate for actor network (0.0005)
- `critic_lr`: Learning rate for critic network (0.001)
- `discount_factor`: Future reward discount factor (0.98)
- `tau`: Target network update parameter (0.005)
- `exploration_noise`: Exploration noise magnitude (decayed exponentially)

**Q-Learning Agent**
- `learning_rate`: Learning rate (0.05)
- `exploration_rate`: Exploration rate (decayed linearly with episodes)
- `discount_factor`: Future reward discount factor (0.9-0.98)

#### Demand Model
The demand prediction uses a LightGBM model trained on historical retail data. Features include:

- Product price
- Competitor prices
- Week of year
- Holiday periods
- Historical demand (lag features)
