"""
Multi-Agent E-commerce Retargeting Simulator
============================================

Sistem simulator pemasaran multi-agen yang menggabungkan:
- Multiple competing agents (different marketing strategies)
- Customer lifecycle management & retargeting
- E-commerce specific features (products, purchase history)
- Advanced auction mechanisms
- Sophisticated reward functions for different business objectives

Author: [Your Name]
Date: 2024
"""

import gymnasium as gym
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict, deque
import json

# ───────── Configuration ─────────
DATA_PATH = Path("avazu_dev_pro.parquet")
MODEL_PATH = Path("ctr_model_pro_two.txt")

# Load CTR model once
CTR_MODEL = lgb.Booster(model_file=str(MODEL_PATH))

FEATURES = [
    "hour_of_day", "day_of_week",
    "banner_pos", "device_type", "device_conn_type", 
    "site_category", "app_category",
    "C14", "C17", "C20", "C21"
]
CAT_COLS = ["site_category", "app_category"]

class CustomerStage(Enum):
    """Customer lifecycle stages for retargeting"""
    AWARENESS = "awareness"      # First visit
    INTEREST = "interest"        # Browsed products  
    CONSIDERATION = "consideration"  # Added to cart
    PURCHASE = "purchase"        # Made purchase
    RETENTION = "retention"      # Repeat customer
    CHURN_RISK = "churn_risk"   # Haven't engaged recently

class AgentStrategy(Enum):
    """Different agent marketing strategies"""
    AGGRESSIVE = "aggressive"    # High bids, broad targeting
    CONSERVATIVE = "conservative"  # Low bids, precise targeting
    ADAPTIVE = "adaptive"        # Dynamic strategy based on performance
    RETARGETING_FOCUSED = "retargeting_focused"  # Specialized in retargeting
    BRAND_AWARENESS = "brand_awareness"  # Focus on new customer acquisition

@dataclass 
class Customer:
    """Customer profile for retargeting simulation"""
    id: str
    stage: CustomerStage = CustomerStage.AWARENESS
    visit_count: int = 0
    purchase_count: int = 0
    total_spent: float = 0.0
    last_activity: int = 0  # timestep of last interaction
    preferred_categories: List[str] = field(default_factory=list)
    cart_value: float = 0.0
    lifetime_value: float = 0.0
    churn_probability: float = 0.0
    conversion_probability: float = 0.0
    response_history: Dict[str, List[bool]] = field(default_factory=dict)  # agent_id -> [responses]
    
    def update_stage(self, timestep: int):
        """Update customer stage based on behavior"""
        days_since_activity = (timestep - self.last_activity) / 24  # assuming hourly timesteps
        
        if days_since_activity > 30:
            self.stage = CustomerStage.CHURN_RISK
        elif self.purchase_count > 2:
            self.stage = CustomerStage.RETENTION
        elif self.purchase_count > 0:
            self.stage = CustomerStage.PURCHASE
        elif self.cart_value > 0:
            self.stage = CustomerStage.CONSIDERATION
        elif self.visit_count > 3:
            self.stage = CustomerStage.INTEREST
        else:
            self.stage = CustomerStage.AWARENESS

@dataclass
class Agent:
    """Marketing agent with specific strategy"""
    id: str
    strategy: AgentStrategy
    budget: float = 1000.0
    spent: float = 0.0
    revenue: float = 0.0
    wins: int = 0
    total_bids: int = 0
    performance_history: List[float] = field(default_factory=list)
    target_stages: List[CustomerStage] = field(default_factory=list)
    bid_adjustment_factor: float = 1.0
    
    def calculate_roas(self) -> float:
        """Calculate Return on Ad Spend"""
        return self.revenue / self.spent if self.spent > 0 else 0.0
    
    def update_performance(self, reward: float):
        """Update agent performance metrics"""
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:  # Keep last 100 records
            self.performance_history.pop(0)
    
    def get_average_performance(self) -> float:
        """Get average performance over recent history"""
        return np.mean(self.performance_history) if self.performance_history else 0.0

class MultiAgentRetargetingEnv(gym.Env):
    """
    Multi-Agent E-commerce Retargeting Environment
    
    Features:
    - Multiple agents competing in auctions
    - Customer lifecycle tracking
    - Sophisticated retargeting logic
    - Dynamic pricing and bidding
    - E-commerce specific metrics
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 data_path: str | Path = DATA_PATH,
                 num_agents: int = 4,
                 max_customers: int = 10000,
                 episode_length: int = 50000):
        
        # Load and preprocess data
        self._load_data(data_path)
        
        # Environment parameters
        self.num_agents = num_agents
        self.max_customers = max_customers
        self.episode_length = episode_length
        self.timestep = 0
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Customer management
        self.customers: Dict[str, Customer] = {}
        self.customer_pool = deque()  # Pool of customer IDs for cycling
        
        # Auction parameters
        self.min_bid = 0.01
        self.max_bid = 2.0
        self.reserve_price = 0.05
        
        # Action and observation spaces (per agent)
        # Action: [bid_amount(0-20), target_stage(0-5), retargeting_intensity(0-10)]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([20, 5, 10]), 
            dtype=np.float32
        )
        
        # Observation: [customer_features(10), market_state(5), agent_state(8)]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(23,), dtype=np.float32
        )
        
        # Metrics tracking
        self.metrics = {
            'total_revenue': 0.0,
            'total_spend': 0.0,
            'total_conversions': 0,
            'total_impressions': 0,
            'agent_metrics': {agent.id: {'revenue': 0, 'spend': 0, 'wins': 0} for agent in self.agents}
        }
    
    def _load_data(self, data_path: Path):
        """Load and preprocess auction data"""
        lf = pl.scan_parquet(str(data_path))
        
        # Remove duplicates and add time features
        for col in ("hour_of_day", "day_of_week"):
            if col in lf.columns:
                lf = lf.drop(col)
        
        lf = lf.with_columns([
            (pl.col("hour") % 100).alias("hour_of_day"),
            (((pl.col("hour") // 100) // 100) % 7).alias("day_of_week")
        ])
        
        self.data = lf
        self.data_ptr = 0
        
    def _create_agents(self) -> List[Agent]:
        """Create agents with different strategies"""
        strategies = [
            AgentStrategy.AGGRESSIVE,
            AgentStrategy.CONSERVATIVE, 
            AgentStrategy.ADAPTIVE,
            AgentStrategy.RETARGETING_FOCUSED
        ]
        
        agents = []
        for i in range(self.num_agents):
            strategy = strategies[i % len(strategies)]
            agent = Agent(
                id=f"agent_{i}",
                strategy=strategy,
                budget=1000.0
            )
            
            # Set target stages based on strategy
            if strategy == AgentStrategy.RETARGETING_FOCUSED:
                agent.target_stages = [CustomerStage.CONSIDERATION, CustomerStage.CHURN_RISK]
            elif strategy == AgentStrategy.BRAND_AWARENESS:
                agent.target_stages = [CustomerStage.AWARENESS, CustomerStage.INTEREST]
            else:
                agent.target_stages = list(CustomerStage)  # All stages
                
            agents.append(agent)
        
        return agents
    
    def _get_or_create_customer(self, row: pd.Series) -> Customer:
        """Get existing customer or create new one"""
        # Use C14 as customer identifier (simplified)
        customer_id = f"customer_{int(row['C14']) % self.max_customers}"
        
        if customer_id not in self.customers:
            self.customers[customer_id] = Customer(
                id=customer_id,
                preferred_categories=[row.get('site_category', ''), row.get('app_category', '')],
                conversion_probability=np.random.beta(2, 8)  # Realistic conversion distribution
            )
            self.customer_pool.append(customer_id)
        
        customer = self.customers[customer_id]
        customer.last_activity = self.timestep
        customer.visit_count += 1
        customer.update_stage(self.timestep)
        
        return customer
    
    def _calculate_base_ctr(self, row: pd.Series) -> float:
        """Calculate base CTR using trained model"""
        row_dict = {f: row[f] for f in FEATURES}
        row_df = pd.DataFrame([row_dict])
        for c in CAT_COLS:
            row_df[c] = row_df[c].astype("category")
        
        return float(CTR_MODEL.predict(row_df)[0])
    
    def _adjust_ctr_for_retargeting(self, base_ctr: float, customer: Customer, 
                                   agent: Agent, retargeting_intensity: float) -> float:
        """Adjust CTR based on retargeting factors"""
        adjustment = 1.0
        
        # Customer stage adjustment
        stage_multipliers = {
            CustomerStage.AWARENESS: 0.8,
            CustomerStage.INTEREST: 1.2,
            CustomerStage.CONSIDERATION: 1.5,
            CustomerStage.PURCHASE: 0.9,
            CustomerStage.RETENTION: 1.1,
            CustomerStage.CHURN_RISK: 1.3
        }
        adjustment *= stage_multipliers.get(customer.stage, 1.0)
        
        # Retargeting frequency adjustment (ad fatigue)
        if agent.id in customer.response_history:
            recent_responses = customer.response_history[agent.id][-10:]  # Last 10 interactions
            if len(recent_responses) > 3:
                response_rate = sum(recent_responses) / len(recent_responses)
                if response_rate < 0.1:  # Low response rate indicates fatigue
                    adjustment *= 0.7
        
        # Retargeting intensity effect
        intensity_effect = 1.0 + (retargeting_intensity / 10) * 0.5
        adjustment *= intensity_effect
        
        return min(base_ctr * adjustment, 0.95)  # Cap at 95%
    
    def _run_auction(self, customer: Customer, row: pd.Series) -> Tuple[Optional[Agent], float]:
        """Run auction among agents for this impression"""
        bids = []
        eligible_agents = []
        
        for agent in self.agents:
            if agent.spent >= agent.budget:  # Budget constraint
                continue
                
            # Strategy-based bidding
            obs = self._get_observation(agent, customer, row)
            base_bid = self._get_agent_bid(agent, obs, customer)
            
            if base_bid >= self.reserve_price:
                bids.append(base_bid)
                eligible_agents.append(agent)
        
        if not bids:
            return None, 0.0
        
        # Second-price auction
        sorted_bids = sorted(zip(bids, eligible_agents), reverse=True)
        winner = sorted_bids[0][1]
        winning_price = sorted_bids[1][0] if len(sorted_bids) > 1 else sorted_bids[0][0]
        
        return winner, winning_price
    
    def _get_agent_bid(self, agent: Agent, obs: np.ndarray, customer: Customer) -> float:
        """Get agent's bid based on strategy and observation"""
        # Base bid from observation (this would be replaced by trained policy)
        base_bid = obs[0] * self.max_bid  # Simplified: use first obs dimension
        
        # Strategy adjustments
        if agent.strategy == AgentStrategy.AGGRESSIVE:
            base_bid *= 1.3
        elif agent.strategy == AgentStrategy.CONSERVATIVE:
            base_bid *= 0.7
        elif agent.strategy == AgentStrategy.RETARGETING_FOCUSED:
            if customer.stage in [CustomerStage.CONSIDERATION, CustomerStage.CHURN_RISK]:
                base_bid *= 1.5
            else:
                base_bid *= 0.5
        elif agent.strategy == AgentStrategy.ADAPTIVE:
            # Adjust based on recent performance
            perf = agent.get_average_performance()
            if perf > 0:
                base_bid *= 1.2
            elif perf < -0.1:
                base_bid *= 0.8
        
        # Target stage alignment
        if customer.stage in agent.target_stages:
            base_bid *= 1.2
        
        return max(min(base_bid, self.max_bid), 0)
    
    def _get_observation(self, agent: Agent, customer: Customer, row: pd.Series) -> np.ndarray:
        """Generate observation for agent"""
        # Customer features (10 dimensions)
        customer_obs = [
            customer.visit_count / 100,  # Normalized visit count
            customer.purchase_count / 10,  # Normalized purchase count
            customer.total_spent / 1000,  # Normalized spend
            (self.timestep - customer.last_activity) / 1000,  # Recency
            customer.cart_value / 500,  # Normalized cart value
            customer.lifetime_value / 2000,  # Normalized LTV
            customer.churn_probability,
            customer.conversion_probability,
            list(CustomerStage).index(customer.stage) / len(CustomerStage),  # Stage encoding
            len(customer.preferred_categories) / 5  # Category diversity
        ]
        
        # Market state (5 dimensions)
        market_obs = [
            row["hour_of_day"] / 23,
            row["day_of_week"] / 6,
            row["banner_pos"] / 7,
            (int(row["C14"]) % 100) / 100,  # Market competition proxy
            self.timestep / self.episode_length  # Episode progress
        ]
        
        # Agent state (8 dimensions)
        agent_obs = [
            agent.spent / agent.budget,  # Budget utilization
            agent.calculate_roas() / 5,  # Normalized ROAS
            agent.wins / max(agent.total_bids, 1),  # Win rate
            agent.get_average_performance() + 1,  # Shifted performance
            agent.bid_adjustment_factor,
            len(agent.target_stages) / len(CustomerStage),  # Target diversity
            list(AgentStrategy).index(agent.strategy) / len(AgentStrategy),  # Strategy encoding
            len(agent.performance_history) / 100  # Experience
        ]
        
        return np.array(customer_obs + market_obs + agent_obs, dtype=np.float32)
    
    def _next_row(self) -> pd.Series:
        """Get next data row"""
        row = (
            self.data.slice(self.data_ptr, 1)
            .collect()
            .to_pandas()
            .iloc[0]
        )
        self.data_ptr = (self.data_ptr + 1) % 1000  # Cycle through data
        return row
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.timestep = 0
        self.data_ptr = 0
        
        # Reset agents
        for agent in self.agents:
            agent.spent = 0.0
            agent.revenue = 0.0
            agent.wins = 0
            agent.total_bids = 0
            agent.performance_history = []
        
        # Clear customers (or keep some for continuity)
        if options and options.get('keep_customers', False):
            # Reset customer states but keep profiles
            for customer in self.customers.values():
                customer.last_activity = 0
                customer.stage = CustomerStage.AWARENESS
        else:
            self.customers.clear()
            self.customer_pool.clear()
        
        # Reset metrics
        self.metrics = {
            'total_revenue': 0.0,
            'total_spend': 0.0,
            'total_conversions': 0,
            'total_impressions': 0,
            'agent_metrics': {agent.id: {'revenue': 0, 'spend': 0, 'wins': 0} for agent in self.agents}
        }
        
        # Initial observations
        row = self._next_row()
        customer = self._get_or_create_customer(row)
        
        observations = {}
        for agent in self.agents:
            observations[agent.id] = self._get_observation(agent, customer, row)
        
        return observations, {}
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step of the environment"""
        row = self._next_row()
        customer = self._get_or_create_customer(row)
        self.timestep += 1
        
        # Parse actions and update agent bids
        for agent_id, action in actions.items():
            agent = next(a for a in self.agents if a.id == agent_id)
            agent.total_bids += 1
            # Action format: [bid_amount, target_stage, retargeting_intensity]
            # Store in agent for auction
            agent.current_action = action
        
        # Run auction
        winner, price = self._run_auction(customer, row)
        
        # Calculate base CTR
        base_ctr = self._calculate_base_ctr(row)
        
        # Initialize rewards
        rewards = {agent.id: 0.0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}
        
        if winner:
            # Winner pays and gets impression
            winner.spent += price
            winner.wins += 1
            
            # Adjust CTR for retargeting
            retargeting_intensity = winner.current_action[2]
            adjusted_ctr = self._adjust_ctr_for_retargeting(
                base_ctr, customer, winner, retargeting_intensity
            )
            
            # Simulate click
            clicked = np.random.rand() < adjusted_ctr
            
            if clicked:
                # Update customer
                customer.last_activity = self.timestep
                if winner.id not in customer.response_history:
                    customer.response_history[winner.id] = []
                customer.response_history[winner.id].append(True)
                
                # Simulate conversion based on customer stage and agent strategy
                conversion_prob = customer.conversion_probability
                if customer.stage == CustomerStage.CONSIDERATION:
                    conversion_prob *= 2.0
                elif customer.stage == CustomerStage.CHURN_RISK:
                    conversion_prob *= 1.5
                    
                converted = np.random.rand() < conversion_prob
                
                if converted:
                    # Calculate revenue based on customer stage
                    base_revenue = 50.0  # Base conversion value
                    if customer.stage == CustomerStage.RETENTION:
                        revenue = base_revenue * 1.5  # Existing customers spend more
                    elif customer.stage == CustomerStage.CHURN_RISK:
                        revenue = base_revenue * 2.0  # Win-back campaigns high value
                    else:
                        revenue = base_revenue
                    
                    winner.revenue += revenue
                    customer.total_spent += revenue
                    customer.purchase_count += 1
                    
                    # Update metrics
                    self.metrics['total_revenue'] += revenue
                    self.metrics['total_conversions'] += 1
                    self.metrics['agent_metrics'][winner.id]['revenue'] += revenue
                    
                    # Reward for winner
                    rewards[winner.id] = revenue - price
                else:
                    # Click but no conversion
                    rewards[winner.id] = -price * 0.1  # Small penalty
            else:
                # No click
                if winner.id not in customer.response_history:
                    customer.response_history[winner.id] = []
                customer.response_history[winner.id].append(False)
                rewards[winner.id] = -price  # Full cost penalty
            
            # Update agent performance
            winner.update_performance(rewards[winner.id])
            self.metrics['total_spend'] += price
            self.metrics['agent_metrics'][winner.id]['spend'] += price
            self.metrics['agent_metrics'][winner.id]['wins'] += 1
        
        # Update all agent observations
        observations = {}
        for agent in self.agents:
            observations[agent.id] = self._get_observation(agent, customer, row)
            dones[agent.id] = self.timestep >= self.episode_length
        
        self.metrics['total_impressions'] += 1
        
        # Episode info
        info = {
            'timestep': self.timestep,
            'customer_stage': customer.stage.value,
            'auction_winner': winner.id if winner else None,
            'winning_price': price,
            'base_ctr': base_ctr,
            'total_customers': len(self.customers),
            'metrics': self.metrics.copy()
        }
        
        # All done when episode ends
        done = self.timestep >= self.episode_length
        dones['__all__'] = done
        
        return observations, rewards, dones, dones, info  # Ray RLLib format
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n=== Timestep {self.timestep} ===")
            print(f"Total Customers: {len(self.customers)}")
            print(f"Total Revenue: ${self.metrics['total_revenue']:.2f}")
            print(f"Total Spend: ${self.metrics['total_spend']:.2f}")
            print(f"Overall ROAS: {self.metrics['total_revenue']/max(self.metrics['total_spend'], 0.01):.2f}x")
            print(f"Conversion Rate: {self.metrics['total_conversions']/max(self.metrics['total_impressions'], 1)*100:.1f}%")
            
            print("\nAgent Performance:")
            for agent in self.agents:
                metrics = self.metrics['agent_metrics'][agent.id]
                roas = metrics['revenue'] / max(metrics['spend'], 0.01)
                win_rate = metrics['wins'] / max(agent.total_bids, 1) * 100
                print(f"  {agent.id} ({agent.strategy.value}): "
                      f"ROAS {roas:.2f}x | Win Rate {win_rate:.1f}% | "
                      f"Revenue ${metrics['revenue']:.2f}")

# Quick test
if __name__ == "__main__":
    env = MultiAgentRetargetingEnv(num_agents=3, episode_length=100)
    obs, _ = env.reset()
    
    print("Multi-Agent Retargeting Environment Test")
    print(f"Agents: {[a.id for a in env.agents]}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    for step in range(10):
        # Random actions for testing
        actions = {}
        for agent in env.agents:
            actions[agent.id] = env.action_space.sample()
        
        obs, rewards, dones, _, info = env.step(actions)
        
        if step % 5 == 0:
            env.render()
        
        if dones['__all__']:
            break
    
    print("\nTest completed successfully!") 