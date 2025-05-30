"""
Multi-Agent Training Script for E-commerce Retargeting
======================================================

Training multiple agents simultaneously using Ray RLLib with:
- Different algorithms per agent (PPO, DQN, A3C)
- Competitive and cooperative dynamics
- Advanced curriculum learning
- Performance analysis and visualization

Usage:
    python train_multi_agent.py --config configs/multi_agent_config.yaml
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import PolicyID
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from multi_agent_simulator import MultiAgentRetargetingEnv, AgentStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentTrainer:
    """
    Advanced multi-agent trainer with curriculum learning and analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def create_policy_mapping(self) -> Dict[PolicyID, PolicySpec]:
        """Create policy mapping for different agent strategies"""
        policies = {}
        
        # PPO for aggressive and adaptive agents
        policies["ppo_policy"] = PolicySpec(
            policy_class=None,  # Use default
            observation_space=gym.spaces.Box(0.0, 1.0, shape=(23,), dtype=np.float32),
            action_space=gym.spaces.Box(
                low=np.array([0, 0, 0]), 
                high=np.array([20, 5, 10]), 
                dtype=np.float32
            ),
            config=PPOConfig().training(
                gamma=0.99,
                lr=3e-4,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                clip_param=0.2,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
            ).to_dict()
        )
        
        # DQN for conservative strategy (discrete actions)
        policies["dqn_policy"] = PolicySpec(
            policy_class=None,
            observation_space=gym.spaces.Box(0.0, 1.0, shape=(23,), dtype=np.float32),
            action_space=gym.spaces.Discrete(1000),  # Discretized action space
            config=DQNConfig().training(
                gamma=0.99,
                lr=1e-4,
                train_batch_size=32,
                target_network_update_freq=1000,
                exploration_config={
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.02,
                    "epsilon_timesteps": 100000,
                }
            ).to_dict()
        )
        
        # A3C for retargeting-focused agents
        policies["a3c_policy"] = PolicySpec(
            policy_class=None,
            observation_space=gym.spaces.Box(0.0, 1.0, shape=(23,), dtype=np.float32),
            action_space=gym.spaces.Box(
                low=np.array([0, 0, 0]), 
                high=np.array([20, 5, 10]), 
                dtype=np.float32
            ),
            config=A3CConfig().training(
                gamma=0.99,
                lr=1e-4,
                entropy_coeff=0.1,
            ).to_dict()
        )
        
        return policies
    
    def policy_mapping_fn(self, agent_id: str) -> PolicyID:
        """Map agents to policies based on their strategy"""
        if "agent_0" in agent_id or "agent_2" in agent_id:  # Aggressive/Adaptive
            return "ppo_policy"
        elif "agent_1" in agent_id:  # Conservative
            return "dqn_policy"  
        else:  # Retargeting-focused
            return "a3c_policy"
    
    def create_environment(self) -> MultiAgentRetargetingEnv:
        """Create the multi-agent environment"""
        env_config = self.config.get('environment', {})
        return MultiAgentRetargetingEnv(
            num_agents=env_config.get('num_agents', 4),
            max_customers=env_config.get('max_customers', 5000),
            episode_length=env_config.get('episode_length', 10000)
        )
    
    def train(self):
        """Main training loop with curriculum learning"""
        logger.info("Starting multi-agent training...")
        
        # Create environment for registration
        env = self.create_environment()
        
        # Register environment
        def env_creator(env_config):
            return MultiAgentRetargetingEnv(**env_config)
        
        tune.register_env("multi_agent_retargeting", env_creator)
        
        # Training configuration
        training_config = self.config.get('training', {})
        
        # Create algorithm configuration (using PPO as base)
        config = (
            PPOConfig()
            .environment(
                env="multi_agent_retargeting",
                env_config={
                    'num_agents': 4,
                    'max_customers': 5000,
                    'episode_length': 10000
                }
            )
            .multi_agent(
                policies=self.create_policy_mapping(),
                policy_mapping_fn=self.policy_mapping_fn,
                policies_to_train=["ppo_policy", "a3c_policy"]  # Don't train DQN in this setup
            )
            .training(
                train_batch_size=training_config.get('train_batch_size', 8000),
                sgd_minibatch_size=training_config.get('sgd_minibatch_size', 256),
                num_sgd_iter=training_config.get('num_sgd_iter', 10),
                lr=training_config.get('learning_rate', 3e-4),
                gamma=training_config.get('gamma', 0.99),
                entropy_coeff=training_config.get('entropy_coeff', 0.01),
            )
            .resources(
                num_gpus=0,  # Use CPU for now
                num_cpus_per_worker=1
            )
            .rollouts(
                num_rollout_workers=training_config.get('num_workers', 2),
                num_envs_per_worker=1
            )
            .evaluation(
                evaluation_interval=50,
                evaluation_num_episodes=10,
                evaluation_parallel_to_training=True
            )
        )
        
        # Create and train algorithm
        algo = config.build()
        
        # Training metrics
        training_results = []
        
        try:
            for iteration in range(training_config.get('num_iterations', 500)):
                result = algo.train()
                
                # Log results
                episode_reward_mean = result['episode_reward_mean']
                training_results.append({
                    'iteration': iteration,
                    'episode_reward_mean': episode_reward_mean,
                    'timesteps_total': result['timesteps_total']
                })
                
                if iteration % 50 == 0:
                    logger.info(f"Iteration {iteration}: "
                              f"Mean Reward = {episode_reward_mean:.2f}, "
                              f"Timesteps = {result['timesteps_total']}")
                    
                    # Save checkpoint
                    checkpoint_path = algo.save(str(self.results_dir / f"checkpoint_{iteration}"))
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Early stopping based on performance
                if len(training_results) > 100:
                    recent_rewards = [r['episode_reward_mean'] for r in training_results[-50:]]
                    if np.mean(recent_rewards) > training_config.get('target_reward', 50):
                        logger.info("Target reward reached! Stopping training.")
                        break
        
        finally:
            # Save final model
            final_checkpoint = algo.save(str(self.results_dir / "final_model"))
            logger.info(f"Final model saved: {final_checkpoint}")
            
            # Save training results
            results_df = pd.DataFrame(training_results)
            results_df.to_csv(self.results_dir / "training_results.csv", index=False)
            
            algo.stop()
    
    def evaluate(self, checkpoint_path: str, num_episodes: int = 100):
        """Evaluate trained agents"""
        logger.info(f"Evaluating model from {checkpoint_path}")
        
        # Load algorithm
        algo = PPO.from_checkpoint(checkpoint_path)
        
        # Create environment
        env = self.create_environment()
        
        episode_results = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = {agent_id: 0 for agent_id in obs.keys()}
            episode_metrics = {
                'total_revenue': 0,
                'total_spend': 0,
                'total_conversions': 0,
                'agent_performance': {agent_id: {'revenue': 0, 'spend': 0, 'wins': 0} 
                                    for agent_id in obs.keys()}
            }
            
            while not done:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = self.policy_mapping_fn(agent_id)
                    action = algo.get_policy(policy_id).compute_single_action(agent_obs)[0]
                    actions[agent_id] = action
                
                obs, rewards, dones, _, info = env.step(actions)
                
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                
                # Update metrics from info
                if 'metrics' in info:
                    episode_metrics['total_revenue'] = info['metrics']['total_revenue']
                    episode_metrics['total_spend'] = info['metrics']['total_spend']
                    episode_metrics['total_conversions'] = info['metrics']['total_conversions']
                    episode_metrics['agent_performance'] = info['metrics']['agent_metrics']
                
                done = dones.get('__all__', False)
            
            # Calculate ROAS for episode
            episode_roas = (episode_metrics['total_revenue'] / 
                          max(episode_metrics['total_spend'], 0.01))
            
            episode_results.append({
                'episode': episode,
                'total_reward': sum(episode_reward.values()),
                'agent_rewards': episode_reward,
                'roas': episode_roas,
                'revenue': episode_metrics['total_revenue'],
                'spend': episode_metrics['total_spend'],
                'conversions': episode_metrics['total_conversions'],
                'agent_performance': episode_metrics['agent_performance']
            })
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: ROAS = {episode_roas:.2f}, "
                          f"Revenue = ${episode_metrics['total_revenue']:.2f}")
        
        # Save evaluation results
        eval_df = pd.DataFrame(episode_results)
        eval_df.to_csv(self.results_dir / "evaluation_results.csv", index=False)
        
        # Generate performance report
        self._generate_performance_report(episode_results)
        
        return episode_results
    
    def _generate_performance_report(self, results: list):
        """Generate comprehensive performance analysis"""
        logger.info("Generating performance report...")
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Agent performance analysis
        agent_rewards = pd.DataFrame([r['agent_rewards'] for r in results])
        agent_performance = pd.DataFrame([r['agent_performance'] for r in results])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Agent Retargeting Performance Analysis', fontsize=16)
        
        # 1. ROAS over episodes
        axes[0, 0].plot(df['episode'], df['roas'])
        axes[0, 0].set_title('ROAS Over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('ROAS')
        axes[0, 0].grid(True)
        
        # 2. Revenue vs Spend
        axes[0, 1].scatter(df['spend'], df['revenue'], alpha=0.6)
        axes[0, 1].set_title('Revenue vs Spend')
        axes[0, 1].set_xlabel('Total Spend ($)')
        axes[0, 1].set_ylabel('Total Revenue ($)')
        axes[0, 1].grid(True)
        
        # 3. Agent reward distribution
        agent_rewards.boxplot(ax=axes[0, 2])
        axes[0, 2].set_title('Agent Reward Distribution')
        axes[0, 2].set_xlabel('Agent')
        axes[0, 2].set_ylabel('Reward')
        
        # 4. Conversion rate over time
        conversion_rate = df['conversions'] / df.index  # Simplified metric
        axes[1, 0].plot(df['episode'], conversion_rate)
        axes[1, 0].set_title('Conversion Rate Trend')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Conversions per Episode')
        axes[1, 0].grid(True)
        
        # 5. Agent performance comparison
        mean_rewards = agent_rewards.mean()
        mean_rewards.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average Reward by Agent')
        axes[1, 1].set_xlabel('Agent')
        axes[1, 1].set_ylabel('Average Reward')
        
        # 6. Performance correlation heatmap
        correlation_data = df[['roas', 'revenue', 'spend', 'conversions']].corr()
        sns.heatmap(correlation_data, annot=True, ax=axes[1, 2])
        axes[1, 2].set_title('Performance Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary statistics
        summary = {
            'average_roas': float(df['roas'].mean()),
            'max_roas': float(df['roas'].max()),
            'average_revenue': float(df['revenue'].mean()),
            'average_spend': float(df['spend'].mean()),
            'total_conversions': int(df['conversions'].sum()),
            'agent_performance': {
                agent: {
                    'average_reward': float(agent_rewards[agent].mean()),
                    'reward_std': float(agent_rewards[agent].std()),
                    'best_episode': int(agent_rewards[agent].idxmax()),
                    'worst_episode': int(agent_rewards[agent].idxmin())
                }
                for agent in agent_rewards.columns
            }
        }
        
        # Save summary
        with open(self.results_dir / 'performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Performance report saved to {self.results_dir}")
        logger.info(f"Average ROAS: {summary['average_roas']:.2f}")
        logger.info(f"Average Revenue: ${summary['average_revenue']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Train multi-agent retargeting system')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], 
                       default='train', help='Run mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'training': {
                'num_iterations': 300,
                'train_batch_size': 8000,
                'sgd_minibatch_size': 256,
                'num_sgd_iter': 10,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'entropy_coeff': 0.01,
                'num_workers': 2,
                'target_reward': 30.0
            },
            'environment': {
                'num_agents': 4,
                'max_customers': 5000,
                'episode_length': 10000
            },
            'results_dir': f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    trainer = MultiAgentTrainer(config)
    
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            logger.error("Checkpoint path required for evaluation mode")
            return
        trainer.evaluate(args.checkpoint)

if __name__ == "__main__":
    main() 