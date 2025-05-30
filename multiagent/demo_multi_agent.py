"""
Demo Multi-Agent Retargeting Simulator
======================================

Simple demo untuk menguji simulator multi-agent tanpa perlu Ray RLLib.
Menggunakan strategi sederhana untuk demonstrasi konsep dan evaluasi performa.

Usage:
    python demo_multi_agent.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys

# Import simulator
from multi_agent_simulator import MultiAgentRetargetingEnv, AgentStrategy, CustomerStage

class SimpleAgent:
    """Simple rule-based agent for demonstration"""
    
    def __init__(self, agent_id: str, strategy: AgentStrategy):
        self.agent_id = agent_id
        self.strategy = strategy
        self.performance_history = []
        
    def get_action(self, observation: np.ndarray, customer_stage: CustomerStage) -> np.ndarray:
        """Get action based on simple rules"""
        # Parse observation
        customer_features = observation[:10]
        market_features = observation[10:15]
        agent_features = observation[15:]
        
        # Base bid calculation
        visit_count_norm = customer_features[0]
        purchase_count_norm = customer_features[1]
        total_spent_norm = customer_features[2]
        recency_norm = customer_features[3]
        conversion_prob = customer_features[7]
        
        # Strategy-specific bidding
        if self.strategy == AgentStrategy.AGGRESSIVE:
            # High bids, especially for high-value customers
            base_bid = 5.0 + (total_spent_norm * 10)
            target_stage = 2  # Focus on consideration stage
            intensity = 8
            
        elif self.strategy == AgentStrategy.CONSERVATIVE:
            # Low bids, focus on proven converters
            base_bid = 1.0 + (conversion_prob * 3)
            target_stage = 1  # Focus on interest stage
            intensity = 3
            
        elif self.strategy == AgentStrategy.RETARGETING_FOCUSED:
            # High bids for retargeting opportunities
            if customer_stage in [CustomerStage.CONSIDERATION, CustomerStage.CHURN_RISK]:
                base_bid = 8.0
                intensity = 9
            else:
                base_bid = 2.0
                intensity = 4
            target_stage = 3  # Focus on purchase stage
            
        elif self.strategy == AgentStrategy.ADAPTIVE:
            # Adjust based on recent performance
            recent_perf = np.mean(self.performance_history[-10:]) if self.performance_history else 0
            if recent_perf > 0:
                base_bid = 4.0 + min(recent_perf, 5.0)
            else:
                base_bid = 2.0
            target_stage = 2
            intensity = 5
            
        else:  # BRAND_AWARENESS
            # Focus on new customers
            if customer_stage == CustomerStage.AWARENESS:
                base_bid = 6.0
                intensity = 7
            else:
                base_bid = 1.0
                intensity = 2
            target_stage = 0  # Focus on awareness stage
        
        # Add some randomness
        base_bid += np.random.normal(0, 0.5)
        base_bid = max(0.1, min(base_bid, 20.0))  # Clamp to valid range
        
        return np.array([base_bid, target_stage, intensity], dtype=np.float32)
    
    def update_performance(self, reward: float):
        """Update performance history"""
        self.performance_history.append(reward)
        if len(self.performance_history) > 50:  # Keep recent history
            self.performance_history.pop(0)

class MultiAgentDemo:
    """Demo runner for multi-agent system"""
    
    def __init__(self, num_episodes: int = 10, episode_length: int = 1000):
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.results = []
        
        # Create environment
        self.env = MultiAgentRetargetingEnv(
            num_agents=4,
            max_customers=1000,
            episode_length=episode_length
        )
        
        # Create simple agents
        strategies = [
            AgentStrategy.AGGRESSIVE,
            AgentStrategy.CONSERVATIVE,
            AgentStrategy.RETARGETING_FOCUSED,
            AgentStrategy.ADAPTIVE
        ]
        
        self.agents = []
        for i, strategy in enumerate(strategies):
            agent = SimpleAgent(f"agent_{i}", strategy)
            self.agents.append(agent)
    
    def run_demo(self):
        """Run the multi-agent demo"""
        print("🚀 Starting Multi-Agent Retargeting Demo")
        print(f"Episodes: {self.num_episodes}")
        print(f"Episode Length: {self.episode_length}")
        print(f"Agents: {[agent.strategy.value for agent in self.agents]}")
        print("=" * 60)
        
        for episode in range(self.num_episodes):
            print(f"\n📊 Episode {episode + 1}/{self.num_episodes}")
            
            # Reset environment
            observations, _ = self.env.reset()
            
            episode_metrics = {
                'episode': episode,
                'agent_rewards': {agent.agent_id: 0 for agent in self.agents},
                'agent_actions': {agent.agent_id: [] for agent in self.agents},
                'customer_stages': [],
                'winning_bids': [],
                'auction_winners': []
            }
            
            for step in range(self.episode_length):
                # Get actions from all agents
                actions = {}
                current_customer_stage = None
                
                for agent in self.agents:
                    obs = observations[agent.agent_id]
                    
                    # Get customer stage from environment info (simplified)
                    if step == 0:
                        current_customer_stage = CustomerStage.AWARENESS
                    else:
                        # Use stage from previous step or estimate
                        stage_index = int(obs[8] * len(CustomerStage))
                        current_customer_stage = list(CustomerStage)[min(stage_index, len(CustomerStage)-1)]
                    
                    action = agent.get_action(obs, current_customer_stage)
                    actions[agent.agent_id] = action
                    
                    episode_metrics['agent_actions'][agent.agent_id].append(action[0])  # Store bid
                
                # Step environment
                observations, rewards, dones, _, info = self.env.step(actions)
                
                # Update agent performance and metrics
                for agent in self.agents:
                    reward = rewards[agent.agent_id]
                    agent.update_performance(reward)
                    episode_metrics['agent_rewards'][agent.agent_id] += reward
                
                # Store step info
                if 'customer_stage' in info:
                    episode_metrics['customer_stages'].append(info['customer_stage'])
                if 'winning_price' in info:
                    episode_metrics['winning_bids'].append(info['winning_price'])
                if 'auction_winner' in info:
                    episode_metrics['auction_winners'].append(info['auction_winner'])
                
                # Check if episode is done
                if dones.get('__all__', False):
                    break
            
            # Store episode results
            episode_metrics.update({
                'total_revenue': info.get('metrics', {}).get('total_revenue', 0),
                'total_spend': info.get('metrics', {}).get('total_spend', 0),
                'total_conversions': info.get('metrics', {}).get('total_conversions', 0),
                'total_customers': info.get('total_customers', 0),
                'roas': info.get('metrics', {}).get('total_revenue', 0) / max(info.get('metrics', {}).get('total_spend', 0.01), 0.01)
            })
            
            self.results.append(episode_metrics)
            
            # Print episode summary
            print(f"  💰 Revenue: ${episode_metrics['total_revenue']:.2f}")
            print(f"  💸 Spend: ${episode_metrics['total_spend']:.2f}")
            print(f"  📈 ROAS: {episode_metrics['roas']:.2f}x")
            print(f"  🎯 Conversions: {episode_metrics['total_conversions']}")
            print(f"  👥 Customers: {episode_metrics['total_customers']}")
            
            # Agent performance
            for agent in self.agents:
                reward = episode_metrics['agent_rewards'][agent.agent_id]
                avg_bid = np.mean(episode_metrics['agent_actions'][agent.agent_id])
                print(f"    {agent.strategy.value:15} | Reward: {reward:7.2f} | Avg Bid: ${avg_bid:.2f}")
        
        print("\n✅ Demo completed!")
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and visualize results"""
        print("\n📊 Performance Analysis")
        print("=" * 60)
        
        # Calculate overall metrics
        total_revenue = sum(r['total_revenue'] for r in self.results)
        total_spend = sum(r['total_spend'] for r in self.results)
        total_conversions = sum(r['total_conversions'] for r in self.results)
        avg_roas = np.mean([r['roas'] for r in self.results])
        
        print(f"📈 Overall Performance:")
        print(f"  Total Revenue: ${total_revenue:.2f}")
        print(f"  Total Spend: ${total_spend:.2f}")
        print(f"  Average ROAS: {avg_roas:.2f}x")
        print(f"  Total Conversions: {total_conversions}")
        
        # Agent comparison
        print(f"\n🤖 Agent Performance Comparison:")
        agent_performance = {}
        for agent in self.agents:
            total_reward = sum(r['agent_rewards'][agent.agent_id] for r in self.results)
            avg_reward = total_reward / len(self.results)
            avg_bid = np.mean([np.mean(r['agent_actions'][agent.agent_id]) for r in self.results])
            
            agent_performance[agent.strategy.value] = {
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'avg_bid': avg_bid
            }
            
            print(f"  {agent.strategy.value:20} | Total: {total_reward:8.2f} | Avg: {avg_reward:6.2f} | Bid: ${avg_bid:.2f}")
        
        # Create visualizations if matplotlib is available
        try:
            self.create_visualizations(agent_performance)
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
        
        # Save results
        self.save_results(agent_performance)
    
    def create_visualizations(self, agent_performance):
        """Create performance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Agent Retargeting Performance Analysis', fontsize=16)
        
        # 1. ROAS over episodes
        roas_values = [r['roas'] for r in self.results]
        axes[0, 0].plot(roas_values, marker='o')
        axes[0, 0].set_title('ROAS Over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('ROAS')
        axes[0, 0].grid(True)
        
        # 2. Revenue vs Spend by episode
        revenues = [r['total_revenue'] for r in self.results]
        spends = [r['total_spend'] for r in self.results]
        axes[0, 1].scatter(spends, revenues, alpha=0.7)
        axes[0, 1].set_title('Revenue vs Spend')
        axes[0, 1].set_xlabel('Spend ($)')
        axes[0, 1].set_ylabel('Revenue ($)')
        axes[0, 1].grid(True)
        
        # 3. Agent performance comparison
        agents = list(agent_performance.keys())
        rewards = [agent_performance[agent]['avg_reward'] for agent in agents]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = axes[1, 0].bar(agents, rewards, color=colors)
        axes[1, 0].set_title('Average Reward by Agent Strategy')
        axes[1, 0].set_xlabel('Agent Strategy')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           f'{reward:.1f}', ha='center', va='bottom')
        
        # 4. Bidding behavior comparison
        bids = [agent_performance[agent]['avg_bid'] for agent in agents]
        axes[1, 1].bar(agents, bids, color=colors, alpha=0.7)
        axes[1, 1].set_title('Average Bid by Agent Strategy')
        axes[1, 1].set_xlabel('Agent Strategy')
        axes[1, 1].set_ylabel('Average Bid ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path('demo_results')
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'multi_agent_performance.png', dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {results_dir / 'multi_agent_performance.png'}")
        
        # Show plot if in interactive mode
        try:
            plt.show()
        except:
            pass  # Non-interactive environment
    
    def save_results(self, agent_performance):
        """Save results to files"""
        results_dir = Path('demo_results')
        results_dir.mkdir(exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        # Convert results for JSON serialization
        serializable_results = convert_types(self.results)
        serializable_performance = convert_types(agent_performance)
        
        # Save detailed results
        with open(results_dir / 'demo_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'num_episodes': self.num_episodes,
                    'episode_length': self.episode_length,
                    'num_agents': len(self.agents)
                },
                'results': serializable_results,
                'agent_performance': serializable_performance
            }, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for i, result in enumerate(self.results):
            row = {
                'episode': i,
                'revenue': float(result['total_revenue']),
                'spend': float(result['total_spend']),
                'roas': float(result['roas']),
                'conversions': int(result['total_conversions']),
                'customers': int(result['total_customers'])
            }
            # Add agent rewards
            for agent in self.agents:
                row[f'{agent.strategy.value}_reward'] = float(result['agent_rewards'][agent.agent_id])
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir / 'episode_summary.csv', index=False)
        
        print(f"💾 Results saved to: {results_dir}")

def main():
    """Main demo function"""
    print("🎯 Multi-Agent E-commerce Retargeting Simulator Demo")
    print("=" * 60)
    
    # Check if data files exist
    if not Path("avazu_dev_pro.parquet").exists():
        print("❌ Error: Data file 'avazu_dev_pro.parquet' not found!")
        print("Please ensure the data file is available in the current directory.")
        sys.exit(1)
    
    if not Path("ctr_model_pro_two.txt").exists():
        print("❌ Error: Model file 'ctr_model_pro_two.txt' not found!")
        print("Please ensure the CTR model is available in the current directory.")
        sys.exit(1)
    
    # Create and run demo
    demo = MultiAgentDemo(num_episodes=5, episode_length=500)
    demo.run_demo()
    
    print("\n🎉 Demo completed successfully!")
    print("Check the 'demo_results' directory for detailed analysis.")

if __name__ == "__main__":
    main() 