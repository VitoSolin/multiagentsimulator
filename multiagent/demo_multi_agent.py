"""
Demo Multi-Agent Retargeting Simulator (Versi Final dengan Aliansi)
======================================================================

Demonstrasi canggih untuk menguji simulator multi-agen yang dilengkapi dengan:
1.  Strategi agen yang berbeda.
2.  Batasan anggaran (budget) untuk setiap agen.
3.  Komunikasi tidak langsung (shared channel) untuk agen adaptif.
4.  Mekanisme Aliansi untuk kerja sama strategis antar agen.

Agen dapat membentuk aliansi untuk menghindari perang harga (bidding war)
dan bekerja sama untuk mencapai tujuan bersama, mencerminkan strategi dunia nyata.

Usage:
    python demo_aliansi.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import Dict, List, Optional
from enum import Enum
import gymnasium as gym
import polars as pl
import lightgbm as lgb
from dataclasses import dataclass, field
import random
from collections import defaultdict, deque
from typing import Tuple, Any

# ==============================================================================
# KODE TERINTEGRASI DARI SIMULATOR
# ==============================================================================

# ───────── Konfigurasi Finansial ─────────
USD_TO_IDR = 15000

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
    AWARENESS = "awareness"
    INTEREST = "interest"
    CONSIDERATION = "consideration"
    PURCHASE = "purchase"
    RETENTION = "retention"
    CHURN_RISK = "churn_risk"

class AgentStrategy(Enum):
    """Different agent marketing strategies"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    RETARGETING_FOCUSED = "retargeting_focused"
    BRAND_AWARENESS = "brand_awareness"

@dataclass 
class Customer:
    """Customer profile for retargeting simulation"""
    id: str
    stage: CustomerStage = CustomerStage.AWARENESS
    visit_count: int = 0
    purchase_count: int = 0
    total_spent: float = 0.0
    last_activity: int = 0
    preferred_categories: List[str] = field(default_factory=list)
    cart_value: float = 0.0
    lifetime_value: float = 0.0
    churn_probability: float = 0.0
    conversion_probability: float = 0.0
    response_history: Dict[str, List[bool]] = field(default_factory=dict)
    
    def update_stage(self, timestep: int):
        days_since_activity = (timestep - self.last_activity) / 24
        if days_since_activity > 30: self.stage = CustomerStage.CHURN_RISK
        elif self.purchase_count > 2: self.stage = CustomerStage.RETENTION
        elif self.purchase_count > 0: self.stage = CustomerStage.PURCHASE
        elif self.cart_value > 0: self.stage = CustomerStage.CONSIDERATION
        elif self.visit_count > 3: self.stage = CustomerStage.INTEREST
        else: self.stage = CustomerStage.AWARENESS

@dataclass
class Agent:
    """Marketing agent with specific strategy"""
    id: str
    strategy: AgentStrategy
    budget: float = 1000.0 * USD_TO_IDR
    spent: float = 0.0
    revenue: float = 0.0
    wins: int = 0
    total_bids: int = 0
    performance_history: List[float] = field(default_factory=list)
    target_stages: List[CustomerStage] = field(default_factory=list)
    bid_adjustment_factor: float = 1.0
    
    def calculate_roas(self) -> float:
        return self.revenue / self.spent if self.spent > 0 else 0.0
    
    def update_performance(self, reward: float):
        self.performance_history.append(reward)
        if len(self.performance_history) > 100: self.performance_history.pop(0)
    
    def get_average_performance(self) -> float:
        return np.mean(self.performance_history) if self.performance_history else 0.0

class MultiAgentRetargetingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, data_path: str | Path = DATA_PATH, num_agents: int = 4, max_customers: int = 10000, episode_length: int = 50000):
        self._load_data(data_path)
        self.num_agents = num_agents
        self.max_customers = max_customers
        self.episode_length = episode_length
        self.timestep = 0
        self.agents = self._create_agents()
        self.customers: Dict[str, Customer] = {}
        self.customer_pool = deque()
        self.min_bid, self.max_bid, self.reserve_price = 0.01 * USD_TO_IDR, 20.0 * USD_TO_IDR, 0.05 * USD_TO_IDR
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.max_bid, 5, 10]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(23,), dtype=np.float32)
        self.metrics = {'total_revenue': 0.0, 'total_spend': 0.0, 'total_conversions': 0, 'total_impressions': 0, 'agent_metrics': {agent.id: {'revenue': 0, 'spend': 0, 'wins': 0} for agent in self.agents}}
    
    def _load_data(self, data_path: Path):
        lf = pl.scan_parquet(str(data_path))
        for col in ("hour_of_day", "day_of_week"):
            if col in lf.columns: lf = lf.drop(col)
        lf = lf.with_columns([(pl.col("hour") % 100).alias("hour_of_day"), (((pl.col("hour") // 100) // 100) % 7).alias("day_of_week")])
        self.data = lf
        self.data_ptr = 0
        
    def _create_agents(self) -> List[Agent]:
        strategies = [AgentStrategy.AGGRESSIVE, AgentStrategy.CONSERVATIVE, AgentStrategy.ADAPTIVE, AgentStrategy.RETARGETING_FOCUSED]
        agents = []
        for i in range(self.num_agents):
            strategy = strategies[i % len(strategies)]
            agent = Agent(id=f"agent_{i}", strategy=strategy)
            if strategy == AgentStrategy.RETARGETING_FOCUSED: agent.target_stages = [CustomerStage.CONSIDERATION, CustomerStage.CHURN_RISK]
            elif strategy == AgentStrategy.BRAND_AWARENESS: agent.target_stages = [CustomerStage.AWARENESS, CustomerStage.INTEREST]
            else: agent.target_stages = list(CustomerStage)
            agents.append(agent)
        return agents
    
    def _get_or_create_customer(self, row: pd.Series) -> Customer:
        customer_id = f"customer_{int(row['C14']) % self.max_customers}"
        if customer_id not in self.customers:
            self.customers[customer_id] = Customer(id=customer_id, preferred_categories=[row.get('site_category', ''), row.get('app_category', '')], conversion_probability=np.random.beta(2, 8))
            self.customer_pool.append(customer_id)
        customer = self.customers[customer_id]
        customer.last_activity, customer.visit_count = self.timestep, customer.visit_count + 1
        customer.update_stage(self.timestep)
        return customer
    
    def _calculate_base_ctr(self, row: pd.Series) -> float:
        row_dict = {f: row[f] for f in FEATURES}
        row_df = pd.DataFrame([row_dict])
        for c in CAT_COLS: row_df[c] = row_df[c].astype("category")
        return float(CTR_MODEL.predict(row_df)[0])
    
    def _adjust_ctr_for_retargeting(self, base_ctr: float, customer: Customer, agent: Agent, retargeting_intensity: float) -> float:
        adjustment = 1.0
        stage_multipliers = {CustomerStage.AWARENESS: 0.8, CustomerStage.INTEREST: 1.2, CustomerStage.CONSIDERATION: 1.5, CustomerStage.PURCHASE: 0.9, CustomerStage.RETENTION: 1.1, CustomerStage.CHURN_RISK: 1.3}
        adjustment *= stage_multipliers.get(customer.stage, 1.0)
        if agent.id in customer.response_history and len(customer.response_history[agent.id][-10:]) > 3 and sum(customer.response_history[agent.id][-10:]) / len(customer.response_history[agent.id][-10:]) < 0.1: adjustment *= 0.7
        adjustment *= 1.0 + (retargeting_intensity / 10) * 0.5
        return min(base_ctr * adjustment, 0.95)
    
    def _run_auction(self, customer: Customer, row: pd.Series) -> Tuple[Optional[Agent], float]:
        bids, eligible_agents = [], []
        for agent in self.agents:
            if agent.spent >= agent.budget: continue
            obs = self._get_observation(agent, customer, row)
            base_bid = self._get_agent_bid(agent, obs, customer)
            if base_bid >= self.reserve_price:
                bids.append(base_bid)
                eligible_agents.append(agent)
        
        if not bids: return None, 0.0

        # Menggunakan 'key' untuk memastikan sorting hanya berdasarkan bid (elemen pertama)
        # untuk menghindari TypeError jika ada bid yang sama.
        sorted_bids = sorted(zip(bids, eligible_agents), key=lambda item: item[0], reverse=True)
        
        # --- PERBAIKAN LOGIKA LELANG (SECOND-PRICE AUCTION) ---
        winning_agent = sorted_bids[0][1]
        # Pemenang membayar harga penawar tertinggi kedua, atau harga cadangan jika tidak ada penawar kedua.
        second_price = sorted_bids[1][0] if len(sorted_bids) > 1 else self.reserve_price
        price = max(second_price, self.reserve_price) # Pastikan harga tidak di bawah cadangan
        
        return winning_agent, price
    
    def _get_agent_bid(self, agent: Agent, obs: np.ndarray, customer: Customer) -> float:
        # NOTE: Logika ini sekarang dikendalikan oleh SimpleAgent melalui `agent.current_action`
        # Namun, kita simpan sebagai fallback jika diperlukan.
        return agent.current_action[0] if hasattr(agent, 'current_action') else 0.0
    
    def _get_observation(self, agent: Agent, customer: Customer, row: pd.Series) -> np.ndarray:
        customer_obs = [customer.visit_count / 100, customer.purchase_count / 10, customer.total_spent / 1000, (self.timestep - customer.last_activity) / 1000, customer.cart_value / 500, customer.lifetime_value / 2000, customer.churn_probability, customer.conversion_probability, list(CustomerStage).index(customer.stage) / len(CustomerStage), len(customer.preferred_categories) / 5]
        market_obs = [row["hour_of_day"] / 23, row["day_of_week"] / 6, row["banner_pos"] / 7, (int(row["C14"]) % 100) / 100, self.timestep / self.episode_length]
        agent_obs = [agent.spent / agent.budget, agent.calculate_roas() / 5, agent.wins / max(agent.total_bids, 1), agent.get_average_performance() + 1, agent.bid_adjustment_factor, len(agent.target_stages) / len(CustomerStage), list(AgentStrategy).index(agent.strategy) / len(AgentStrategy), len(agent.performance_history) / 100]
        return np.array(customer_obs + market_obs + agent_obs, dtype=np.float32)
    
    def _next_row(self) -> pd.Series:
        row = self.data.slice(self.data_ptr, 1).collect().to_pandas().iloc[0]
        self.data_ptr = (self.data_ptr + 1) % 1000
        return row
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.timestep, self.data_ptr = 0, 0
        for agent in self.agents: agent.spent, agent.revenue, agent.wins, agent.total_bids, agent.performance_history = 0.0, 0.0, 0, 0, []
        if options and options.get('keep_customers', False):
            for customer in self.customers.values(): customer.last_activity, customer.stage = 0, CustomerStage.AWARENESS
        else: self.customers.clear(); self.customer_pool.clear()
        self.metrics = {'total_revenue': 0.0, 'total_spend': 0.0, 'total_conversions': 0, 'total_impressions': 0, 'agent_metrics': {agent.id: {'revenue': 0, 'spend': 0, 'wins': 0} for agent in self.agents}}
        row = self._next_row()
        customer = self._get_or_create_customer(row)
        return {agent.id: self._get_observation(agent, customer, row) for agent in self.agents}, {}
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        row = self._next_row()
        customer = self._get_or_create_customer(row)
        self.timestep += 1
        for agent_id, action in actions.items():
            agent = next(a for a in self.agents if a.id == agent_id)
            agent.total_bids += 1
            agent.current_action = action
        winner, price = self._run_auction(customer, row)
        base_ctr = self._calculate_base_ctr(row)
        rewards = {agent.id: 0.0 for agent in self.agents}
        dones = {agent.id: False for agent in self.agents}
        if winner:
            winner.spent += price
            winner.wins += 1
            adjusted_ctr = self._adjust_ctr_for_retargeting(base_ctr, customer, winner, winner.current_action[2])
            clicked = np.random.rand() < adjusted_ctr
            if clicked:
                customer.last_activity = self.timestep
                if winner.id not in customer.response_history: customer.response_history[winner.id] = []
                customer.response_history[winner.id].append(True)
                conversion_prob = customer.conversion_probability * (2.0 if customer.stage == CustomerStage.CONSIDERATION else 1.5 if customer.stage == CustomerStage.CHURN_RISK else 1.0)
                if np.random.rand() < conversion_prob:
                    base_revenue = 50.0 * USD_TO_IDR
                    revenue = base_revenue * (1.5 if customer.stage == CustomerStage.RETENTION else 2.0 if customer.stage == CustomerStage.CHURN_RISK else 1.0)
                    winner.revenue += revenue
                    customer.total_spent, customer.purchase_count, customer.cart_value = customer.total_spent + revenue, customer.purchase_count + 1, 0.0 # Kosongkan keranjang setelah pembelian
                    self.metrics['total_revenue'] += revenue
                    self.metrics['total_conversions'] += 1
                    self.metrics['agent_metrics'][winner.id]['revenue'] += revenue
                    rewards[winner.id] = revenue - price
                else: 
                    # Jika klik tapi tidak konversi, ada kemungkinan customer memasukkan barang ke keranjang
                    if np.random.rand() < 0.25: # 25% kemungkinan menambah ke keranjang
                        customer.cart_value += np.random.uniform(10, 150) * USD_TO_IDR
                    rewards[winner.id] = -price * 0.1
            else:
                if winner.id not in customer.response_history: customer.response_history[winner.id] = []
                customer.response_history[winner.id].append(False)
                rewards[winner.id] = -price
            winner.update_performance(rewards[winner.id])
            self.metrics['total_spend'] += price
            self.metrics['agent_metrics'][winner.id]['spend'] += price
            self.metrics['agent_metrics'][winner.id]['wins'] += 1
        observations = {agent.id: self._get_observation(agent, customer, row) for agent in self.agents}
        for agent in self.agents: dones[agent.id] = self.timestep >= self.episode_length
        self.metrics['total_impressions'] += 1
        info = {'timestep': self.timestep, 'customer_stage': customer.stage.value, 'auction_winner': winner.id if winner else None, 'winning_price': price, 'base_ctr': base_ctr, 'total_customers': len(self.customers), 'metrics': self.metrics.copy()}
        done = self.timestep >= self.episode_length
        dones['__all__'] = done
        return observations, rewards, dones, dones, info
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\n=== Timestep {self.timestep} ===")
            print(f"Total Customers: {len(self.customers)}")
            print(f"Total Revenue: Rp {self.metrics['total_revenue']:,.0f}, Total Spend: Rp {self.metrics['total_spend']:,.0f}, Overall ROAS: {self.metrics['total_revenue']/max(self.metrics['total_spend'], 0.01):.2f}x")
            print(f"Conversion Rate: {self.metrics['total_conversions']/max(self.metrics['total_impressions'], 1)*100:.1f}%")
            print("\nAgent Performance:")
            for agent in self.agents:
                metrics = self.metrics['agent_metrics'][agent.id]
                roas = metrics['revenue'] / max(metrics['spend'], 0.01)
                win_rate = metrics['wins'] / max(agent.total_bids, 1) * 100
                print(f"  {agent.id} ({agent.strategy.value}): ROAS {roas:.2f}x | Win Rate {win_rate:.1f}% | Revenue Rp {metrics['revenue']:,.0f}")



# Tipe data untuk kejelasan kode
CommunicationChannel = Dict[str, Dict[str, float]]

class SimpleAgent:
    """Agen dengan kemampuan aliansi, komunikasi, dan budget (VERSI SEIMBANG)"""
    
    def __init__(self, agent_id: str, strategy: AgentStrategy, initial_budget: float = 1000.0):
        self.agent_id = agent_id
        self.strategy = strategy
        self.performance_history = []
        
        self.initial_budget = initial_budget
        self.budget = initial_budget
        
        self.alliance_id: Optional[str] = None
        self.allies: List[str] = []

    def set_alliance(self, alliance_id: str, ally_ids: List[str]):
        self.alliance_id = alliance_id
        self.allies = [ally for ally in ally_ids if ally != self.agent_id]

    def reset(self):
        self.performance_history = []
        self.budget = self.initial_budget
        
    def calculate_suitability(self, observation: np.ndarray, customer_stage: CustomerStage) -> float:
        """TAHAP 1: Skor kecocokan yang lebih seimbang untuk persaingan."""
        customer_features = observation[:10]
        recency_norm = customer_features[3]
        conversion_prob = customer_features[7]
        
        score = 0.0
        # Formula skor disesuaikan agar lebih kompetitif
        if self.strategy == AgentStrategy.AGGRESSIVE:
            # Tetap agresif, tapi sangat bergantung pada probabilitas konversi
            score = 10 + conversion_prob * 90
        elif self.strategy == AgentStrategy.CONSERVATIVE:
            # Lebih percaya diri pada targetnya
            score = 60 + conversion_prob * 50
        elif self.strategy == AgentStrategy.RETARGETING_FOCUSED:
            # Sangat menginginkan customer di tahap pertimbangan/risiko churn
            if customer_stage in [CustomerStage.CONSIDERATION, CustomerStage.CHURN_RISK]:
                score = 70 + recency_norm * 15 + conversion_prob * 15
        elif self.strategy == AgentStrategy.ADAPTIVE:
            # Lebih optimis secara default
            score = 40 + conversion_prob * 60
        else: # BRAND_AWARENESS
             if customer_stage == CustomerStage.AWARENESS:
                 score = 60
        
        return score + np.random.normal(0, 5)

    def get_action(self, observation: np.ndarray, customer_stage: CustomerStage, 
                   communication_channel: CommunicationChannel, 
                   suitability_scores: Dict[str, float]) -> np.ndarray:
        """TAHAP 2: Logika bidding yang dirombak total untuk keseimbangan."""
        
        if self.budget <= 0.1:
            return np.array([0.0, 0, 1], dtype=np.float32)

        is_alliance_champion = False
        my_score = suitability_scores.get(self.agent_id, 0)
        
        if self.allies:
            highest_ally_score = max(suitability_scores.get(ally_id, 0) for ally_id in self.allies)
            if highest_ally_score > my_score * 1.3: # Threshold sedikit diturunkan
                return np.array([0.1, 0, 1], dtype=np.float32) # Mengalah
            else:
                is_alliance_champion = True # Saya atau sekutu sama kuat, jadi saya maju!
        
        customer_features = observation[:10]
        recency_norm = customer_features[3]
        conversion_prob = customer_features[7]
        
        base_bid = 0.0
        target_stage_map = {s: i for i, s in enumerate(CustomerStage)}

        # --- FORMULA BIDDING BARU YANG LEBIH SEIMBANG (V3) ---
        if self.strategy == AgentStrategy.AGGRESSIVE:
            # Dikalibrasi ulang: tetap kuat tapi tidak mendominasi total.
            # Bid lebih bergantung pada probabilitas konversi yang tinggi (non-linear).
            base_bid = (0.6 + (conversion_prob**1.5 * 10)) * USD_TO_IDR
            intensity = 7 

        elif self.strategy == AgentStrategy.CONSERVATIVE:
            # Lebih berani: Dihilangkan syarat probabilitas, bid naik secara eksponensial.
            # Memberi reward besar untuk target yang sangat menjanjikan.
            base_bid = (0.5 + (conversion_prob**2 * 15)) * USD_TO_IDR
            intensity = 4

        elif self.strategy == AgentStrategy.RETARGETING_FOCUSED:
            # Disempurnakan: 'Raja' di segmennya, dengan bid minor di tahap 'Interest'.
            if customer_stage in [CustomerStage.CONSIDERATION, CustomerStage.CHURN_RISK]:
                # Tawaran premium untuk customer bernilai tinggi, sangat agresif.
                base_bid = (2.5 + (recency_norm * 2.0) + (conversion_prob**1.5 * 12)) * USD_TO_IDR
                intensity = 10
            elif customer_stage == CustomerStage.INTEREST:
                # Tawaran kecil untuk 'memanaskan' customer
                base_bid = (0.4 + (conversion_prob * 3)) * USD_TO_IDR
                intensity = 3
            else:
                base_bid = (0.1) * USD_TO_IDR # Sangat pasif di luar target utama
                intensity = 1

        elif self.strategy == AgentStrategy.ADAPTIVE:
            # REVISI V4: Dibuat lebih kompetitif sebagai 'generalis' yang oportunis.
            roas = self.get_roas()

            # Formula dasar yang solid, tidak terlalu bergantung pada pasar di awal.
            # Kuat pada probabilitas konversi menengah ke atas.
            base_bid = (0.8 + (conversion_prob**1.5 * 8)) * USD_TO_IDR

            # Pengali ROAS yang lebih sederhana dan efektif.
            # Jika ROAS bagus (di atas 1.2), bid akan meningkat secara signifikan.
            # Jika buruk (di bawah 0.8), bid akan dikurangi.
            if roas > 1.2:
                roas_multiplier = 1 + (roas - 1.2) * 0.5
            elif roas < 0.8 and roas > 0:
                roas_multiplier = 1 - (0.8 - roas)
            else:
                roas_multiplier = 1.0
            
            base_bid *= max(0.5, min(roas_multiplier, 2.0)) # Batasi efek multiplier antara 0.5x dan 2.0x

            intensity = 5 + int(roas * 2) # Intensitas dinamis
        
        else: # BRAND_AWARENESS
            base_bid = (2.5 if customer_stage == CustomerStage.AWARENESS else 0.4) * USD_TO_IDR
            intensity = 6 if customer_stage == CustomerStage.AWARENESS else 2
        
        # --- BONUS UNTUK JAGOAN ALIANSI ---
        if is_alliance_champion:
            base_bid *= 1.25 # Dapat bonus kekuatan 25% karena mewakili aliansi

        base_bid += np.random.normal(0, 0.3 * USD_TO_IDR)
        final_bid = max(0.1 * USD_TO_IDR, min(base_bid, 20.0 * USD_TO_IDR, self.budget))
        
        # Target stage bisa disederhanakan
        target_stage = [0,1,2,3,2][list(AgentStrategy).index(self.strategy)]
        
        return np.array([final_bid, target_stage, intensity], dtype=np.float32)
    
    def update_performance(self, reward: float):
        self.performance_history.append(reward)
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
    
    def get_roas(self) -> float:
        """Menghitung Return on Ad Spend dari histori performa."""
        total_revenue = sum(max(0, r) for r in self.performance_history)
        total_cost = sum(abs(min(0, r)) for r in self.performance_history)
        return total_revenue / total_cost if total_cost > 0 else 0

    def update_budget(self, cost: float):
        self.budget -= cost

class MultiAgentDemo:
    """Runner untuk demo multi-agen dengan aliansi."""
    
    def __init__(self, num_episodes: int = 10, episode_length: int = 1000):
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.results = []
        
        self.env = MultiAgentRetargetingEnv(num_agents=4, max_customers=1000, episode_length=episode_length)
        
        strategies = [AgentStrategy.AGGRESSIVE, AgentStrategy.CONSERVATIVE, AgentStrategy.RETARGETING_FOCUSED, AgentStrategy.ADAPTIVE]
        self.agent_map = {strategy: f"agent_{i}" for i, strategy in enumerate(strategies)}
        self.agents = [SimpleAgent(self.agent_map[s], s, initial_budget=500.0 * USD_TO_IDR) for s in strategies]
            
        # Mendefinisikan Aliansi antara agen Konservatif dan Adaptif
        self.alliances = {
            "alliance_1": [self.agent_map[AgentStrategy.CONSERVATIVE], self.agent_map[AgentStrategy.ADAPTIVE]]
        }
        self._setup_alliances()
            
        self.communication_channel: CommunicationChannel = {agent.agent_id: {'avg_bid': 0.0} for agent in self.agents}

    def _setup_alliances(self):
        """Memberitahu setiap agen tentang aliansi mereka."""
        for alliance_id, member_ids in self.alliances.items():
            for agent in self.agents:
                if agent.agent_id in member_ids:
                    agent.set_alliance(alliance_id, member_ids)

    def run_demo(self):
        print("🚀 Memulai Demo Multi-Agen dengan Mekanisme ALIANSI")
        print(f"Aliansi aktif: {self.alliances}")
        print("=" * 60)
        
        for episode in range(self.num_episodes):
            print(f"\n📊 Episode {episode + 1}/{self.num_episodes}")
            observations, _ = self.env.reset()
            for agent in self.agents:
                agent.reset()
            
            episode_metrics = {
                'agent_rewards': {agent.agent_id: 0 for agent in self.agents},
                'agent_bids': {agent.agent_id: [] for agent in self.agents},
                'auction_winners': []
            }
            
            for step in range(self.episode_length):
                # --- Loop Simulasi 2 Tahap ---
                # Tahap 1: Hitung dan kumpulkan Skor Kecocokan
                suitability_scores = {}
                # Ambil observasi dari salah satu agen untuk mendapatkan info customer stage
                sample_obs = list(observations.values())[0]
                stage_index = int(sample_obs[8] * len(CustomerStage))
                current_customer_stage = list(CustomerStage)[min(stage_index, len(CustomerStage)-1)]

                for agent in self.agents:
                    obs = observations[agent.agent_id]
                    suitability_scores[agent.agent_id] = agent.calculate_suitability(obs, current_customer_stage)

                # Tahap 2: Dapatkan aksi final dengan info aliansi
                actions = {}
                for agent in self.agents:
                    obs = observations[agent.agent_id]
                    action = agent.get_action(obs, current_customer_stage, self.communication_channel, suitability_scores)
                    actions[agent.agent_id] = action
                    episode_metrics['agent_bids'][agent.agent_id].append(action[0])
                
                observations, rewards, dones, _, info = self.env.step(actions)
                
                # Proses setelah lelang
                auction_winner = info.get('auction_winner')
                winning_price = info.get('winning_price', 0)
                
                for agent in self.agents:
                    cost = winning_price if agent.agent_id == auction_winner else 0
                    agent.update_budget(cost)
                    reward = rewards[agent.agent_id]
                    agent.update_performance(reward)
                    episode_metrics['agent_rewards'][agent.agent_id] += reward

                    # Update channel komunikasi dengan aman untuk menghindari warning
                    recent_bids = episode_metrics['agent_bids'][agent.agent_id][-20:]
                    self.communication_channel[agent.agent_id]['avg_bid'] = np.mean(recent_bids) if recent_bids else 0.0

                if auction_winner:
                    episode_metrics['auction_winners'].append(auction_winner)
                
                if dones.get('__all__', False):
                    break
            
            # Cetak ringkasan episode
            total_spend = info.get('metrics', {}).get('total_spend', 0.01)
            total_revenue = info.get('metrics', {}).get('total_revenue', 0)
            roas_final = total_revenue / total_spend if total_spend > 0.01 else 0

            print(f"  💰 Revenue: Rp {total_revenue:,.0f} | 💸 Spend: Rp {total_spend:,.0f} | 📈 ROAS: {roas_final:.2f}x")
            print("  --- Performa Agen ---")

            episode_summary = {
                'episode': episode + 1,
                'total_revenue': total_revenue,
                'total_spend': total_spend,
                'roas': roas_final
            }

            for agent in self.agents:
                reward = episode_metrics['agent_rewards'][agent.agent_id]
                
                # Menghitung bid rata-rata dengan aman
                agent_bids = episode_metrics['agent_bids'][agent.agent_id]
                avg_bid = np.mean(agent_bids) if agent_bids else 0.0

                wins = episode_metrics['auction_winners'].count(agent.agent_id)
                ally_tag = " (Aliansi)" if agent.allies else ""
                print(f"    - {agent.strategy.value:19}{ally_tag} | Reward: {reward:10,.2f} | Avg Bid: Rp {avg_bid:7,.0f} | Wins: {wins:3d} | Budget Left: Rp {agent.budget:9,.0f}")

                # Menggunakan strategy.value sebagai bagian dari kunci untuk legenda plot yang lebih baik
                episode_summary[f'{agent.strategy.value}_wins'] = wins
                episode_summary[f'{agent.strategy.value}_budget_left'] = agent.budget
            
            self.results.append(episode_summary)

        print("\n✅ Demo selesai!")
        self.analyze_and_visualize_results()

    def analyze_and_visualize_results(self):
        """Menganalisis dan memvisualisasikan hasil dari semua episode."""
        if not self.results:
            print("\nTidak ada hasil untuk dianalisis.")
            return

        # Membuat direktori untuk menyimpan hasil visualisasi
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path(f"visualisasi_hasil/{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📊 Menganalisis dan menyimpan visualisasi ke folder: {save_dir}")
        
        df = pd.DataFrame(self.results)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # --- Plot 1: Kemenangan Lelang per Agen ---
        plt.figure(figsize=(12, 7))
        win_cols = [col for col in df.columns if 'wins' in col]
        agent_names = [col.replace('_wins', '').replace('_', ' ').title() for col in win_cols]
        
        df_wins = df[win_cols]
        df_wins.columns = agent_names
        df_wins.plot(kind='bar', stacked=False, ax=plt.gca())
        
        plt.title('Distribusi Kemenangan Lelang per Episode', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Jumlah Kemenangan', fontsize=12)
        plt.xticks(ticks=df.index, labels=df['episode'], rotation=0)
        plt.legend(title='Strategi Agen')
        plt.tight_layout()
        plt.savefig(save_dir / "distribusi_kemenangan.png")
        plt.show()

        # --- Plot 2: Metrik Finansial Global ---
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Jumlah (Rp)', fontsize=12, color='tab:blue')
        ax1.plot(df['episode'], df['total_revenue'], 'o-', color='tab:blue', label='Total Pendapatan', linewidth=2)
        ax1.plot(df['episode'], df['total_spend'], 'o--', color='tab:cyan', label='Total Pengeluaran')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {int(x/1000)}k'))

        ax2 = ax1.twinx()
        ax2.set_ylabel('ROAS (x)', fontsize=12, color='tab:red')
        ax2.plot(df['episode'], df['roas'], 's-', color='tab:red', label='ROAS', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(bottom=0)

        plt.title('Performa Finansial Global per Episode', fontsize=16, fontweight='bold')
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
        fig.tight_layout()
        plt.savefig(save_dir / "performa_finansial.png")
        plt.show()

        # --- Plot 3: Sisa Anggaran per Agen ---
        plt.figure(figsize=(12, 7))
        budget_cols = [col for col in df.columns if 'budget_left' in col]
        agent_names_budget = [col.replace('_budget_left', '').replace('_', ' ').title() for col in budget_cols]
        
        for i, col in enumerate(budget_cols):
            plt.plot(df['episode'], df[col], 'o-', label=agent_names_budget[i])
            
        plt.title('Sisa Anggaran Agen di Akhir Episode', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Sisa Anggaran (Rp)', fontsize=12)
        plt.xticks(df['episode'])
        plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {int(x/1000)}k'))
        plt.legend(title='Strategi Agen')
        plt.tight_layout()
        plt.savefig(save_dir / "sisa_anggaran.png")
        plt.show()


def main():
    """Fungsi utama untuk menjalankan demo."""
    print("🎯 Multi-Agent E-commerce Retargeting Simulator Demo")
    print("=" * 60)
    
    # Cek keberadaan file data dan model
    # Ganti dengan path yang sesuai jika perlu
    if not Path("avazu_dev_pro.parquet").exists():
        print("❌ Error: File data 'avazu_dev_pro.parquet' tidak ditemukan!")
        sys.exit(1)
    if not Path("ctr_model_pro_two.txt").exists():
        print("❌ Error: File model 'ctr_model_pro_two.txt' tidak ditemukan!")
        sys.exit(1)
    
    # Buat dan jalankan demo
    demo = MultiAgentDemo(num_episodes=5, episode_length=500)
    demo.run_demo()
    
    print("\n🎉 Demo berhasil diselesaikan!")

if __name__ == "__main__":
    main()