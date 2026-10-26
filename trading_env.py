import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces

class TransferMarketEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, ticker="AAPL"):
        super(TransferMarketEnv, self).__init__()
        
        print(f"Downloading market data for {ticker}")
        data = yf.download(ticker, start="2025-01-01", end="2026-10-01", progress=False)
        self.prices = data['Close'].values.flatten().astype(np.float32)
        
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.holdings = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.balance, float(self.holdings), self.prices[self.current_step]], dtype=np.float32)

    def step(self, action):
        current_price = float(self.prices[self.current_step])
        reward = 0.0
        
        if action == 1 and self.balance >= current_price:
            units_to_buy = min(5, int(self.balance // current_price))
            if units_to_buy > 0:
                self.holdings += units_to_buy
                self.balance -= units_to_buy * current_price
                reward = 0.5  
        
        elif action == 2 and self.holdings > 0:
            units_to_sell = min(5, self.holdings)
            self.holdings -= units_to_sell
            self.balance += units_to_sell * current_price
            prev_price = float(self.prices[max(0, self.current_step - 1)])
            reward = float((current_price - prev_price) * units_to_sell)

        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        
        if terminated:
            net_worth = self.balance + (self.holdings * current_price)
            reward = float(net_worth - self.initial_balance)
            
        return self._get_obs(), float(reward), terminated, False, {}
