import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces

class TransferMarketEnv(gym.Env):
    """Custom RL Environment for digital asset transfer markets."""
    metadata = {'render_modes': ['human']}

    def __init__(self, ticker="AAPL"):
        super(TransferMarketEnv, self).__init__()
        
        print(f"Downloading market data for {ticker}...")
        data = yf.download(ticker, start="2025-01-01", end="2026-10-01", progress=False)
        self.prices = data['Close'].values.flatten().astype(np.float32)
        
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.holdings = 0

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        # State: [Current Balance, Current Holdings, Current Price]
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
        
        # Buy: buy as many whole units as feasible (up to 5 per step)
        if action == 1 and self.balance >= current_price:
            units_to_buy = min(5, int(self.balance // current_price))
            if units_to_buy > 0:
                self.holdings += units_to_buy
                self.balance -= units_to_buy * current_price
                reward = 0.5  # small incentive for active trading
        
        # Sell: liquidate up to 5 units
        elif action == 2 and self.holdings > 0:
            units_to_sell = min(5, self.holdings)
            self.holdings -= units_to_sell
            self.balance += units_to_sell * current_price
            prev_price = float(self.prices[max(0, self.current_step - 1)])
            reward = float((current_price - prev_price) * units_to_sell)

        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        
        # Final net profit reward at the end of the episode
        if terminated:
            net_worth = self.balance + (self.holdings * current_price)
            reward = float(net_worth - self.initial_balance)
            
        return self._get_obs(), float(reward), terminated, False, {}