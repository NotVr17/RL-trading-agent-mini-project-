import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from trading_env import TransferMarketEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(state)).to(device),
                torch.LongTensor(action).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

env = TransferMarketEnv(ticker="AAPL")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(obs_dim, n_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
criterion = nn.SmoothL1Loss() 
memory = ReplayBuffer(capacity=10000)

epochs = 300
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.02

episode_rewards = []

print("Starting training with Experience Replay")
for epoch in range(epochs):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = int(torch.argmax(policy_net(s_tensor)).item())

        next_state, reward, done, _, _ = env.step(action)
        memory.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        if len(memory) >= batch_size:
            b_state, b_action, b_reward, b_next_state, b_done = memory.sample(batch_size)
            
            q_values = policy_net(b_state).gather(1, b_action.unsqueeze(1))
            with torch.no_grad():
                next_q_values = policy_net(b_next_state).max(1)[0].unsqueeze(1)
                target_q = b_reward + (1 - b_done) * gamma * next_q_values

            loss = criterion(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if (epoch + 1) % 30 == 0:
        avg_last_30 = np.mean(episode_rewards[-30:])
        print(f"Epoch {epoch+1:3d}/{epochs} | Avg Profit (Last 30): ${avg_last_30:8.2f} | Epsilon: {epsilon:.3f}")

start_price = float(env.prices[0])
end_price = float(env.prices[-1])
units_buy_hold = int(env.initial_balance // start_price)
cash_left = env.initial_balance - (units_buy_hold * start_price)
buy_hold_final_value = cash_left + (units_buy_hold * end_price)
buy_hold_profit = buy_hold_final_value - env.initial_balance

print("\n" + "="*45)
print(f"Baseline (Buy & Hold) Profit : ${buy_hold_profit:8.2f}")
print(f"Agent Final Epoch Profit     : ${episode_rewards[-1]:8.2f}")
print(f"Agent Peak Epoch Profit      : ${max(episode_rewards):8.2f}")
print("="*45)

torch.save(policy_net.state_dict(), "trained_agent.pth")

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="DQN Agent Return ($)", color="#1f77b4", alpha=0.6)
moving_avg = np.convolve(episode_rewards, np.ones(15)/15, mode='valid')
plt.plot(range(14, len(episode_rewards)), moving_avg, label="15-Epoch Moving Average", color="#d62728", linewidth=2)
plt.axhline(y=buy_hold_profit, color="green", linestyle="--", label=f"Buy & Hold Baseline (${buy_hold_profit:.2f})")

plt.title("RL Trading Agent Training Convergence vs Baseline", fontsize=14, fontweight="bold")
plt.xlabel("Episodes", fontsize=11)
plt.ylabel("Net Profit ($)", fontsize=11)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("training_performance.png", dpi=300)
