import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOPICS, MODELS_DIR
from rl.candidate_profiles import CANDIDATE_PROFILES
from rl.interview_env import InterviewEnv

class InterviewPolicy(nn.Module):
    def __init__(self, state_dim: int, num_topics: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_topics),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

def train_policy(num_episodes: int = 1000):
    num_topics = len(TOPICS)
    state_dim = num_topics * 2 + 2
    policy = InterviewPolicy(state_dim, num_topics)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.95
    
    profile_names = list(CANDIDATE_PROFILES.keys())
    
    print("[Phase 4] Training RL Policy...")
    for episode in range(num_episodes):
        prof_name = np.random.choice(profile_names)
        env = InterviewEnv(CANDIDATE_PROFILES[prof_name], TOPICS)
        state = env.reset()
        
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            state_t = torch.tensor(state)
            probs = policy(state_t)
            m = Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            
            state, reward, done = env.step(action.item())
            rewards.append(reward)
            
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()
            
        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
            
        loss = torch.stack(loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Average Reward: {np.mean(rewards):.2f}")
            
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, 'interview_policy.pt')
    torch.save(policy.state_dict(), save_path)
    print(f"Policy saved to {save_path}")

if __name__ == "__main__":
    train_policy()
