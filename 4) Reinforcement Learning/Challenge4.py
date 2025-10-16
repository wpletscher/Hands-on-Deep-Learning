import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import flappy_bird_gymnasium

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
screen_w = 288
player_w = 34
d_max = 0.8 * screen_w - player_w


class TensorWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
      
  def observation(self, observation):
    obs_tensor = torch.tensor(observation, dtype=torch.float32)
    return obs_tensor / (obs_tensor.max() + 1e-8)

class ReplayBuffer:
  def __init__(self, capacity, batch_size):
    self.buffer = deque(maxlen=capacity)
    self.batch_size = batch_size
    
  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))
    
  def sample(self):
    batch = random.sample(self.buffer, self.batch_size)
    states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
    return states, actions, rewards, next_states, dones
    
  def __len__(self):
    return len(self.buffer)



class Agent(nn.Module):
  def __init__(self, input_dim=180, output_dim=2):
    super().__init__()
    self.policy_net = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.LayerNorm(512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.LayerNorm(256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, output_dim)
    ).to(device)
    
    self.target_net = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.LayerNorm(512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.LayerNorm(256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, output_dim)
    ).to(device)
    
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    
    self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0001)
    self.memory = ReplayBuffer(capacity=100000, batch_size=64)
    self.batch_size = 64
    self.gamma = 0.99
    self.tau = 0.005

  def act(self, observation, epsilon=0.0):
    if random.random() > epsilon:
      with torch.no_grad():
        if not isinstance(observation, torch.Tensor):
          observation = torch.tensor(observation, dtype=torch.float32).to(device)
        else:
          observation = observation.to(device)
        
        q_values = self.policy_net(observation)
        return q_values.argmax().item()
    else:
      return random.randint(0, 1)  # Random action (0 or 1)
  
  def learn(self):
    if len(self.memory) < self.batch_size:
      return None
    
    states, actions, rewards, next_states, dones = self.memory.sample()
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
      next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
      next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
      target_q = rewards + self.gamma * next_q * (1 - dones)
    
    loss = F.smooth_l1_loss(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
    self.optimizer.step()
    
    for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
      target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    return loss.item()




def apply_wrappers(env):
  return TensorWrapper(env)




def init_model(train_env):
  input_dim = 180
  output_dim = 2
  
  agent = Agent(input_dim, output_dim)
  return agent




def train_model(agent, env):
  env = apply_wrappers(env)

  # Training parameters
  num_episodes = 301
  epsilon_start = 0.4
  epsilon_end = 0.01
  epsilon_decay = 0.9
  epsilon = epsilon_start
  
  best_reward = -float('inf')
  
  for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
      action = agent.act(state, epsilon)
      
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      
      if isinstance(state, torch.Tensor):
        state_np = state.cpu().numpy()
      else:
        state_np = state
        
      if isinstance(next_state, torch.Tensor):
        next_state_np = next_state.cpu().numpy()
      else:
        next_state_np = next_state
      
      agent.memory.push(state_np, action, reward, next_state_np, done)
      
      agent.learn()
      
      state = next_state
      total_reward += reward
    
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    if episode % 100 == 0:
      print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
      
      if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.policy_net.state_dict(), 'policy_net.pth')
        print(f"New best reward: {best_reward}. Model saved.")
  
  try:
    agent.policy_net.load_state_dict(torch.load('policy_net.pth'))
    print("Loaded best model for evaluation.")
  except:
    print("Could not load saved model. Using current model.")
  
  #generate_gif(env, agent, n_frames=300)
  return agent


