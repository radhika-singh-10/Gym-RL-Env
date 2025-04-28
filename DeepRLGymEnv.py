import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

#to add deepSARSA,AND DEEP-DOUBLEQ results from local
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQLearning
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0  
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_target_every = 10
        self.step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Cur Q
        q_values = self.q_network(states).gather(1, actions)

        # Target Q
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

   
        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# Training loop
env = LawnmowerGridWorldEnvironment(max_timesteps=20)
state_dim = env.observation_space.n  
action_dim = env.action_space.n     
agent = DQNAgent(state_dim, action_dim)

num_episodes = 500

for episode in range(num_episodes):
    agent_pos, state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_agent_pos, next_state = env.step(action)[1], env.step(action)[0]
        reward = env.step(action)[1]  
        done = env.step(action)[2]
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

