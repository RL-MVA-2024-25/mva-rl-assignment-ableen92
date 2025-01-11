import random
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from evaluate import evaluate_HIV
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
from torch.optim.lr_scheduler import StepLR

# Wrapper pour l'environnement HIV avec un TimeLimit de 200 étapes
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, device):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def append(self, s, a, r, s_, d):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_, d))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (s, a, r, s_, d)
            self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
            torch.tensor(weights, dtype=torch.float32).to(self.device),
            indices
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-5)  # Éviter les priorités nulles

    def __len__(self):
        return len(self.buffer)

# Réseau Double DQN
class DoubleDQN(nn.Module):
    def __init__(self, env, hidden_size, depth):
        super(DoubleDQN, self).__init__()
        self.in_layer = nn.Linear(env.observation_space.shape[0], hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(hidden_size, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.out_layer(x)

# L'agent avec Double DQN et Prioritized Experience Replay
class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.batch_size = 128
        self.buffer_size = int(1e5)
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_frames = 1e5
        self.memory = PrioritizedReplayBuffer(self.buffer_size, self.alpha, self.device)
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_period = 1000
        self.hidden_size = 256
        self.depth = 3
        self.model = DoubleDQN(env, self.hidden_size, self.depth).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.95)
        self.update_target_freq = 100
        self.gradient_clip = 1.0
        self.best_score = -np.inf
        self.model_name = "best_agent"
        self.steps_done = 0

    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
            return Q.argmax(dim=1).item()

    def act(self, state):
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay_period)
        self.steps_done += 1
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model, state)

    def gradient_step(self, beta):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, beta)
        
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
            td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()
            loss = (weights * self.criterion(current_q_values, target_q_values)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.memory.update_priorities(indices, td_errors)

    def train(self, env, max_episode):
        beta = self.beta_start
        for episode in range(max_episode):
            state, _ = env.reset()
            total_reward = 0
            done, trunc = False, False

            while not (done or trunc):
                action = self.act(state)
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                total_reward += reward
                beta = min(1.0, beta + (1.0 - self.beta_start) / self.beta_frames)
                self.gradient_step(beta)
                if self.steps_done % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                state = next_state

            print(f"Episode {episode + 1}, Reward: {total_reward}")
            # Évaluation périodique pour sauvegarder le meilleur modèle
            if (episode + 1) % 10 == 0:
                eval_score = evaluate_HIV(self, nb_episode=5)
                print(f"Evaluation Score: {eval_score}")
                if eval_score > self.best_score:
                    self.best_score = eval_score
                    self.save(f"{self.model_name}.pth")
                    print(f"New best score: {self.best_score}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(f"{self.model_name}.pth", map_location=self.device))
        self.model.eval()


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(env, max_episode=10)
