import random
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit

# Wrapper pour l'environnement HIV avec un TimeLimit de 200 étapes
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

# Replay buffer pour stocker les transitions (s, a, r, s', d)
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)

# Réseau neuronal pour approximer la fonction Q (DQN)
class DQN(nn.Module):
    def __init__(self, env, hidden_size, depth):
        super(DQN, self).__init__()
        self.in_layer = nn.Linear(env.observation_space.shape[0], hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(hidden_size, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.out_layer(x)

# L'agent DQN
class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.95
        self.batch_size = 512
        self.buffer_size = int(1e6)
        self.memory = ReplayBuffer(self.buffer_size, self.device)
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_period = 1000
        self.epsilon_delay_decay = 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        self.hidden_size = 256
        self.depth = 5
        self.model = DQN(env, self.hidden_size, self.depth).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = nn.SmoothL1Loss()
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.update_target_strategy = "replace"  # 'replace' ou 'ema'
        self.update_target_freq = 20
        self.update_target_tau = 0.005
        self.best_score = -np.inf
        self.model_name = "best_agent"

    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation)

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode = 0
        step = 0
        epsilon = self.epsilon_max
        while episode < max_episode:
            state, _ = env.reset()
            episode_cum_reward = 0
            done = False
            trunc = False
            while not (done or trunc):
                # Mise à jour de epsilon
                if step > self.epsilon_delay_decay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
                # Action epsilon-greedy
                if np.random.rand() < epsilon:
                    action = self.act(state, use_random=True)
                else:
                    action = self.act(state, use_random=False)
                # Étape dans l'environnement
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward
                # Mise à jour du réseau
                self.gradient_step()
                # Mise à jour du réseau cible
                if self.update_target_strategy == "replace":
                    if step % self.update_target_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                elif self.update_target_strategy == "ema":
                    target_state_dict = self.target_model.state_dict()
                    model_state_dict = self.model.state_dict()
                    for key in model_state_dict:
                        target_state_dict[key] = (
                            self.update_target_tau * model_state_dict[key]
                            + (1 - self.update_target_tau) * target_state_dict[key]
                        )
                    self.target_model.load_state_dict(target_state_dict)
                state = next_state
                step += 1
            # Fin de l'épisode
            episode += 1
            print(f"Episode {episode}, Reward: {episode_cum_reward}")
            # Sauvegarde du meilleur modèle
            score_agent = evaluate_HIV(agent=self, nb_episode=1)
            if score_agent > self.best_score:
                self.best_score = score_agent
                self.save(self.model_name + ".pth")
                print(f"New best score: {self.best_score}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(f"{self.model_name}.pth", map_location=self.device))
        self.model.eval()


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(env, max_episode=1000)
