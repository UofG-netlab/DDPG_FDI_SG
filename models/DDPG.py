import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim = 1, hidden_dim = 128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim = 1, hidden_dim = 128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim =1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class MultiAgentDDPGTrainer:
    def __init__(self, state_dim, action_dim, trafo_indices, gamma = 0.99, tau = 0.005, actor_lr = 1e-4, critic_lr = 1e-3, batch_size = 64):
        self.agents = {}
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        for idx in trafo_indices:
            actor = Actor(state_dim, action_dim)
            target_actor = Actor(state_dim, action_dim)
            target_actor.load_state_dict(actor.state_dict())

            critic = Critic(state_dim, action_dim)
            target_critic = Critic(state_dim, action_dim)
            target_critic.load_state_dict(critic.state_dict())

            agent = {
                "actor": actor,
                "target_actor": target_actor,
                "critic": critic,
                "target_critic": target_critic,
                "actor_optimizer": optim.Adam(actor.parameters(), lr=actor_lr),
                "critic_optimizer": optim.Adam(critic.parameters(), lr=critic_lr),
                "memory": [],
                "loss_history": []
            }
            self.agents[idx] = agent

    def select_action(self, idx, state, noise_std=0.2):
        agent = self.agents[idx]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape: [1, state_dim]
        with torch.no_grad():
            action_tensor = agent["actor"](state_tensor)  # shape: [1, 1]
        action = action_tensor[0, 0].item()
        action += np.random.normal(0, noise_std)
        return np.clip(action, 0.0, 1.0)

    def store_experience(self, idx, state, action, reward, next_state, done):
        actual_temp = state[2] if len(state) > 2 else state[-1]
        is_overheat = actual_temp > 90
        times = 3 if is_overheat else 1
        for _ in range(times):
            self.agents[idx]["memory"].append((state, action, reward, next_state, done))
            if len(self.agents[idx]["memory"]) > 1000:
                self.agents[idx]["memory"].pop(0)
        #
        # self.agents[idx]["memory"].append((state, action, reward, next_state, done))
        # if len(self.agents[idx]["memory"])>10000:
        #     self.agents[idx]["memory"].pop(0)

    def train(self, idx):
        agent = self.agents[idx]
        if len(agent["memory"]) < self.batch_size:
            return

        batch = random.sample(agent["memory"], self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_actions = agent["target_actor"](next_states)
            target_q = agent["target_critic"](next_states, next_actions)
            expected_q = rewards + self.gamma * target_q * (1.0 - dones)

        current_q = agent["critic"](states, actions)
        critic_loss = nn.MSELoss()(current_q, expected_q)
        agent["critic_optimizer"].zero_grad()
        critic_loss.backward()
        agent["critic_optimizer"].step()

        predicted_actions = agent["actor"](states)
        actor_loss = -agent["critic"](states, predicted_actions).mean()
        agent["actor_optimizer"].zero_grad()
        actor_loss.backward()
        agent["actor_optimizer"].step()

        agent["loss_history"].append((critic_loss.item(), actor_loss.item()))

        self.soft_update(agent["target_actor"], agent["actor"])
        self.soft_update(agent["target_critic"], agent["critic"])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_all_models(self, prefix="./models_ddpg"):
        os.makedirs(prefix, exist_ok=True)
        for idx, agent in self.agents.items():
            torch.save(agent["actor"].state_dict(), f"{prefix}/actor_trafo_{idx}.pth")
            torch.save(agent["critic"].state_dict(), f"{prefix}/critic_trafo_{idx}.pth")

    def load_all_models(self, prefix="./models_ddpg"):
        for idx, agent in self.agents.items():
            agent["actor"].load_state_dict(torch.load(f"{prefix}/actor_trafo_{idx}.pth"))
            agent["critic"].load_state_dict(torch.load(f"{prefix}/critic_trafo_{idx}.pth"))
            agent["target_actor"].load_state_dict(agent["actor"].state_dict())
            agent["target_critic"].load_state_dict(agent["critic"].state_dict())

    def learn_all(self):
        for idx in self.agents:
            self.train(idx)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        # for idx, agent in self.agents.items():
        #     loss_list = agent["loss_history"]
        #     if loss_list:
        #         plt.plot(loss_list, label=f"Transformer {idx}")
        plt.figure(figsize=(8, 5))
        for idx, agent in self.agents.items():
            critic_losses, actor_losses = zip(*self.agents[idx]["loss_history"])
            plt.plot(critic_losses, label=f"Critic Loss Transformer {idx}")
            plt.plot(actor_losses, label=f"Actor Loss Transformer {idx}")
        plt.title(f"Loss for Transformer")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss_curves_1(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for idx, agent in self.agents.items():
            loss_list = agent["loss_history"]
            if loss_list:
                plt.plot(loss_list, label=f"Transformer {idx}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("DQN Loss per Transformer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()






