import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Sigmoid()

        self.fc.bias.data.fill_(2.0)

    def forward(self, state_seq, hidden=None):
        lstm_out, hidden = self.lstm(state_seq, hidden)
        last_output = lstm_out[:, -1, :]  # take the last timestep
        action = self.activation(self.fc(last_output))  # output disconnect probability
        return action, hidden

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq, action, hidden=None):
        lstm_out, hidden = self.lstm(state_seq, hidden)
        last_output = lstm_out[:, -1, :]  # [B, H]
        x = torch.cat([last_output, action], dim=-1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value, hidden


class MultiAgentLSTMDDPGTrainer:
    def __init__(self, state_dim, action_dim, trafo_indices, gamma=0.99, tau=0.005,
                 actor_lr=1e-4, critic_lr=1e-3, batch_size=64):
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

            self.agents[idx] = {
                "actor": actor,
                "target_actor": target_actor,
                "critic": critic,
                "target_critic": target_critic,
                "actor_optimizer": optim.Adam(actor.parameters(), lr=actor_lr),
                "critic_optimizer": optim.Adam(critic.parameters(), lr=critic_lr),
                "memory": [],
                "loss_history": [],
                "actor_hidden": None,
                "critic_hidden": None
            }

    def reset_hidden_states(self):
        for agent in self.agents.values():
            agent["actor_hidden"] = None
            agent["critic_hidden"] = None

    def select_action(self, idx, state_seq, noise_std=0.2):
        agent = self.agents[idx]
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # [1, T, D]
        with torch.no_grad():
            action_tensor, hidden = agent["actor"](state_tensor, agent["actor_hidden"])
            agent["actor_hidden"] = hidden
        action = action_tensor[0, 0].item() + np.random.normal(0, noise_std)
        return np.clip(action, 0.0, 1.0)

    def store_experience(self, idx, state_seq, action, reward, next_state_seq, done):
        self.agents[idx]["memory"].append((state_seq, action, reward, next_state_seq, done))
        if len(self.agents[idx]["memory"]) > 10000:
            self.agents[idx]["memory"].pop(0)

    def train(self, idx):
        agent = self.agents[idx]
        if len(agent["memory"]) < self.batch_size:
            return

        batch = random.sample(agent["memory"], self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.stack(states))         # [B, T, D]
        next_states = torch.FloatTensor(np.stack(next_states))
        actions = torch.FloatTensor(actions).unsqueeze(1)    # [B, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)    # [B, 1]
        dones = torch.FloatTensor(dones).unsqueeze(1)        # [B, 1]

        with torch.no_grad():
            next_actions, _ = agent["target_actor"](next_states)
            target_q, _ = agent["target_critic"](next_states, next_actions)
            expected_q = rewards + self.gamma * target_q * (1.0 - dones)

        current_q, _ = agent["critic"](states, actions)
        critic_loss = nn.MSELoss()(current_q, expected_q)
        agent["critic_optimizer"].zero_grad()
        critic_loss.backward()
        agent["critic_optimizer"].step()

        predicted_actions, _ = agent["actor"](states)
        actor_loss = -agent["critic"](states, predicted_actions)[0].mean()
        agent["actor_optimizer"].zero_grad()
        actor_loss.backward()
        agent["actor_optimizer"].step()

        agent["loss_history"].append((critic_loss.item(), actor_loss.item()))

        self.soft_update(agent["target_actor"], agent["actor"])
        self.soft_update(agent["target_critic"], agent["critic"])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_all_models(self, prefix="./models_lstm_ddpg"):
        os.makedirs(prefix, exist_ok=True)
        for idx, agent in self.agents.items():
            torch.save(agent["actor"].state_dict(), f"{prefix}/actor_trafo_{idx}.pth")
            torch.save(agent["critic"].state_dict(), f"{prefix}/critic_trafo_{idx}.pth")
        print(f"✅ All models saved to: {prefix}")

    def load_all_models(self, prefix="./models_lstm_ddpg"):
        for idx, agent in self.agents.items():
            actor_path = f"{prefix}/actor_trafo_{idx}.pth"
            critic_path = f"{prefix}/critic_trafo_{idx}.pth"
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                agent["actor"].load_state_dict(torch.load(actor_path))
                agent["critic"].load_state_dict(torch.load(critic_path))
                agent["target_actor"].load_state_dict(agent["actor"].state_dict())
                agent["target_critic"].load_state_dict(agent["critic"].state_dict())
                print(f"🔁 Loaded model for Trafo {idx}")
            else:
                print(f"⚠️ Model not found for Trafo {idx}, skipping.")

    def learn_all(self):
        for idx in self.agents:
            self.train(idx)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        for idx, agent in self.agents.items():
            losses = agent["loss_history"]
            if losses:
                critic_losses, actor_losses = zip(*losses)
                plt.plot(critic_losses, label=f"Critic T{idx}", linestyle='--')
                plt.plot(actor_losses, label=f"Actor T{idx}")
        plt.title("Training Loss per Transformer")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
