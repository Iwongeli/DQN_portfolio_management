import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from model import Actor, Critic

class DDPGAgent:
    def __init__(self, state_size, action_size, gamma=0.99, tau=0.005, lr_actor=1e-4, lr_critic=1e-3, batch_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = Actor(state_size, action_size).to(self.device)
        self.target_actor = Actor(state_size, action_size).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic
        self.critic = Critic(state_size, action_size).to(self.device)
        self.target_critic = Critic(state_size, action_size).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.memory = deque(maxlen=20000)

        # Szum eksploracyjny
        self.noise_std = 0.2  # Gaussowski
        self.noise_clip = 0.5

        self.loss_fn = nn.MSELoss()

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().squeeze()
        self.actor.train()

        if add_noise:
            noise = np.clip(np.random.normal(0, self.noise_std, size=self.action_size), -self.noise_clip, self.noise_clip)
            action += noise

        return np.clip(action, -1, 1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # 🎯 Krytyk
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions).detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 🎯 Aktor (maksymalizujemy Q)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 🔄 Soft update sieci docelowych
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, actor_path="ddpg_actor.pth", critic_path="ddpg_critic.pth"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"✅ Modele zapisane: {actor_path}, {critic_path}")

    def load_model(self, actor_path="ddpg_actor.pth", critic_path="ddpg_critic.pth"):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.eval()
        self.critic.eval()
        print(f"🔄 Modele wczytane z: {actor_path}, {critic_path}")
