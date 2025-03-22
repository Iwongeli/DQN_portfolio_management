import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.bn1 = nn.LayerNorm(256)
        self.bn2 = nn.LayerNorm(128)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))  # Skala [-1, 1]


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()

        # Wejście: state + action
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.bn1 = nn.LayerNorm(256)
        self.bn2 = nn.LayerNorm(128)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Sklejamy stan i akcję
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)  # Wartość Q (bez aktywacji)
