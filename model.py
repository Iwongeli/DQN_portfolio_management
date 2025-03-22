import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # 🔥 Trzy warstwy ukryte
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)  # Warstwa wyjściowa

        # 🔄 Normalizacja warstw
        self.bn1 = nn.LayerNorm(256)
        self.bn2 = nn.LayerNorm(128)
        self.bn3 = nn.LayerNorm(64)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))  # 🔄 Zamiana Tanh na ReLU dla lepszego uczenia
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return torch.tanh(self.fc4(x))  # 🔥 Wynik w zakresie [-1, 1]
