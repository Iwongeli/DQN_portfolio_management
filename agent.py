import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import DQN  # Sieć neuronowa

class DQNAgent:
    def __init__(self, 
                 state_size, 
                 action_size, 
                 gamma=0.99, 
                 lr=0.001, 
                 batch_size=128, 
                 epsilon=1.0, 
                 epsilon_decay=0.999, 
                 epsilon_min=0.1, 
                 tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Współczynnik dyskontowania przyszłych nagród
        self.lr = lr  # Learning rate
        self.batch_size = batch_size
        self.epsilon = epsilon  # Początkowa eksploracja
        self.epsilon_decay = epsilon_decay  # Stopniowa redukcja epsilon
        self.epsilon_min = epsilon_min  # Minimalna eksploracja
        self.tau = tau  # 🆕 Współczynnik do soft update w Double DQN
        self.memory = deque(maxlen=15000)  # 🆕 Powiększony Replay Buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🏗 Tworzymy dwie sieci neuronowe
        self.model = DQN(state_size, action_size).to(self.device)  # Sieć główna
        self.target_model = DQN(state_size, action_size).to(self.device)  # Sieć docelowa
        self.target_model.load_state_dict(self.model.state_dict())  # Kopiujemy wagi
        self.target_model.eval()  # Sieć docelowa tylko do ewaluacji

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Przechowywanie doświadczeń w pamięci Replay Buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
        
    def act(self, state):
        """Wybór akcji zgodnie z polityką epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)  # 🏆 Eksploracja daje wartości w zakresie (-1,1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        
        return q_values.cpu().numpy().squeeze()  # 🏆 Model może zwrócić wartości pośrednie


    def replay(self):
        """Trenuj model na losowo wybranych doświadczeniach z pamięci Replay Buffer"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)  # 🏆 Akcje są ciągłe!
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)  # 🏆 Dopasowanie wymiaru [batch_size, 1]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)  # 🏆 Konwersja na [batch_size, 1]

        # 🏆 Obliczamy przewidywane wartości Q dla obecnego stanu
        q_values = self.model(states)

        # 🏆 Obliczamy wartości Q dla następnych stanów (ale bez gradientów)
        next_q_values = self.target_model(next_states).detach()  # 🏆 Bez użycia `.max()`, bo mamy akcje ciągłe

        # 🏆 Aktualizacja wartości Q przy użyciu Bellmana
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # 🏆 Bellman Equation

        # 🏆 Upewniamy się, że wymiary pasują do MSELoss
        loss = self.loss_fn(q_values, target_q_values)
        
        #print(f"🔍 q_values.shape: {q_values.shape}, target_q_values.shape: {target_q_values.shape}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        loss.backward()
        self.optimizer.step()

        # Stopniowe zmniejszanie epsilon (eksploracja)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
    def update_target_network(self):
        """Soft update sieci docelowej"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def save_model(self, filename="dqn_trading_model.pth"):
        """Zapisuje model do pliku"""
        torch.save(self.model.state_dict(), filename)
        print(f"✅ Model zapisany jako {filename}")


    def load_model(self, filename="dqn_trading_model.pth"):
        """Wczytuje model z pliku"""
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Tryb ewaluacji
        print(f"🔄 Model wczytany z {filename}")
