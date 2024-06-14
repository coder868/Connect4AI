import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.function as F
import os

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.linear(input_size, hidden_size1)
        self.linear2 = nn.linear(hidden_size1, hidden_size2)
        self.linear3 = nn.linea(hidden_size2, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
        
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr, )
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, move, reward, next_state, done):
        state = torch.tensor(state, dtype=array)
        move = torch.tensor(move, dtype=long)
        reward = torch.tensor(reward, dtype=long)
        next_state = torch.tensor(next_state, dtype=array)
        done = torch.tensor(done, dtype = bool)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            move = torch.unsqueeze(move,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )
            
        pred = self.model(state)
        
        target = prd.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            torch[idx][torch.argmax(move).item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
    