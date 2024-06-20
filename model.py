import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.relu(self.linear3(x))
        return x
        
    def save(self, file_name = 'model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    def load(self, file_name='4_step_model.pth'):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Set the model to evaluation mode
        
        
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay=1e-41 )
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):#
        
        device = next(self.model.parameters()).device
        
        state = torch.tensor(np.array(state), dtype=torch.float32).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(device)
        done = torch.tensor(np.array(done), dtype = bool).to(device)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )
        
        
        pred = self.model(state)
        
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][action[idx].item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
    