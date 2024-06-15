import torch
import random as rand
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gameboard import Gameboard
from model import QNet, QTrainer
import time


MAX_MEMORY = 1000
BATCH_SIZE = 10
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = 0.95
        self.memory = deque(maxlen = MAX_MEMORY) #pop_left if too many
        self.model = QNet(43, 64, 64, 7) #TODO
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma) #TODO
        #TODO model and trainer
    
    def get_state(self, board, player):
        grid_flattened = board.get_state()
        state = np.insert(grid_flattened, 0, player)
        return state
        
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #pop left if MAX_MEMORY is reached
    
    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = rand.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory
        players, states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(players, states, actions, rewards, next_states, dones)
            
    
    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        #random moves sometimes
        self.epsilon = self.epsilon * 0.995
        if rand.random() < self.epsilon:
            action = rand.randint(0,6)
            return action
        else:
            print('actioning')
            #state0 = torch.tensor(state, dtype = torch.int32)
            # print('State:')
            # print(state)
            # print('Player:')
            # print(player)
            # print('Input:')
            state = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state)
            action = torch.argmax(prediction).item()
        return action
            
    

def train(max_games=10):
    print('Training\n\n')
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    board = Gameboard()
    n = 0
    while (agent.n_games<max_games):
        #get old state
        if n % 2 == 0:
            player = 1
        else:
            player = 2
        state_old = agent.get_state(board, player)
        
        action = agent.get_action(state_old)
        
    #     #perform move and get new state
        reward, done, winner = board.turn(player, action)
        player_new = 3-player
        state_new = agent.get_state(board, player)

        
    #     # #train short
        agent.train_short(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)#self, player, state, action, reward, next_state, done
    #     if done:
    #     #     #train long memory
    #         print(board.grid)
    #         print('Game', agent.n_games, 'Winner', winner, 'wincon', board.wincon)
    #         board.reset()
    #         agent.n_games += 1
    #         agent.train_long()
            
    #     #     if agent.n_games%25 == 0:
    #     #         agent.model.save()
            
    #         #TODO plot
        
        print(action)
        n += 1
        
if __name__ == '__main__':
    print('Running\n')
    train()