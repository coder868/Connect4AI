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
    
    def get_state(self, board):
        return board.get_state()
    
    def remember(self, player, state, action, reward, next_state, done):
        self.memory.append((player, state, action, reward, next_state, done)) #pop left if MAX_MEMORY is reached
    
    # def train_long(self, player):
    #     if len(self.memory) > BATCH_SIZE:
    #         mini_sample = rand.sample(self.memory, BATCH_SIZE) #returns list of tuples
    #     else:
    #         mini_sample = self.memory
    #     states, actions, rewards, next_states, dones = zip(*mini_sample)
    #     self.trainer.train_step(player, states, actions, rewards, next_states, dones)
            
    
    def train_short(self, player, state, action, reward, next_state, done):
        self.trainer.train_step(player, state, action, reward, next_state, done)
    
    def get_action(self, player, state):
        #random moves sometimes
        self.epsilon = self.epsilon * 0.995
        if rand.random() < self.epsilon:
            print('rand')
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
            input = np.insert(state, 0, player)
            # print('Input')
            input = torch.tensor(input, dtype=torch.float)
            prediction = self.model(input)
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
        print('Turn:')
        print(n)
        print(board.grid)
        #get old state
        if n % 2 == 0:
            player = 1
        else:
            player = 2
        state_old = agent.get_state(board)
        action = agent.get_action(player, state_old)
        
        #perform move and get new state
        reward, done, winner = board.turn(player, action)
        state_new = agent.get_state(board)

        
        # #train short
        agent.train_short(player, state_old, action, reward, state_new, done)
        agent.remember(player, state_old, action, reward, state_new, done)#self, player, state, action, reward, next_state, done
        input = np.insert(board.grid, 0, player)
        print(input)
        if done:
        #     #train long memory
            board.reset()
            agent.n_games += 1
            print('Game:')
            print(agent.n_games)
        #     agent.train_long(player)
            
        #     if n%25 == 0:
        #         agent.model.save()
            
            print('Game', agent.n_games, 'Winner', winner)
            
            #TODO plot
        
        n += 1
print('Running\n')
if __name__ == '__main__':
    train() 