import torch
import random as rand
import numpy as np
from collections import deque
from gameboard import Gameboard
from model import QNet, QTrainer


MAX_MEMORY = 1000
BATCH_SIZE = 10
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games
        self.epsilon = 1.0
        self.gamma = 0.95
        self.player
        self.memory = deque(maxlen = MAX_MEMORY) #pop_left if too many
        self.model = QNet(1, 64, 64, 7) #TODO
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma) #TODO
        #TODO model and trainer
    
    def get_state(self, board):
        return board.get_state()
    
    def remember(self, player, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #pop left if MAX_MEMORY is reached
    
    def train_long(self, player, state, action, reward, next_state, done):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
            
    
    def train_short(self, player, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, player, state):
        #random moves sometimes
        self.epsilon = self.epsilon * 0.995
        if random.random() < self.epsilon:
            column = rand.randint(0,6)
        else:
            state0 = torch.tensor(state, dtype = object)
            prediction = self.model(state0)
            column = torch.argmax(prediction).item()
        return column
            
    

def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    board = Gameboard()
    n = 0
    while True():
        #get old state
        if n % 2 == 0:
            player = 'red'
        else:
            player = 'yellow'
        state_old = agent.get_state(board)
        column = agent.get_action(player, state)
        
        #perform move and get new state
        reward, done, winner = board.turn(player, column)
        state_new = agent.get_state(game)
        
        #train short
        agent.train_short(player, state_old, final_move, state_new, done)
        
        agent.remember(player, state_old, final_move, state_new, done)
        
        if done:
            #train long memory
            board.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if n%25 == 0:
                agent.model.save()y
            
            print('Game', agent.n_games, 'Winner', winner)
            
            #TODO plot
        
        n += 1

if __name__ == 'train':
    train()
    