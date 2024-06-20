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


MAX_MEMORY = 4000
BATCH_SIZE = 64
LR = 0.001
N_STEP = 4

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = 0.95
        self.memory = deque(maxlen = MAX_MEMORY) #pop_left if too many
        self.n_step_buffer = deque(maxlen = N_STEP)
        self.model = QNet(43, 64, 64, 7).to(device) #TODO
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma) #TODO
        #TODO model and trainer

    def get_state(self, board, player):
        grid_flattened = board.get_state()
        state = np.insert(grid_flattened, 0, player)
        return state


    def remember(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == N_STEP:
            state, action, _, _, _ = self.n_step_buffer[0]
            reward = sum([(-(-self.gamma)**i) * self.n_step_buffer[i][2] for i in range(N_STEP)])
            next_state, done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            self.memory.append((state, action, reward, next_state, done)) #pop left if MAX_MEMORY is reached

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = rand.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #random moves sometimes
        self.epsilon = max(self.epsilon * 0.99999, 0.2)#0.999, 0.1?

        if rand.random() < self.epsilon:
            action = rand.randint(0,6)
            return action
        else:
            #state0 = torch.tensor(state, dtype = torch.int32)
            # print('State:')
            # print(state)
            # print('Player:')
            # print(player)
            # print('Input:')
            state = torch.tensor(state, dtype=torch.float32).to(device)
            prediction = self.model(state)
            action = torch.argmax(prediction).item()
        return action

    def get_valid_action(self, board, state):
        while True:
            action = self.get_action(state)
            if board.is_valid_action(action):
                return action




def train(max_games=100):
    print('Training\n\n')
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    board = Gameboard()
    n = 0
    firsts = 0
    stacker = 1

    while (agent.n_games<max_games):
        #get old state
        n += 1
        if n % 2 == 1:
            player = 1
        else:
            player = 2

        state_old = agent.get_state(board, player)
        if agent.n_games % 25 == 3 and player == stacker:
            action = agent.n_games % 7
        else:
            action = agent.get_action(state_old)
        # if agent.get_action(state_old) == 0:
        #     firsts += 1
    #     #perform move and get new state
        # action = agent.get_valid_action(board, state_old)
        reward, done, winner = board.turn(player, action)
        player_new = 3-player
        state_new = agent.get_state(board, player_new)


    #     # #train short
        agent.train_short(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)#self, player, state, action, reward, next_state, done
        if done:
    #     #     #train long memory
    #         print(board.grid)
            #print('Game', agent.n_games, 'Winner', winner, 'wincon', board.wincon)
            # print(board)
            # print('\n')
            if agent.n_games%100==0:
                print(board)
                print(f'Game {agent.n_games}')
                print(agent.epsilon)
                # print(stacker)
            # firsts = 0
            # stacker = 3 - stacker
            
            state = agent.get_state(board, player)
            action = None
            for n in range(3):
                player = 3-player
                agent.remember(state, action, reward, state, done)#self, player, state, action, reward, next_state, done

            agent.train_long()
            board.reset()
            agent.n_step_buffer.clear()
            n = 0
            agent.n_games += 1

            if agent.n_games%25 == 0:
                agent.model.save()



    #         #TODO plot





    print('Training Completed\n')

if __name__ == '__main__':
    print('Running\n')
    train(10000)