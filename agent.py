# -*- coding: utf-8 -*-
import pickle
import bz2
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
from kaggle_environments import make
import os
from helper import plot

# hyper parameters
input_size = 77 # 11 * 7 (field size)
hidden_sizes = [512, 256, 128, 64]
epochs = 1000
learning_rate = 0.001
gamma = 0.98 # discount rate

classes = ['NORTH', 'EAST','SOUTH','WEST']
num_classes = len(classes) # Actions

MAX_MEMORY = 500000
BATCH_SIZE = 1000


class LinearGooseNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.hidden_layers = []

        # input layer
        self.linearInput = nn.Linear(input_size, hidden_sizes[0])

        # hidden layers
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 

        #output layer
        self.linearOutput = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x):
        x = F.relu(self.linearInput(x))
        for i in range(len(self.hidden_layers)):
            x = F.relu(self.hidden_layers[i](x))
        x = self.linearOutput(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #(n, x)

        if  len(state.shape) == 1:
            #(1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            done = (done, )

        # 1: predicted Q value with current state
        pred = self.model(state)
        #print(pred)

        # 2: Q_new = r + y * max(next predicted Q value) -> only if not done
        # pred.clone()
        # preds[argmax(actions)] = Q_new
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx] :
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state[idx]))

        target[idx][torch.argmax(action).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class GooseAgent:

    def __init__(self, input_size, hidden_sizes, num_classes):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = LinearGooseNet(input_size, hidden_sizes, num_classes)
        self.trainer = QTrainer(self.model, lr=learning_rate, gamma=self.gamma)
        model_folder_path = './model'
        model_file_name = 'model.pth'
        model_file_name = os.path.join(model_folder_path, 'model.pth')
         
        #state_dict = pickle.loads(bz2.decompress(base64.b64decode(PARAM)))
        if os.path.isfile(model_file_name):
            print('Loading model')
            self.model.load_state_dict(torch.load(model_file_name))
            self.model.eval()
            #print(base64.b64encode(pickle.dumps(self.model.state_dict())))


    def get_state(self, observation, configuration):
        state = np.zeros([7, 11])
        my_index = observation.index

        # place food
        for food_index in range(len(observation.food)):
            food = observation.food[food_index]
            food_row , food_column = row_col(food, configuration.columns)
            state[food_row , food_column] = 4
        
        # place geese
        for goose_index in range(len(observation.geese)):
            goose = observation.geese[goose_index]
            if(len(goose)>0):
                goose_head = goose[0]
                head_row , head_column = row_col(goose_head, configuration.columns)
                if goose_index == my_index:
                    state[head_row , head_column] = 3 # my head
                else:
                    state[head_row , head_column] = 2
                # place body
                for body in range(1, len(goose)):
                    body_row , body_column = row_col(goose[body], configuration.columns)
                    state[body_row , body_column] = 1 #

        #print(state)
        #print()
        return np.array(state.flatten(), dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves : tradeoff exploration / exploitation
        self.epsilon = 200 - self.n_games
        final_move = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move



def agent(obs_dict, config_dict):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    food = observation.food[0]
    food_row, food_column = row_col(food, configuration.columns)

    if food_row > player_row:
        return Action.SOUTH.name
    if food_row < player_row:
        return Action.NORTH.name
    if food_column > player_column:
        return Action.EAST.name
    return Action.WEST.name

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    env = make("hungry_geese") # , debug=True)
    trainer = env.train([None, "greedy", "greedy", "greedy"])
 #   trainer = env.train([None])
    my_index=0

    obs_dict = trainer.reset()
    config_dict = env.configuration
    config = Configuration(config_dict)

    agent = GooseAgent(input_size, hidden_sizes, num_classes)
    #game = SnakeGameAI()
    while True:
        #observation = Observation(obs_dict)
        # get old state
        state_old = agent.get_state(Observation(obs_dict), config)
        

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        #reward, done, score = game.play_step(final_move)
        obs_dict, reward, done, info = trainer.step(Action[classes[final_move.index(1)]].name)
        state_new = agent.get_state(Observation(obs_dict), config)

        if done:
            my_score = env.state[my_index]['reward']
            my_place = sum([g['reward']>my_score for g in env.state])
            reward = reward*10**(4-my_place)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        

        if done:
            my_score = env.state[my_index]['reward']
            my_place = sum([g['reward']>my_score for g in env.state])
            print(f"your goose died on step {env.state[0]['observation']['step']+1} (placed: {my_place+1}; scores:{[g['reward'] for g in env.state]})")
            # train long memory and plot result
            trainer.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if my_score > record:
                record = my_score
                agent.model.save()

            #print(f"your goose died on step {env.state[0]['observation']['step']+1} (placed: {my_place+1}; scores:{[g['reward'] for g in env.state]})")
            #print("Game: ", agent.n_games, " Score: ", my_score, " Record: ", record)

            plot_scores.append(my_score)
            total_score += my_score
            mean_score = total_score / agent.n_games
#            plot_mean_scores.append(mean_score)
#            plot(plot_scores, plot_mean_scores)
        #env.render(mode="ipython")
if __name__ == '__main__':
    train()
