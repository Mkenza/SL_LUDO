from collections import deque
import copy
import random
import shutil
import torch
from torch import nn

import numpy as np
import ludopy
from MCTS import MCTS
class NNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[2], out_channels=32, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28800, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        self.target = copy.deepcopy(self.main)
        for p in self.target.parameters(): # don't need gradient in the target as they're copied from main
            p.requires_grad = False
    def forward(self, input, model):
        if model == "main":
            return self.main(input)
        elif model == "target":
            return  self.target(input)

    
class DQN:
    def __init__(self, game, state_dim, action_dim):
        self.game = game
        self.player_idx = -1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.0001

        self.net = NNet(self.state_dim, self.action_dim).float()
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99995
        self.exploration_rate_min = 0.01
        self.batch_size = 256
        self.gamma = 0.99
        self.c_exp = 1
        self.current_step = 0
        self.burnin = 1000
        self.memory = deque(maxlen=1000000)
        self.nsim = 25

    def act(self, state, possible_moves):
        action_idx = -1
        while action_idx not in possible_moves:
            if np.random.rand() < self.exploration_rate:
                action_idx = np.random.randint(self.action_dim)
            else:
                state = torch.tensor(state)
                if len(state.shape) == 3:
                    state = state.unsqueeze(0)
                action_values = self.net(state, model="main")
                action_idx = torch.argmax(action_values, axis=1).item()
        
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1   
        return action_idx
    def act_mcts(self, state, possible_moves):
        args = {"numMCTSSims": self.nsim, "cpuct": self.c_exp, "gamma": self.gamma}
        mcts = MCTS(self.game, self.net, args)        
        if np.random.rand() < self.exploration_rate:
            action = random.choice(possible_moves)
        else:
            
            probs = mcts.getActionProb(state)
            if len(probs):
                action_idx = np.argmax(probs)
                action = possible_moves[action_idx]
            else:
                action = -1 
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1    
        return action       
    def collect_experience(self, state, action, next_state, reward, done):
        self.memory.append([state, next_state, [action],[reward], [done]])
    def sample(self):
        sample = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.stack, zip(*sample))
        # print(state.shape)
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor(action).long()
        reward = torch.tensor(reward)
        done = torch.tensor(done)
        # might need to change it to torch data format
        return state, next_state, action, reward, done

    def net_estimate(self, state, action):
        pred = self.net(state, model="main")[np.arange(0, self.batch_size), action]
        return pred
    def target_estimate(self, reward, next_state, done):
        next_Q = self.net(next_state, model="target")
        max_next_Q, _ = torch.max(next_Q, axis=1)
        return (reward + (1 - done.float()) * self.gamma*max_next_Q).float()
    
    def update_main_network(self, estimate, target):
        print(estimate, target)
        loss = self.loss_fn.forward(estimate, target)
        self.optimizer.zero_grad()
        loss.backward()
        print(loss)
        self.optimizer.step()
        return loss  
    def update_target_network(self):
        self.net.target.load_state_dict(self.net.main.state_dict())
    
    def learn(self):
        if self.current_step < self.burnin:
            return None, None
        if self.current_step % 4 != 0:
            return None, None
        # print("starting deep learning")
        state, next_state, action, reward, done = self.sample()
        pred = self.net_estimate(state, action)
        target = self.target_estimate(reward, next_state, done)
        loss = self.update_main_network(pred, target)

        return (pred.mean().item(), loss)
    def save_checkpoint(self, state, best = False, filename='checkpoint.pth.tar', prefix=''):
        torch.save(state, prefix + filename)
        if best:
            shutil.copyfile(prefix + filename, prefix + 'model.pth.tar')
            



        