from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, use_tensorboard=True, log_dir='runs'):

        if use_tensorboard:
            self.use_tensorboard = use_tensorboard
            self.writer = SummaryWriter(log_dir)
        
        self.ep_reward = 0.0
        self.ep_loss = 0
        self.ep_num_step = 0.0
        self.current_step = 0.0
        self.episode_count = 0
    
    def log_step(self, reward, loss, q):
        self.ep_reward +=reward
        self.ep_num_step +=1

        if loss:
            self.ep_loss +=loss
        self.current_step +=1
    
    def log_episode(self, exploration_rate, time):

        if self.use_tensorboard:
            self.writer.add_scalar('Train/Ep_Reward',
                self.ep_reward, self.current_step)
            self.writer.add_scalar('Train/Ep_length',
                self.ep_num_step, self.current_step)
            self.writer.add_scalar('Train/avg_loss',
                self.ep_loss/self.ep_num_step, self.current_step)
            self.writer.add_scalar('Train/step_per_sec',
                self.current_step/time, self.current_step)
            self.writer.add_scalar('Train/Ep_exp_rate',
                exploration_rate, self.current_step)
        self.ep_reward = 0.0
        self.ep_loss = 0
        self.ep_num_step = 0.0
        self.episode_count += 1                
