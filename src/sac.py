import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model.net import CNNPolicy, QNetwork
from model.utils import soft_update, hard_update



class SAC(object):
    def __init__(self, num_inputs, action_space, param, action_bound):

        self.action_bound = action_bound
        self.alpha = param['alpha']
        self.gamma = param['gamma']
        self.tau = param['tau']
        self.target_update_interval = param['target_update_interval']
        self.automatic_entropy_tuning = param['automatic_entropy_tuning']
        self.lr =param['lr'] 
        
        self.device = torch.device("cuda" if param['cuda'] else "cpu")

        self.critic = QNetwork(num_inputs, action_space).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(num_inputs, action_space).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = -dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=param['lr'])

        self.policy = CNNPolicy(num_inputs, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=param['lr'])


    def select_action(self, state, evaluate=False):
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        obs_stack, s_list, speed = state
        obs_stack = Variable(torch.from_numpy(obs_stack)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed = Variable(torch.from_numpy(speed)).float().cuda()

        if evaluate is False:
            action, _, _ = self.policy.sample(obs_stack, goal_list, speed)
        else:
            _, _, action = self.policy.sample(obs_stack, goal_list, speed)
        
        a = a.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=slef.action_bound[0], a_max=self.action_bound[1])
        
        return scaled_action


    def update_parameters(self, memory, updates):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory
        obss, goals, speeds = state_batch
        obss_, goals_, speeds_ = next_state_batch


        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
      
        obss = torch.FloatTensor(obss).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        speeds = torch.FloatTensor(speeds).to(self.device)
        obss_ = torch.FloatTensor(obss_).to(self.device)
        goals_ = torch.FloatTensor(goals_).to(self.device)
        speeds_ = torch.FloatTensor(speeds_).to(self.device)

  
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(obss_, goals_, speeds_)
            qf1_next_target, qf2_next_target = self.critic_target(obss_, goals_, speeds_, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + done_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(obss, goals, speeds, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        pi, log_pi, _ = self.policy.sample(obss, goals, speeds)

        qf1_pi, qf2_pi = self.critic(obss, goals, speeds, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def get_policy_state_dict(self):
        return self.policy.state_dict()
    
    def load_policy_state_dict(self, statedict):
        self.policy.load_state_dict(statedict)


    def load_model(self, actor_path, critic_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)




