
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque


# Implementing Experience Replay
class ReplayBuffer(object):

    def __init__(self, max_size=6e2):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_orientation , batch_next_states, batch_next_orientation,  batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
        for i in ind:
            state, orientation,  next_state, next_orientation, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_orientation.append(np.array(orientation, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_next_orientation.append(np.array(next_orientation, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_orientation),np.array(batch_next_states), np.array(batch_next_orientation), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


#
# class Actor(nn.Module):
#
#   def __init__(self, state_dim, action_dim, max_action):
#     super(Actor, self).__init__()
#     self.layer_1 = nn.Linear(state_dim, 40)
#     self.layer_2 = nn.Linear(40, 30)
#     self.layer_3 = nn.Linear(30, action_dim)
#     self.max_action = max_action
#
#   def forward(self, x):
#     x = F.relu(self.layer_1(x))
#     x = F.relu(self.layer_2(x))
#     x = self.max_action * torch.tanh(self.layer_3(x))
#     return x

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        #self.pool2 = nn.MaxPool2d(2,2)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.pool3 = nn.MaxPool2d(2,2)
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(4),
            # nn.ReLU() NEVER!
        )
        self.layer_1 = nn.Linear(16 + 2, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.layer_3 = nn.Linear(8, action_dim)
        self.max_action = max_action


    def forward(self, x, o):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        #x = self.pool2(x)
        x = self.convblock4(x)
        x = self.pool3(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, o], 1)
        x = F.relu(self.layer_1(x))
        print(x)
        x = F.relu(self.layer_2(x))
        print(x)
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.pool3 = nn.MaxPool2d(2,2)
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1, 1), padding=0,),
            nn.BatchNorm2d(4)
            # nn.ReLU() NEVER!
        )


        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.pool5 = nn.MaxPool2d(2, 2)
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.pool6 = nn.MaxPool2d(2,2)
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(4)
            # nn.ReLU() NEVER!
        )

        self.layer_1 = nn.Linear(16+2+action_dim, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.layer_3 = nn.Linear(8, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(16+2+action_dim, 16)
        self.layer_5 = nn.Linear(16, 8)
        self.layer_6 = nn.Linear(8, 1)

    def forward(self, x, o, u):
        #print("x : ", x)
        #print("u : ", u)
        x1 = self.convblock1(x)
        x1 = self.convblock2(x1)
        x1 = self.pool1(x1)
        x1 = self.convblock3(x1)
        #x1 = self.pool2(x1)
        x1 = self.convblock4(x1)
        x1 = self.pool3(x1)
        x1 = self.convblock5(x1)
        x1 = self.convblock6(x1)
        x1 = x1.view( x1.size(0), -1)
        x1o = torch.cat([x1,o], 1)
        x1u = torch.cat([x1o, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(x1u))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = self.convblock7(x)
        x2 = self.convblock8(x2)
        x2 = self.pool4(x2)
        x2 = self.convblock9(x2)
        #x2 = self.pool5(x2)
        x2 = self.convblock10(x2)
        x2 = self.pool6(x2)
        x2 = self.convblock11(x2)
        x2 = self.convblock12(x2)
        x2 = x2.view( x2.size(0), -1)
        x2o = torch.cat([x2,o], 1)
        x2u = torch.cat([x2o, u], 1)
        x2 = F.relu(self.layer_4(x2u))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, o, u):
        x1 = self.convblock1(x)
        x1 = self.convblock2(x1)
        x1 = self.pool1(x1)
        x1 = self.convblock3(x1)
        x1 = self.convblock4(x1)
        x1 = self.pool1(x1)
        x1 = self.convblock5(x1)
        x1 = self.convblock6(x1)
        x1 = x1.view( x1.size(0), -1)
        x1o = torch.cat([x1,o], 1)
        x1u = torch.cat([x1o, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(x1u))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        return x1

# class Critic(nn.Module):
#
#   def __init__(self, state_dim, action_dim):
#     super(Critic, self).__init__()
#     # Defining the first Critic neural network
#     self.layer_1 = nn.Linear(state_dim + action_dim, 40)
#     self.layer_2 = nn.Linear(40, 30)
#     self.layer_3 = nn.Linear(30, 1)
#     # Defining the second Critic neural network
#     self.layer_4 = nn.Linear(state_dim + action_dim, 40)
#     self.layer_5 = nn.Linear(40, 30)
#     self.layer_6 = nn.Linear(30, 1)
#
#   def forward(self, x, u):
#     xu = torch.cat([x, u], 1)
#     # Forward-Propagation on the first Critic Neural Network
#     x1 = F.relu(self.layer_1(xu))
#     x1 = F.relu(self.layer_2(x1))
#     x1 = self.layer_3(x1)
#     # Forward-Propagation on the second Critic Neural Network
#     x2 = F.relu(self.layer_4(xu))
#     x2 = F.relu(self.layer_5(x2))
#     x2 = self.layer_6(x2)
#     return x1, x2
#
#   def Q1(self, x, u):
#     xu = torch.cat([x, u], 1)
#     x1 = F.relu(self.layer_1(xu))
#     x1 = F.relu(self.layer_2(x1))
#     x1 = self.layer_3(x1)
#     return x1

# Building the whole Training Process into a class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state, orientation):
        # state input for list state
        #state = torch.Tensor(state.reshape(1, -1)).to(device)

        # state input for image state
        state = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(device)
        orientation = torch.Tensor(orientation).unsqueeze(0).to(device)
        return self.actor(state, orientation).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

        # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_orientation, batch_next_states, batch_next_orientation, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            #for cnn state calc
            state = torch.Tensor(batch_states).unsqueeze(1).to(device)
            next_state = torch.Tensor(batch_next_states).unsqueeze(1).to(device)
            # for cnn state calc

            #for sensor state calc
            # state = torch.Tensor(batch_states).to(device)
            # next_state = torch.Tensor(batch_next_states).to(device)
            # for sensor state calc

            orientation = torch.Tensor(batch_orientation).to(device)
            #print("orientation shape : ",orientation.shape)
            next_orientation = torch.Tensor(batch_next_orientation).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)



            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state, next_orientation)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_orientation,  next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, orientation, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                # actor_loss,_ = -self.critic(state, orientation, self.actor(state))
                # actor_loss = actor_loss.mean()
                #print("debug : ",state.shape, type(state), orientation.shape, type(orientation), self.actor(state).shape, type(self.actor(state)))
                actor_loss = -self.critic.Q1(state, orientation, self.actor(state, orientation)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename = "temp", directory = "models"):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename = "temp", directory = "models"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))



    # class TD3(object):
    #
    #     def __init__(self, state_dim, action_dim, max_action, batch_size=100, discount=0.95, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, image_size = 60):
    #         #print("torch version : ",torch.__version__)
    #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
    #         self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
    #         self.actor_target.load_state_dict(self.actor.state_dict())
    #         self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    #         self.critic = Critic(state_dim, action_dim).to(self.device)
    #         self.critic_target = Critic(state_dim, action_dim).to(self.device)
    #         self.critic_target.load_state_dict(self.critic.state_dict())
    #         self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    #         self.max_action = max_action
    #         #new stuff
    #         self.replay_buffer = ReplayBuffer()
    #         self.reward_window = []
    #         self.batch_size = batch_size
    #         self.discount = discount
    #         self.tau = tau
    #         self.policy_noise = policy_noise
    #         self.noise_clip = noise_clip
    #         self.policy_freq = policy_freq
    #         #inherited stuff
    #         self.image_size = image_size
    #         # self.last_state = [[]]
    #         # for i in range(self.image_size):
    #         #     self.last_state[0].append([])
    #         #     for j in range(self.image_size):
    #         #         self.last_state[0][i].append(255)
    #
    #
    #         # self.last_state =  [[i for i in range(self.image_size)]]
    #         self.last_state = [[0,0,1,0,0]]#torch.Tensor(state_dim).unsqueeze(0)
    #         self.last_action = [0.0, 1.0]
    #         self.last_action = torch.Tensor(self.last_action)
    #         self.last_reward = 0
    #         self.update_count = 0
    #         self.start_timesteps = 300
    #         self.expl_noise = 0 #0.22
    #         self.total_timesteps = 0
    #         self.episode_reward = 0
    #         self.episode_num = 0
    #         self.episode_timesteps = 0
    #
    #
    #
    #     def select_action(self, state):
    #         state = torch.Tensor(state).unsqueeze(0).to(self.device)
    #         return self.actor(state).cpu().data.numpy().flatten()
    #
    #     def update(self, reward, new_signal, done):
    #         self.episode_reward += reward
    #         new_state = torch.Tensor(new_signal).float().unsqueeze(0)
    #         #print("buffer variable : ", self.last_state[0], new_state.tolist()[0], self.last_action, reward,done)
    #         self.replay_buffer.add((self.last_state[0], new_state.tolist()[0], self.last_action,self.last_reward,done))
    #         self.episode_timesteps += 1
    #         self.total_timesteps += 1
    #
    #         if done :
    #             # if self.episode_num in (5, 7, 9, 12, 15):
    #             #     for param in self.actor.parameters():
    #             #         print(param.data, param.data.shape)
    #             print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
    #             self.learn(self.episode_timesteps)
    #             self.episode_reward = 0
    #             self.episode_timesteps = 0
    #             self.episode_num += 1
    #         if self.total_timesteps < self.start_timesteps:
    #             #action = [random.uniform(-1.0, 1.0)]
    #             action = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
    #             print("random_action : ",action)
    #         else :
    #             action = self.select_action(new_state)
    #             print("state_action : ", action)
    #             if self.expl_noise != 0:
    #                 action = (action + np.random.normal(0, self.expl_noise, size=1)).clip([-self.max_action],[self.max_action])
    #         self.episode_reward += reward
    #         self.last_state = new_state.tolist()
    #         self.last_reward = reward
    #         self.last_action = action
    #         self.reward_window.append(reward)
    #         # if type(action) == type([]):
    #         #
    #         #     return action[0]
    #         # else :
    #         return torch.max(torch.Tensor(action),0)[0].tolist(),torch.max(torch.Tensor(action),0)[1].tolist()
    #             #return action.tolist()[0]
    #
    #
    #
    #
    #         # ## need to understand reward, new_signal ###
    #         # self.update_count += 1
    #         # #print("shape : ",new_signal.shape)
    #         # new_state = torch.Tensor(new_signal).float().unsqueeze(0)
    #         # #print("batch _state : ",type(new_state.tolist()))
    #         # #print(self.update_count ," iter : ",(self.last_state[0], new_state.tolist()[0], self.last_action, self.last_reward,0))
    #         # #self.replay_buffer.add((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]),0))
    #         #
    #         # self.replay_buffer.add((self.last_state[0], new_state.tolist()[0], self.last_action, self.last_reward,done))#((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]),float(0)))
    #         # action = self.select_action(new_state)
    #         # action = (action + np.random.normal(0, self.expl_noise, size=1)).clip([-self.max_action],[self.max_action])
    #         # if len(self.replay_buffer.storage) > self.start_timesteps : #batch_size:
    #         #     #for i in range(100):
    #         #     batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(self.batch_size)
    #         #     self.learn(batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones)
    #         # self.last_action = action
    #         # self.last_state = new_state
    #         # self.last_reward = reward
    #         # self.reward_window.append(reward)
    #         # if len(self.reward_window) > 1000:
    #         #     del self.reward_window[0]
    #         # print("action : ", action.tolist(), "reward : " ,sum(self.reward_window))
    #         # #return action.tolist()[0]#
    #         # #print("max_actions ; ", torch.max(torch.Tensor(action),0)[0].tolist())
    #         # #return torch.max(torch.Tensor(action),0)[0].tolist(),torch.max(torch.Tensor(action),0)[1].tolist()
    #         #
    #         # return action.tolist()[0]
    #
    #
    #     #def learn(self, batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones):
    #     def learn(self, iterations):
    #
    #         for i in range(iterations):
    #
    #             # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
    #             # batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
    #             #while self.episode_timesteps
    #             batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(self.batch_size)
    #             state = torch.Tensor(batch_states).to(self.device)
    #             next_state = torch.Tensor(batch_next_states).to(self.device)
    #             action = torch.Tensor(batch_actions).to(self.device)
    #             reward = torch.Tensor(batch_rewards).to(self.device)
    #             done = torch.Tensor(batch_dones).to(self.device)
    #
    #             # Step 5: From the next state s’, the Actor target plays the next action a’
    #             next_action = self.actor_target(next_state)
    #
    #             # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
    #             noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
    #             noise = noise.clamp(-self.noise_clip, self.noise_clip)
    #             next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
    #
    #             # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
    #             target_Q1, target_Q2 = self.critic_target(next_state, next_action)
    #
    #             # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
    #             target_Q = torch.min(target_Q1, target_Q2)
    #             #print(target_Q, reward)
    #             # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
    #             target_Q = reward + ((1 - done) * self.discount * target_Q).detach()
    #
    #             # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
    #             current_Q1, current_Q2 = self.critic(state, action)
    #
    #             # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
    #             critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    #
    #             # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
    #             self.critic_optimizer.zero_grad()
    #             critic_loss.backward()
    #             self.critic_optimizer.step()
    #
    #             # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
    #             if self.update_count % self.policy_freq == 0:
    #                 actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
    #                 self.actor_optimizer.zero_grad()
    #                 actor_loss.backward()
    #                 self.actor_optimizer.step()
    #
    #                 # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
    #                 for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
    #                     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #
    #                 # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
    #                 for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
    #                     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #
    #
    #     def score(self):
    #         return sum(self.reward_window)/(len(self.reward_window)+1.)
    #
    #     def get_last_n_rewards(self, n):
    #         return self.reward_window[-n:]
    #     # Making a save method to save a trained model
    #     def save(self, filename, directory):
    #         torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    #         torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    #
    #     # Making a load method to load a pre-trained model
    #     def load(self, filename, directory):
    #         self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    #         self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

















    #
    #
    #
    #
    # if self.total_timesteps < self.max_timesteps:
    #             # If the episode is done
    #             if self.done:
    #                 # If we are not at the very beginning, we start the training process of the model
    #                 if self.total_timesteps != 0:
    #                     print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps,self.episode_num, self.episode_reward))
    #                 if self.total_timesteps > start_timesteps:
    #                     brain.train(replay_buffer, self.episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
    #                 #reset set state dimenssion elements once episode is done
    #                 self.car.center = self.center
    #                 xx = goal_x - self.car.x
    #                 yy = goal_y - self.car.y
    #                 orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
    #
    #                 # When the training step is done, we reset the state of the environment
    #                 self.state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
    #
    #
    #                 print(orientation)
    #                 # Set the Done to False
    #                 self.done = False
    #
    #                 # Set rewards and episode timesteps to zero
    #                 self.episode_reward = 0
    #                 self.episode_timesteps = 0
    #                 self.episode_num += 1
    #
    #             # Before 10000 timesteps, we play random actions based on uniform distn
    #             #if self.total_timesteps < start_timesteps:
    #             #   action = [np.random.uniform(-45,45)]
    #             #else:
    #             ##debug:
    #             #if self.total_timesteps == 10500:
    #             #   print("check")
    #             #else: # After 10000 timesteps, we switch to the model
    #             action = brain.select_action(self.state)
    #             # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    #             print("earlier action:", action)
    #             if expl_noise != 0:
    #                 action = (action + np.random.normal(0, 0.1)).clip(-max_action, max_action)
    #
    #             print("noise action:", action)
    #             # The agent performs the action in the environment, then reaches the next state and receives the reward
    #             new_state = self.car.move(action[0])
    #
    #
    #             #set new_state dimenssion elements
    #             xx = goal_x - self.car.x
    #             yy = goal_y - self.car.y
    #             orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
    #             #new_state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
    #             distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
    #             self.ball1.pos = self.car.sensor1
    #             self.ball2.pos = self.car.sensor2
    #             self.ball3.pos = self.car.sensor3
    #
    #             # evaluating reward and done
    #             if sand[int(self.car.x),int(self.car.y)] > 0:
    #                 self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
    #                 print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
    #                 print("sand: ", 1,"distance: ", distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
    #                 reward = -1
    #                 self.done = False
    #
    #             else: # otherwise
    #                 self.car.velocity = Vector(2, 0).rotate(self.car.angle)
    #                 reward = -0.2
    #                 print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
    #                 print("sand: ", 0,"distance: ", distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
    #                 if distance < last_distance:
    #                     reward = 0.1
    #
    #                 # else:
    #                 #     last_reward = last_reward +(-0.2)
    # if (self.car.x < 5) or (self.car.x > self.width - 5) or (self.car.y < 5) or (self.car.y > self.height - 5):
    #                 reward = -1
    #                 self.done = True
    #
    #             if distance < 25:
    #                 if swap == 1:
    #                     goal_x = 1420
    #                     goal_y = 622
    #                     swap = 0
    #                     #self.done = False
    #                 else:
    #                     goal_x = 9
    #                     goal_y = 85
    #                     swap = 1
    #                     #self.done = True
    #             last_distance = distance
    #
    #
    #             # We check if the episode is done
    #             if self.episode_timesteps == 1000 and self.total_timesteps<start_timesteps:
    #                 self.done = True
    #
    #             if self.episode_timesteps == 5000 and self.total_timesteps>start_timesteps:
    #                 self.done = True
    #
    #
    #
    #             # We increase the total reward
    #             self.episode_reward += reward
    #
    #             # We store the new transition into the Experience Replay memory (ReplayBuffer)
    #             replay_buffer.add((self.state, new_state, action, reward, self.done))
    #             print(self.state, new_state, action, reward, self.done)
    #             self.state = new_state
    #             # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
    #             #new_state =
    #             self.episode_timesteps += 1
    #             self.total_timesteps += 1