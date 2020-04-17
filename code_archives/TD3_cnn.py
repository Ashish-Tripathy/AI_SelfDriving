import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#define replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
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
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
            state, next_state, action, reward, done = self.storage[i]
            #state, next_state, action, reward = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# TD3(1, 1, 10, 1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, latent_dim):
        super(Actor, self).__init__()

        self.encoder = torch.nn.ModuleList([  ## input size:[192, 192]
            torch.nn.Conv2d(1, 16, 3, stride = 2), ##changed to 1 ## output size: [16, 39, 39] , No of filters = 16, kernel = 5, stride = 2, padding =2 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, stride = 2),  ## output size: [32, 19, 19]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, stride=2),  ## output size: [64, 9, 9]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, stride=2),  ## output size: [128, 4, 4]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 3),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 2),  ## output size: [512, 1, 1]
            #torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(512),
            #torch.nn.Conv2d(512, 1024, 5, 2, padding=2),  ## output size: [1024, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, action_dim),
            torch.nn.Tanh(),
        ])

        self.max_action = max_action

    def forward(self, x):

        for layer in self.encoder:
#            print(layer)
#            print("input:", x.size())
            x = layer(x)
#            print("output:", x.size())
        
        for layer in self.linear:
            x = layer(x)
#            print("unclipped action: ", x)
            
        x = self.max_action * x
        return x
		
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(Critic, self).__init__()

        self.encoder_1 = torch.nn.ModuleList([  ## input size:[192, 192]
            torch.nn.Conv2d(1, 16, 3, stride = 2), ##changed to 1 ## output size: [16, 39, 39] , No of filters = 16, kernel = 5, stride = 2, padding =2 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, stride = 2),  ## output size: [32, 19, 19]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, stride=2),  ## output size: [64, 9, 9]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, stride=2),  ## output size: [128, 4, 4]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 3),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 2),  ## output size: [512, 1, 1]
            #torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(512),
            #torch.nn.Conv2d(512, 1024, 5, 2, padding=2),  ## output size: [1024, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear_1 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

        self.encoder_2 = torch.nn.ModuleList([  ## input size:[192, 192]
            torch.nn.Conv2d(1, 16, 3, stride = 2), ##changed to 1 ## output size: [16, 39, 39] , No of filters = 16, kernel = 5, stride = 2, padding =2 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, stride = 2),  ## output size: [32, 19, 19]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, stride=2),  ## output size: [64, 9, 9]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, stride=2),  ## output size: [128, 4, 4]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 3),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 2),  ## output size: [512, 1, 1]
            #torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(512),
            #torch.nn.Conv2d(512, 1024, 5, 2, padding=2),  ## output size: [1024, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear_2 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

    def forward(self, x, u):
        #print("entered critic")
        x1 = x
        for layer in self.encoder_1:
            x1 = layer(x1)
         #   print(x1.size())
        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 1:
                x1 = torch.cat([x1, u], 1)
                x1 = layer(x1)
            else:
                x1 = layer(x1)

        x2 = x
        for layer in self.encoder_2:
            x2 = layer(x2)
        counter = 0
        for layer in self.linear_2:
            counter += 1
            if counter == 1:
                x2 = torch.cat([x2, u], 1)
                x2 = layer(x2)
            else:
                x2 = layer(x2)

        return x1, x2

    def Q1(self, x, u):

        for layer in self.encoder_1:
            x = layer(x)

        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 1:
                x = torch.cat([x, u], 1)
                x = layer(x)
            else:
                x = layer(x)

        return x
        
class TD3(object):

    def __init__(self, state_dim, action_dim, max_action, latent_dim):
        self.actor = Actor(state_dim, action_dim, max_action, latent_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, latent_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim, latent_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, latent_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        
    def select_action(self, state):
        #state = torch.Tensor(state.reshape(1, -1)).to(device)
        state = torch.Tensor(state).unsqueeze(0).to(device) #add batch info
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        for it in range(iterations):
                
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            #print("iteration: ", it)
            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)
            
            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            
            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
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
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    
    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
