#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque


# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
#rom ai import Dqn
#from td3 import TD3, ReplayBuffer

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')


# ## Replay Buffer
# remove done

# In[2]:


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


# ## TD3 Code


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
    

# Selecting the device (CPU or GPU)
#torch.cuda.set_device(0)
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

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            #batch_states, batch_next_states, batch_actions, batch_rewards = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

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




# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
max_action = 90

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
#brain = Dqn(5,3,0.9)
brain = TD3(5,1,max_action)
action2rotation = [0,5,-5]
replay_buffer = ReplayBuffer()
last_reward = 0
scores = []
#im = CoreImage("./images/MASK1.png")



# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur)) # initializing the sand array with only zeros
    goal_x = 20 # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
    goal_y = largeur - 20 # the goal to reach is at the upper left of the map (y-coordinate)
    first_update = False # trick to initialize the map only once


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = int(rotation)
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate(self.angle+30) + self.pos
        self.sensor3 = Vector(30, 0).rotate(self.angle-30) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.
      


# In[7]:

'''
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

'''

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass


# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    total_timesteps = 0
    episode_num = 0
    done = True
    t0 = time.time()
    max_timesteps = 500000
    #state = torch.ones([4,state_dim,state_dim]) #shape of the cropped car image
    #state = torch.ones([1,state_dim,state_dim]) #shape of the cropped car image
    state = torch.zeros([5])
    episode_reward = 0
    episode_timesteps = 0
    sand_counter = 0

    def serve_car(self):
      self.car.center = self.center
      self.car.velocity = Vector(6, 0)

    def update(self, dt):
      global brain
      global reward
      global scores
      global last_distance
      global goal_x
      global goal_y
      global longueur
      global largeur
      global swap
      #initialising variables for training:
      
      expl_noise = 0.5 # Exploration noise - STD value of exploration Gaussian noise
      start_timesteps = 20000 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
      batch_size = 50 # Size of the batch
      discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
      tau = 0.005 # Target network update rate
      policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
      noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
      policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
      log_interval = 10  # print avg reward after interval
      # logging variables:       

      longueur = self.width
      largeur = self.height
      
      if first_update:
          init()

      # We start the main loop over 500,000 timesteps
      if self.total_timesteps < self.max_timesteps:
          # If the episode is done
          if self.done:
              # If we are not at the very beginning, we start the training process of the model
              if self.total_timesteps != 0:
                  print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
              #if self.total_timesteps > self.start_timesteps:
              #    brain.train(replay_buffer, self.episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

              
              #reset set state dimenssion elements once episode is done
              self.car.center = self.center
              self.car.velocity = Vector(6, 0)
              xx = goal_x - self.car.x
              yy = goal_y - self.car.y
              orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
                      
              # When the training step is done, we reset the state of the environment
              self.state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

              # Set the Done to False
              self.done = False

              # Set rewards and episode timesteps to zero
              self.episode_reward = 0
              self.episode_timesteps = 0
              self.episode_num += 1
        # Before 10000 timesteps, we play random actions
          #if total_timesteps < start_timesteps:
              #action = env.action_space.sample()
           #   action = random.sample(range(-max_action, +max_action), 1)

          #else: # After 10000 timesteps, we switch to the model
          action = brain.select_action(np.array(self.state))
          # If the explore_noise parameter is not 0, we add noise to the action and we clip it
              #if expl_noise != 0:
               #   action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

          
          # The agent performs the action in the environment, then reaches the next state and receives the reward
          new_state = self.car.move(action[0])
          
          
          #set new_state dimenssion elements
          xx = goal_x - self.car.x
          yy = goal_y - self.car.y
          orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
          new_state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
          distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
          self.ball1.pos = self.car.sensor1
          self.ball2.pos = self.car.sensor2
          self.ball3.pos = self.car.sensor3
          
          # evaluating reward and done
          if sand[int(self.car.x),int(self.car.y)] > 0:
              self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
              #print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
              #print("sand: ", 1,"distance: ", distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
              reward = -1
              self.done = False
          
          else: # otherwise
              self.car.velocity = Vector(2, 0).rotate(self.car.angle)
              reward = -0.2
              #print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
              #print("sand: ", 0,"distance: ", distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
              if distance < last_distance:
                  reward = 0.1
              
              # else:
              #     last_reward = last_reward +(-0.2)

          if (self.car.x < 10) or (self.car.x > self.width - 10) or (self.car.y < 10) or (self.car.y > self.height - 10):
              reward = -1
              self.done = True
          
          '''
          if distance < 25:
              if swap == 1:
                  goal_x = 1420
                  goal_y = 622
                  swap = 0
                  done = False
              else:
                  goal_x = 9
                  goal_y = 85
                  swap = 1
                  done = True'''
          last_distance = distance


          # We check if the episode is done
          if self.episode_timesteps == 500:
              done = True
          

          if self.episode_timesteps == 500 and self.total_timesteps<start_timesteps:
               self.done = True
          if self.episode_timesteps == 1000 and self.total_timesteps>start_timesteps:
               self.done = True
            
           #end episode if more time on sand
          if reward == -1:
               self.sand_counter +=1
          else:
               self.sand_counter = 0
          if self.sand_counter == 200:
               reward += -3
               self.done = True           


          # We store the new transition into the Experience Replay memory (ReplayBuffer)
          replay_buffer.add((self.state, new_state, action, reward, self.done))
          #print(self.state, new_state, action, reward, self.done)
          self.state = new_state
          # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
          #new_state = 
          self.episode_timesteps += 1
          self.total_timesteps += 1




# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()

