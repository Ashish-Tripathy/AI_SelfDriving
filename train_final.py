# Self Driving Car
#v2 - 
# 1. updating the max_env_steps to 5000 after 10000 max_stepss
# 2. updating max_action to 5 - car doesnot move a lot mostly straight lines
# 2.1 updating max_action to 45 - car doesnot move a lot mostly straight lines
# 3. updating max episode time
# 4. other hyperparams: 1. velocity and angle updates when car in road or sand; 2. proper randomisation of actions for building buffer
# 5. hyperparams for punishing and rewarding the agent - 
#       done = True if self.av_r(reward) <= -0.1 else False

# may01: change punishments to sand only based on counters
# add steering angle punishments: 0.5 * angle**2
# correct swap logic and episode completions
# stacking rewards basedon categories ra - reward on angle, rs - reward on sand, rd - distance, rb - boundary



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
import time
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch.nn.functional as F
import math
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from PIL import ImageDraw
import cv2
from scipy import ndimage
from PIL import Image
import scipy

import logging 
import sys


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
from kivy.graphics.texture import Texture

# Importing the TD3 object from our AI in TD3_cnn.py
from TD3_cnn import TD3, ReplayBuffer



# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

#Inititalising global variables
max_action = 30                                 #15 #reduced to prevent steep turns
crop_dim = 80                                   #dimension to crop the image from the entire sand
state_dim = 5                                   #state dimension is 80x80 image having car at the center with one channel for grayscale
action_dim = 1                                  #action is just the rotation
latent_dim = 16                                 #latent dimension for dense networks
t0 = time.time()
state = torch.zeros([1,state_dim,state_dim])    #shape of the cropped image
episode_reward = 0                              #reward for entire episode - adds every reward on each step
episode_timesteps = 0                           #number of steps in an episode
total_timesteps = 0                             #total number of steps taken by the agent at any point        
episode_num = 0                                 #number of episodes
done = True                                     #episode done flag
sand_counter = 0                                #counts number of steps on the sand
p_sand = 0                                      #adds punishment by sand per episode
p_living = 0                                    #adds living punishments or penalties per episode
expl_noise_vals = np.linspace(1, int(max_action/1000), num=int(max_timesteps/2000), endpoint=True, retstep=False, dtype=None, axis=0) ##decay expl noise every 2000 timestep; Exploration noise - STD value of exploration Gaussian noise
expl_noise = 0.4                                # exploration noise initialisation
ra,rb,rs,rd = 0,0,0,0                           #Rewards: ra = steering angle, rb = boundary, rs = sand, rd = distance based
im = CoreImage("./images/MASK1.png")
mask = cv2.imread('./images/mask.png',0)
last_distance = 0                               #distance covered in the last step


#initialising variables for training:
brain = TD3(state_dim,action_dim,max_action,latent_dim) #initialising the AI
replay_buffer = ReplayBuffer()                          #Initialising the replay buffer
seed = 0                                                # Random seed number
eval_freq = 5e3                                         # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 500000                                  # Total number of iterations/timesteps
save_models = True                                      # Boolean checker whether or not to save the pre-trained model
start_timesteps = 10000                                 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
batch_size = 30                                         # Size of the batch
discount = 0.99                                         # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005                                             # Target network update rate
policy_noise = 0.2                                      # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5                                        # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2                                         # Number of iterations to wait before the policy network (Actor model) is updated


#function to extract car image
def extract_car(x, y, width, height, angle):
        car_ = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
        theta = (np.pi / 180.0) * angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        car_offset = np.array([x, y])
        cropped_car = np.dot(car_, R) + car_offset
        return cropped_car
#function to extract image and rotate
def get_target_image(img, angle, center, size, fill_with = 255):
    angle = angle + 90
    center[0] -= 0
    img = np.pad(img, size, 'constant', constant_values = fill_with)
    a_0 = center[0]
    a_1 = center[1]
    img_tmp = PILImage.fromarray(img)#.astype("uint8")*255)        
    draw = ImageDraw.Draw(img_tmp)
    extract_car_area = extract_car(x=int(a_1+80), y=int(a_0+80), width=10, height=20, angle = angle-90)#+180)
    draw.polygon([tuple(p) for p in extract_car_area], fill=128)
    #debug
    #plt.imshow(img_tmp)
    #plt.show()
    init_size = 1.6*size
    center[0] += size
    center[1] += size
    img = np.asarray(img_tmp)
    #print(int(center[0]-(init_size/2)) , int(center[1]-(init_size/2)),int(center[0]+(init_size/2)) , int(center[1]+(init_size/2)))
    cropped = img[int(center[0]-(init_size/2)) : int(center[0]+(init_size/2)) ,int(center[1]-(init_size/2)): int(center[1]+(init_size/2))]
    rotated = ndimage.rotate(cropped, angle, reshape = False, cval = 255.0)
    y,x = rotated.shape
    final = rotated[int(y/2-(size/2)):int(y/2+(size/2)),int(x/2-(size/2)):int(x/2+(size/2))]
    final = torch.from_numpy(np.array(final)).float().div(255)
    final = final.unsqueeze(0).unsqueeze(0)
    final = F.interpolate(final,size=(28,28))
    #print(rotated.shape)
    return final.squeeze(0)


# Initializing the environment
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    #sand = np.pad(sand, 160, 'constant', constant_values = 1)
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0




# Creating the car class
class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
 

    def move(self, rotation):
        #signals have been removed from any computaion for TD3, but are still visible
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        

# Creating the game class

class Game(Widget):
    car = ObjectProperty(None)

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
        global orientation
        global last_action
        global last_distance_travelled
        global start_timesteps
        global batch_size
        global discount
        global tau
        global policy_noise
        global noise_clip
        global policy_freq
        global expl_noise
        global reward_window
        global total_timesteps
        global episode_num
        global done
        global t0
        global max_timesteps
        global state
        global episode_reward
        global episode_timesteps
        global sand_counter
        global p_sand
        global p_living
        global lp_counter
        #decay expl noise every 4000 timestep
        global expl_noise_vals
        global crop_dim
        global ra
        global rb
        global rs
        global rd
        #global ck1x, ck2y = 1000, 400
        #global ck2x, ck2y = 
        # global ck3x, ck3y = 
        # global ck4x, ck4y = 


 
        # total_timesteps = 0
        # episode_num = 0
        # done = True
        # t0 = time.time()

        longueur = self.width
        largeur = self.height
        #state = np.zeros(5)
        sand_time = []

        if first_update:
            init()

        # max_timesteps = 10
        # We start the main loop over 500,000 timesteps
        if total_timesteps < max_timesteps:
            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    #print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps,self.episode_num, self.episode_reward))
                    distance_travelled = np.sqrt((self.car.x - 715)**2 + (self.car.y - 360)**2)
                    distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
                    s_reward = float(p_sand)
                    l_reward = float(p_living)
                    #c = np.amax(sand_time)
                    #logger.info("Steps: %d , Reward: %d , Ep: %d , Ep steps: %d , Distance: %d , Distance left: %d ", self.total_timesteps,self.episode_num,self.episode_reward, self.episode_timesteps,distance_travelled,distance)
                    with open("./logs/log_06may.txt", 'a') as f:
                       sys.stdout = f
                       print("Steps: ", total_timesteps, "Episode: ",episode_num, "Reward: ", episode_reward,"Ep Steps: ", episode_timesteps,"Distance covered: ", round(float(distance_travelled),2), "Distance left: ", round(float(distance),2), "punish by sand: ", s_reward, "punish living: ", l_reward)            
                    #print("Steps: ", total_timesteps, "Episode: ",episode_num, "Reward: ", episode_reward,"Ep Steps: ", episode_timesteps,"Distance covered: ", round(float(distance_travelled),2), "Distance left: ", round(float(distance),2), "punish by sand: ", s_reward, "punish living: ", l_reward)          
                if total_timesteps > start_timesteps:
                    #print("I am training for steps: ", self.episode_timesteps)
                    #start_time = time.time()
                    brain.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                    #print("debug")
                    #print("time in minutes: ", round((time.time() - start_time)/60))
                #reset set state dimenssion elements once episode is done
                
                #update car position and state
                self.car.x = 850 #750 #+ np.random.normal(20,40) #for random location update
                self.car.y = 400 #360 #+ np.random.normal(20,40)
                self.car.angle = 0
                self.car.velocity = Vector(6, 0)
                goal_x = 1420
                goal_y = 622
                swap = 0
                xx = goal_x - self.car.x
                yy = goal_y - self.car.y
                orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
                orientation = [orientation, -orientation]         
                state = get_target_image(mask, self.car.angle, [self.car.x, self.car.y], crop_dim)
                # Set the Done to False
                done = False
                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                sand_counter = 0
                lp_Counter = 0
                p_living = 0
                p_sand = 0
            
            # Before 10000 timesteps, we play random actions based on uniform distn
            if total_timesteps < start_timesteps:
                action = [random.uniform(-max_action, max_action)]
                
            else:
                action = brain.select_action(state, np.array(orientation))
                #exploraion noise decay, car getting stuck in the same actions needs aggressive exploration, decay atfer it has learnt something
                expl_noise = expl_noise_vals[int(total_timesteps/2000)]
                action = (action + np.random.normal(0, expl_noise)).clip(-max_action, max_action)


            # agent takes the step based on the action calculated
            if type(action) != type([]):
                self.car.move(action.tolist()[0])
                ra =  -0.1 * (action.tolist()[0] ** 2) #steering angle reward
                p_living += ra
            else:
                self.car.move(action[0])
                ra = -0.1 * (action[0] ** 2)
                p_living += -ra
            
            #calculate distance left from the goal with the step taken
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            
            
            # evaluating reward and done
            if sand[int(self.car.x),int(self.car.y)] > 0:# and self.total_timesteps < start_timesteps:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                sand_counter +=1
                if sand_counter>5:
                    rs = -15
                    p_sand += 15

            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                sand_counter = 0
                rd = -10#living penalty
                p_living += 10

                if distance < last_distance:
                    rd = 20
                    p_living -= 20
        

            # Ending the episodes on the boundaries
            if (self.car.x < 5) or (self.car.x > self.width - 5) or (self.car.y < 5) or (self.car.y > self.height - 5): #crude way to handle model failing near boundaries
                done = True
                rb = -30
            
            #Ending episodes when it clocks max steps
            if total_timesteps<=start_timesteps and episode_timesteps == 1500:
                rb = -15
                done = True
            elif total_timesteps>start_timesteps and total_timesteps<=start_timesteps*3 and episode_timesteps==2000:# and episode_reward > -200:
                rb = -15
                done = True
            elif total_timesteps>start_timesteps*3 and episode_timesteps == 5000:
                rb = -15
                done = True

            
            # rewarding destinationa covered
            if distance < 100:
                rd = 40    
            
            if distance < 10:
                #reward = 1
                
                if swap == 1:
                    goal_x = 1420
                    goal_y = 622
                    swap = 0
                    with open("./logs/log_06may", 'a') as f:
                       sys.stdout = f
                       print("Final Destination ----- Yipeeeee")
                    rd = 50            
                    done = True
                else:
                    goal_x = 9
                    goal_y = 85
                    swap = 1
                    with open("./logs/log_06may.txt", 'a') as f:
                       sys.stdout = f
                       print("1st Destination ----- Yipeeeeeee")
                    rb = 70

            
            #set new_state dimension elements
            last_distance = distance
            new_state = get_target_image(mask, self.car.angle, [self.car.x, self.car.y], crop_dim)
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            new_orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
            new_orientation = [new_orientation, -new_orientation]
            distance_travelled = np.sqrt((self.car.x - 715)**2 + (self.car.y - 360)**2)

            #total reward
            reward = ra + rb + rs + rd
            episode_reward += reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((state, new_state, orientation, new_orientation, action, reward, done))
            state = new_state
            orientation = new_orientation
            episode_timesteps += 1
            total_timesteps += 1
            last_action = action
            last_distance_travelled = distance_travelled
            



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
        #Clock.max_iteration = 5
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
        brain.save("",)
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
    #f.close()

