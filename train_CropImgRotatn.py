# Self Driving Car
#v2 - 
# 1. updating the max_env_steps to 5000 after 10000 max_stepss
# 2. updating max_action to 5 - car doesnot move a lot mostly straight lines
# 2.1 updating max_action to 45 - car doesnot move a lot mostly straight lines
# 3. updating max
# 4. other hyperparams: 1. velocity and angle updates when car in road or sand; 2. proper randomisation of actions for building buffer
# 5. hyperparams for punishing and rewarding the agent - 
#       done = True if self.av_r(reward) <= -0.1 else False



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
#from torchvision import transforms



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
from PIL import ImageDraw
# Importing the Dqn object from our AI in ai.py
#rom ai import Dqn
from TD3_cnn import TD3, ReplayBuffer
import cv2
from scipy import ndimage
from PIL import Image
import scipy

import logging 
import sys
#f = open("./run1704.txt", 'w')
#sys.stdout = f

#Create and configure logger 
logging.basicConfig(filename='./run1704.txt', filemode='a',level=logging.DEBUG, format='%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s')
logger=logging.getLogger() 
#logger.setLevel(logging.DEBUG) 


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
max_action = 3
orientation = 0

#function to extract car image
def extract_car(x, y, width, height, angle):
        car_ = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
        theta = (np.pi / 180.0) * angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        car_offset = np.array([x, y])
        cropped_car = np.dot(car_, R) + car_offset
        return cropped_car

def get_target_image(img, angle, center, size, fill_with = 255):
    angle = angle + 90
    center[0] -= 0
    img = np.pad(img, size, 'constant', constant_values = fill_with)
    init_size = 1.6*size
    #print(img.shape)
    center[0] += size
    center[1] += size
    #print(int(center[0]-(init_size/2)) , int(center[1]-(init_size/2)),int(center[0]+(init_size/2)) , int(center[1]+(init_size/2)))
    cropped = img[int(center[0]-(init_size/2)) : int(center[0]+(init_size/2)) ,int(center[1]-(init_size/2)): int(center[1]+(init_size/2))]
    #return cropped
    rotated = ndimage.rotate(cropped, angle, reshape = False, cval = 255.0)
    y,x = rotated.shape
    final = rotated[int(y/2-(size/2)):int(y/2+(size/2)),int(x/2-(size/2)):int(x/2+(size/2))]
    final = torch.from_numpy(np.array(final)).float().div(255)
    final = final.unsqueeze(0)
    #print(rotated.shape)
    return final#cropped, rotated, 


# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
#brain = Dqn(5,3,0.9)
state_dim = 60
action_dim = 1
latent_dim = 32
brain = TD3(state_dim,action_dim,max_action,latent_dim)
replay_buffer = ReplayBuffer()
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
mask = cv2.imread('./images/mask.png',0)


# Initializing the map
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
    camera_x = NumericProperty(0)
    camera_y = NumericProperty(0)
    camera_z = NumericProperty(0)
    camera = ReferenceListProperty(camera_z, camera_x, camera_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = float(rotation)
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate(self.angle+30) + self.pos
        self.sensor3 = Vector(30, 0).rotate(self.angle-30) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        # #self.pos = Vector(20,18)+self.center
        # self.dummycar = Vector(0, 0).rotate(self.angle) + self.pos
        # a = self.dummycar
        # #print(a)               
        # img_tmp = PILImage.fromarray(sand.astype("uint8")*255)        
        # draw = ImageDraw.Draw(img_tmp)
        # extract_car_area = extract_car(x=int(a[1]+160), y=int(a[0]+160), width=10, height=20, angle = self.angle)
        # draw.polygon([tuple(p) for p in extract_car_area], fill=0)

        # sand1 = np.asarray(img_tmp)
        # cropped_img = sand1[int(a[0])-80:int(a[0])+80, int(a[1])-80:int(a[1])+80] #80x80 images
        # camera_data = np.asarray(cropped_img)  
        # #plt.imshow(sand1)
        # #plt.show()
        # #print(sand1.shape)      
        # camera_data = np.expand_dims(camera_data, axis=0) #add channel data
        # camera_data = torch.from_numpy(camera_data).float().div(255) # normalise the image , FloatTensor type
        # camera_data = camera_data.unsqueeze(0)
        # #print(camera_data.size())
        # camera_data = F.interpolate(camera_data,size=(80,80))
        # #print(camera_data.size())
        # #tens = camera_data.view(camera_data.shape[2], camera_data.shape[3])
        # #plt.imshow(tens)
        # #plt.show()
        # #print(camera_data.size())
        # return camera_data.squeeze(0)

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
    max_timesteps = 100000
    state = torch.zeros([1,state_dim,state_dim]) #shape of the cropped car image
    episode_reward = 0
    episode_timesteps = 0
    #orientation = 0
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
        global orientation
        global last_action
        
        #initialising variables for training:
        #seed = 0 # Random seed number
        #eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
        #max_timesteps = 5e5 # Total number of iterations/timesteps
        #save_models = True # Boolean checker whether or not to save the pre-trained model
        expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
        start_timesteps = 10000 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
        batch_size = 30 # Size of the batch
        discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
        tau = 0.005 # Target network update rate
        policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
        noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
        policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
        
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
        if self.total_timesteps < self.max_timesteps:
            # If the episode is done
            if self.done:
                # If we are not at the very beginning, we start the training process of the model
                if self.total_timesteps != 0:
                    #print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps,self.episode_num, self.episode_reward))
                    distance_travelled = np.sqrt((self.car.x - 715)**2 + (self.car.y - 360)**2)
                    distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
                    #c = np.amax(sand_time)
                    #logger.info("Steps: %d , Reward: %d , Ep: %d , Ep steps: %d , Distance: %d , Distance left: %d ", self.total_timesteps,self.episode_num,self.episode_reward, self.episode_timesteps,distance_travelled,distance)
                    with open("log1704.txt", 'a') as f:
                        sys.stdout = f
                        print("Steps: ", self.total_timesteps, "Episode: ",self.episode_num, "Reward: ", self.episode_reward,"Ep Steps: ", self.episode_timesteps,"Distance covered: ", distance_travelled, "Distance left: ", distance)      
            
                if self.total_timesteps > start_timesteps:
                    #print("I am training for steps: ", self.episode_timesteps)
                    start_time = time.time()
                    brain.train(replay_buffer, self.episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                    #print("time in minutes: ", round((time.time() - start_time)/60))
                #reset set state dimenssion elements once episode is done
                
                #update car position
                self.car.x = 715 #+ np.random.normal(20,40)
                self.car.y = 360 #+ np.random.normal(20,40)
                self.car.velocity = Vector(6, 0)
                xx = goal_x - self.car.x
                yy = goal_y - self.car.y
                orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
                orientation = [orientation, -orientation]

                #initialise 1st state after done, move it towards orientaation
                self.car.angle = 0
                self.state = get_target_image(mask, self.car.angle, [self.car.x, self.car.y], 60)
                #print(self.state.size())
                #print(self.state)
                #tens = self.state.view(self.state.shape[1], self.state.shape[2])
                #plt.imshow(tens)
                #plt.show()
                #or self.state = self.car.move(0)

                #print("from update: ",self.state)
                #print(self.state.size())
                #print(orientation)
                # Set the Done to False
                self.done = False
                last_action = [0]
                # Set rewards and episode timesteps to zero
                self.episode_reward = 0
                self.episode_timesteps = 0
                self.episode_num += 1
                self.sand_counter = 0

            # Before 10000 timesteps, we play random actions based on uniform distn
            if self.total_timesteps < start_timesteps:
                action = [np.random.uniform(-max_action, max_action)]
                
            #else:
            ##debug:
            #if self.total_timesteps == 10500:
            #   print("check")
            else: # After 10000 timesteps, we switch to the model
                action = brain.select_action(self.state, np.array(orientation))

            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            #print("earlier action:", action)
                if expl_noise != 0:
                    #exploraion noise decay, car getting stuck in the same actions needs aggressive exploration, decay atfer it has learnt something
                    #expl_noise
                    action = (action + np.random.normal(0, 1)).clip(-max_action, max_action)
            #if round(abs(float(action[0]) - float(last_action[0])))> 10:
             #   action[0] = (action[0] + last_action[0]) / 2
            #print("noise action:", action)
            #The agent performs the action in the environment, then reaches the next state and receives the reward
            self.car.move(action[0])
            new_state = get_target_image(mask, self.car.angle, [self.car.x, self.car.y], 60)
            #tens = new_state.view(self.state.shape[1], self.state.shape[2])
            #plt.imshow(tens)
            #plt.show()
            
            #set new_state dimenssion elements
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            new_orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
            new_orientation = [new_orientation, -new_orientation]
            #new_state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            self.ball1.pos = self.car.sensor1
            self.ball2.pos = self.car.sensor2
            self.ball3.pos = self.car.sensor3
            
            sand_time = []
            # evaluating reward and done
            
            if sand[int(self.car.x),int(self.car.y)] > 0:# and self.total_timesteps < start_timesteps:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                self.sand_counter +=1
                reward = -1
                if self.total_timesteps < start_timesteps:
                    self.done = False
                else:
                    self.done = True
            
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                self.sand_counter = 0
                reward = -0.2
                #print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
                #print("sand: ", 0,"distance: ", distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if distance < last_distance:
                    reward = 0.1 #0.1



                # else:
                #     last_reward = last_reward +(-0.2)

            if (self.car.x < 5) or (self.car.x > self.width - 5) or (self.car.y < 5) or (self.car.y > self.height - 5): #crude way to handle model failing near boundaries
                self.done = True
                reward = -0.5
            
            if distance < 200:
                reward = 1 #0.2
                if swap == 1:
                    goal_x = 1420
                    goal_y = 622
                    swap = 0
                    #self.done = False
                else:
                    goal_x = 9
                    goal_y = 85
                    swap = 1
                    #self.done = True
            last_distance = distance
            
            # We increase the total reward
            self.episode_reward += reward
            
            

            # We check if the episode is done
            #if self.episode_timesteps == 1000: #and self.total_timesteps<start_timesteps:
            #   self.done = True
            
            if self.episode_timesteps == 500 and self.total_timesteps<start_timesteps:
                self.done = True
            if self.episode_timesteps == 2000 and self.total_timesteps>start_timesteps:# and episode_reward > -200:
                self.done = True
            
            
            #additional rewards and punishments:
            
            ##add punishment if sand touched before 10 timesteps
            if self.episode_timesteps < 10 and self.sand_counter == 1:
                reward -= 0.2
            #punish roundabout circles
            if abs(float(action[0]) - float(last_action[0]))/max_action < 0.01:
                reward -= 10
            

            #if round(abs((action - last_action))> 20:
            #   reward -= 0.2



            #end episode if more time on sand
            #if self.sand_counter == 50:
             #   self.done = True

            sand_time.append(self.sand_counter)        


            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((self.state, new_state, orientation, new_orientation, action, reward, self.done))
            #print(self.state, new_state, action, reward, self.done)
            self.state = new_state
            self.orientation = new_orientation
            self.episode_timesteps += 1
            self.total_timesteps += 1
            last_action = action
            



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

