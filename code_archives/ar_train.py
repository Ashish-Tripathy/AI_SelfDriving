
  
# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

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
from ar_ait3d import TD3, ReplayBuffer
import random
import cv2
from scipy import ndimage
from PIL import Image
import scipy


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')


# model parameters START
seed = 0 # Random seed number
start_timesteps = 50 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e2 # How often the evaluation step is performed (after how many timesteps)
#max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 30 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
done = True
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_reward = 0
episode_timesteps = 0

action_len = 1
state_len = 5
last_time_steps = 1
image_size = 60
orientation = -0.9
#obs = [0.23,1,1,0.5, -0.5]
# model parameters END

# model global params
replay_buffer = ReplayBuffer()
# model global params


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
max_action_agent = 40
brain = TD3(state_len,action_len,max_action_agent)
action2rotation = [0,5,-5]
reward = 0
scores = []
reward_window = []
im = CoreImage("./images/MASK1.png")
main_img = cv2.imread('./images/mask.png',0)

# textureMask = CoreImage(source="./kivytest/simplemask1.png")
def save_cropped_image(img, x, y, name = ""):
    # print("entered")
    # data = np.array(img)# * 255.0
    # rescaled = data.astype(np.uint8)
    # im = Image.fromarray(rescaled)
    # im.save("./check/"+name+ "_" + "your_file"+str(x) +"_"+ str(y) +".png")
    return

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
    #print(rotated.shape)
    return cropped, rotated, final


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
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.


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

    def serve_car(self):
        self.car.center = (715, 360)
        self.car.angle = 0
        self.car.velocity = Vector(4, 0)
        # self.car.center = self.center
        # self.car.velocity = Vector(6, 0)

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


        global obs


        # NEW GLOBALS
        global replay_buffer
        global seed
        global start_timesteps
        global eval_freq
        #global max_timesteps
        global save_models
        global expl_noise
        global batch_size
        global discount
        global tau
        global policy_noise
        global noise_clip
        global policy_freq
        global done
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global episode_reward
        global reward_window

        global episode_timesteps
        global main_img
        global image_size

        global last_time_steps
        # NEW GLOBALS


        longueur = self.width
        largeur = self.height
        if first_update:
            init()







        #if total_timesteps < max_timesteps:
        if True :

            # If the episode is done
            if done:


                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Timesteps diff: {} Reward: {} score: {}".format(total_timesteps, episode_num, total_timesteps - last_time_steps,episode_reward, episode_reward/(total_timesteps - last_time_steps)))
                    brain.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                    last_time_steps = total_timesteps
                    print("devug")
                # We evaluate the episode and we save the policy
                # WILL COME TO THIS LATER
                # if timesteps_since_eval >= eval_freq:
                #   timesteps_since_eval %= eval_freq
                #   evaluations.append(evaluate_policy(policy))
                #   policy.save(file_name, directory="./pytorch_models")
                #   np.save("./results/%s" % (file_name), evaluations)

                # state calculation
                # self.serve_car()
                # xx = goal_x - self.car.x
                # yy = goal_y - self.car.y
                # orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
                # obs = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
                # state calculation

                #cnn state calculation
                self.serve_car()
                _,_,obs = get_target_image(main_img, self.car.angle, [self.car.x, self.car.y], image_size)
                save_cropped_image(obs, self.car.x, self.car.y, name = "initial")

                xx = goal_x - self.car.x
                yy = goal_y - self.car.y
                orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
                orientation = [orientation, -orientation]

                #cnn state calculation

                # When the training step is done, we reset the state of the environment
                # obs = env.reset()

                # Set the Done to False
                done = False
                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = [random.uniform(-max_action_agent * 1.0, max_action_agent * 1.0)]
                #action = env.action_space.sample()
            else: # After 10000 timesteps, we switch to the model
                action = brain.select_action(np.array(obs), np.array(orientation))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=action_len)).clip(-1*max_action_agent,max_action_agent)

            # The agent performs the action in the environment, then reaches the next state and receives the reward


            # ENV STEP PERFORM START
            if type(action) != type([]):
                #print("action : ",type(action.tolist()[0]), type(action[0]))
                self.car.move(action.tolist()[0])
            else:
                self.car.move(action[0])
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            self.ball1.pos = self.car.sensor1
            self.ball2.pos = self.car.sensor2
            self.ball3.pos = self.car.sensor3

            if sand[int(self.car.x),int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                #print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))

                reward = -1
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                reward = -0.2
                #print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if distance < last_distance:
                    reward = 0.1
                # else:
                #     last_reward = last_reward +(-0.2)



            if self.car.x < 5:
                self.car.x = 5
                reward = -1
            if self.car.x > self.width - 5:
                self.car.x = self.width - 5
                reward = -1
            if self.car.y < 5:
                self.car.y = 5
                reward = -1
            if self.car.y > self.height - 5:
                self.car.y = self.height - 5
                reward = -1

            if distance < 25:
                if swap == 1:
                    goal_x = 1420
                    goal_y = 622
                    swap = 0
                else:
                    goal_x = 9
                    goal_y = 85
                    swap = 1
            last_distance = distance

            # cnn state calculation
            _,_,new_obs = get_target_image(main_img, self.car.angle, [self.car.x, self.car.y], image_size)
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            new_orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
            new_orientation = [new_orientation, -new_orientation]
            save_cropped_image(new_obs, self.car.x, self.car.y, name = "")
            # cnn state calculation

            # state calculation
            # xx = goal_x - self.car.x
            # yy = goal_y - self.car.y
            # orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
            # new_obs = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
            # state calculation

            reward_window.append(reward)

            if sum(reward_window[len(reward_window)-20:]) <= -19 or episode_timesteps % 2500 == 0 and episode_timesteps != 0:
                done = True
                reward_window = []


            # ENV STEP PERFORM END

            # new_obs, reward, done, _ = env.step(action)



            # We check if the episode is done
            #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, orientation, new_obs, new_orientation, action, reward, done))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            orientation = new_orientation
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1












        # xx = goal_x - self.car.x
        # yy = goal_y - self.car.y
        # orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        # last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # action = brain.update(last_reward, last_signal)
        # scores.append(brain.score())
        # rotation = action2rotation[action]
        # self.car.move(rotation)
        # distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        # self.ball1.pos = self.car.sensor1
        # self.ball2.pos = self.car.sensor2
        # self.ball3.pos = self.car.sensor3
        #
        # if sand[int(self.car.x),int(self.car.y)] > 0:
        #     self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
        #     print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
        #
        #     last_reward = -1
        # else: # otherwise
        #     self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        #     last_reward = -0.2
        #     print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
        #     if distance < last_distance:
        #         last_reward = 0.1
        #     # else:
        #     #     last_reward = last_reward +(-0.2)
        #
        # if self.car.x < 5:
        #     self.car.x = 5
        #     last_reward = -1
        # if self.car.x > self.width - 5:
        #     self.car.x = self.width - 5
        #     last_reward = -1
        # if self.car.y < 5:
        #     self.car.y = 5
        #     last_reward = -1
        # if self.car.y > self.height - 5:
        #     self.car.y = self.height - 5
        #     last_reward = -1
        #
        # if distance < 25:
        #     if swap == 1:
        #         goal_x = 1420
        #         goal_y = 622
        #         swap = 0
        #     else:
        #         goal_x = 9
        #         goal_y = 85
        #         swap = 1
        # last_distance = distance

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
