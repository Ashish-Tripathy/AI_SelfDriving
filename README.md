# Self Driving on a Kivy Environment using TD3



I approached this problem, with the idea to try different things mainly to learn how we can model environment  episodes and rewards to teach a car to drive.

I started with taking the sensor concepts used with the DQN use case and converted it to use TD3 instead of DQN. This exercise helped me understand how will I code the variables, done and rewards to drive the car using TD3. Then I ported the concept to a CNN based TD3 by removing sensors and passing image of the car and its surroundings to the model.

**Folders and Files in the projects:**

1. Code Archives: Copy of codes which were subsequently changed and replaced by newer codes like the code relating to sensors, my approaches were i was initially drawing car on the image

2. Images: has all the images used in the project

3. logs: It has the logs for 3 runs which were recorded, to check my progress

4. train_CropImgRotatn: Uses the approach of crop and rotation

5. train_smoothPenalty: Uses heavy living penalty, introduces smooth action penalty

6. train_FrameSkip: uses logic of frameskip to continue with a predicted action for those number of frames, and also to push transitions in replay buffer

   My commit history can be checked to get better understanding of all the changes I did in my approach.



I will only be describing my approach with CNN based TD3 and what I learnt from the exercise here.

### Environment and Scenario:

We used Kivy to simulate a map to drive a car on the map. The map is of 1629 x 660 dimension. The task was to start driving the car from the center of the map to one corner of the map (Goal 1) and then to the diagonally other extreme of the map (Goal 2) all the time staying on the road.

**Observation space:** Grayscale image, Orientation, -Orientation. To start with we experimented only with Grayscale image but later added orientation as well.

**Action dimension:** Steering angle



## Experiments:

#### Approaches

In the process of solving this task, two approaches were adopted to understand which will help in providing the correct image for CNN to learn.

1. **Cropping sand and Drawing car**: Cropping a 160x160 sand image, drawing car over it with gray(fill=128),  rescaling the image to 80x80 and passing it to a deep cnn with bottlenecks

   1. Performance was not very good even after 50,000 timesteps of training
   2. It needed deeper architecture to recognize car patterns as well 
   3. Could have worked better with the rewards and punishments, I added later
   4. Changing the car shape to a triangle for direction information

   

2. **Cropping sand and rotating**: Cropping a 60x60 sand image, Rotate image based on cars current position by the angle of the car with the x axis of the environment. The 60x60 was scaled down to 28x28 image. This made more sense to me, as it was like an actual self driving car where the car can see how is the road currently from its center. This also not needed

   Image: 

   Actual map:

   ![](https://i.imgur.com/li7BMPW.png)

   

    Cropped 60x60 image scaled to 28x28:

<img src="https://i.imgur.com/Uvawazb.png" style="zoom: 50%;" />

#### Pre-training filling up Replay buffer with  transitions

1. **Random Actor weights**: filling up using the random weights from the actor target.

   Problem: similar actions were getting spitted out most the time

2. **Random uniform actions** between max action and -max action

   It gave more variety to the buffer

#### Network architecture

1. Deciding on cnn architecture: started with very heavy architecture for baseline, slowly streamlined the architecture. For the *drawing car* approach used bottlenecks to understand car pattern. 
2. For *rotation* approach kept making the architecture lighter, major reason was to have my cnn learn faster as the patterns were not very complex.
3. Final architecture: (8, 3x3, Relu, BN), (16, 3x3, Relu, BN), (16, 3x3, stride=2, Relu, BN), (8, 3x3, Relu, BN), (16, 3x3, Relu, BN), (GAP),  Flatten, Dense layer of latent dim 16 to 30 to action_dim* - refer code archives and push history for details of the architectures tried

#### Implementation details

To improve performance of the task, we experimented with various strategies. Ranging from rewards to decaying exploration noise, to deciding when an episode will end.

<u>**Rewards Design**</u>:

My major work spent in this exercise was testing variety of reward combinations and strategies. Following are some of the strategies I tested:

1. First my strategy was to award or punish based on only one condition. During this I tried various combinations like: Punish on every step on sand but reward when distance from the goal less than the last distance, reward when goal reached, punish and completing episodes when reached the borders of the map

2. Slowly with more brainstorming, I came up with various reward strategies like the below. The reward was now made a function of both sand and living penalties with multiple combinations.

   1. Punishing when touched sand after some timesteps
   2. Setting episode done to true when some threshold steps covered on sand and heavy penalty to rewards
   3. Incrementally punishing on sand based on number of timesteps in sand

   With heavy penalties based on the actions taken by the agent on the sand, I observed that this was causing the agent to stay on the road but go on a rotation frenzy.

   To combat this we worked on Living penalty or penalizing the agent for being on the road but not moving towards the target.

   4. Reward as a function of distance left from the goal, lesser the distance more the reward

   5. Penalizing consecutive actions with very heavy or similar rotation with some threshold

      I also tested with smoothening actions returned by the model by taking an average of current action and last action. They did help in reducing the rotation angles but the car was still rotating making bigger circles.

   6. Penalizing every step where the distance travelled from the initial step was less than the previous distance

   7. I also added another smoothening penalty for heavy steering angles. This idea was derived from https://arxiv.org/pdf/1904.09503.pdf 

3. Based on the discussion in the forum, I decided to monitor sand penalty vs living penalty to check if my sand penalty is overshadowing the living penalty. With some ideas checking the logs, i figured out that my sand penalty is high compared to my living penalty. I decided to increase living penalty based on the  counter of number of rotation the car was taking. I plotted the penalties after running for 100,000 steps. 

   ![plot](https://i.imgur.com/ngDEj0I.png)

I reduced the sand penalty values and increased living penalty value but they also did not help in improvement of the performance of the agent. 

I added more penalty with the new smoothening penalty for heavy steering angle. Below is the plot describing the penalties for this run:

![](https://i.imgur.com/imfSPJt.png)

My smoothening penalty did add excess living penalty, the results were the car was going more in the sand. I was not able to balance it effectively.

I have to work on this.



<u>**Episode Done Strategy**:</u>

Experimented with very aggressive episode completions with very specific episode completions on achievement of the goal by the agent.

1. Sand counter episode completions - finished episodes very quickly, episode completions confused agent about whether episode completed 
2. Max episode steps completions - difficult hyper parameter to set, as with more value, the training iterations increase a lot and as noticed with many timesteps nothing was learnt
3. I tried with having episode completion only when the agent closes to the boundary or the goal. But these led to car going in rotation frenzy for eternity in the initial trained steps.
4. My final submission is based on the following episode done strategy:
   1. ​      if episode steps == 1000 and total steps<10,000
   2. ​      if episode steps == 1000 and total steps<30,000
   3. ​      if episode steps == 5000 and total steps>10,000
5. Episode completion had a punishment attached to it, to penalize where goal was not reached but episode finished

<u>**Exploration Noise Strategy**</u>

To enable more exploration for the model, I decided to decay exploration noise linearly every 4000 steps. This exploration noise value was used as standard deviation to generate a random value from the normal distribution.



<u>**Other Hyperparameters**</u>

1. Max action: I started with a very high max action of 90 to suggest 90 degrees, I tested it for lower values. I finally settled on a 5 as the value, to reduce the steering angles.

2. Episode timesteps
3. sand counter: deciding how much we can travel on sand
4. living penalty counter: deciding how much a car can live without increasing in distance
5. Initializing starting position of the car. I tried random start as well for better transitions
6. Frame Skip: I have implemented frame skip as well, you can check that in the code train_frameSkip.py. I need to check how the results are from this. Based on this paper: https://arxiv.org/pdf/1904.09503.pdf. It works for atari games and this use case of self driving car.  We use the frame skip trick during training, where we keep the action unchanged for k consecutive frames. 

These were all implemented for experimentation and learning how we can better teach the car to learn. Some of these were made inactive for trying different scenarios or concepts at different times.

### Next steps:

Sometimes my car progressed and covered distance of almost 300-400 steps, but I noticed once it started going into rotations with minimal distance travel, it was very difficult for it to come out of it. Even the heavy living penalties did not help me much.

This will be my strategy going forward:

1.  I have already prepared a very basic road for my car (easymask.png in Images folder), and I will work towards making it move on that road. A strategy discussed in the discussion forum where we were asked to break the problem into a smaller problem and then use it to tackle on the actual map

2. Need to understand how to effectively check why the car starts going in roundabout circle. I checked my cnn architecture, the images that go into it, but many times the values from Tan are very close to 1 or -1. We need to understand what is my CNN learning from the images. This will be the first thing i need to understand.

3. Also From the discussion forum, I have the idea to not backprop the conv layer, use it for creating embeddings for the image and train the Fc layers only for actor-critic.

4. With this I will be able to decide on a strategy for episode completions correctly

5. Decide on a proper reward strategy with more experiments. Run more timesteps and plot penalties to see what is taking more contribution towards rewards

6. Stacking frames and passing to cnn, theoretically should improve my learning from CNN. Need to check that.

   

   





​     



​      