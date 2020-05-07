# Self Driving on a Kivy Environment using TD3



I approached this problem, with the idea to try different things mainly to learn how we can model environment  episodes and rewards to teach a car to drive.

I started with taking the sensor concepts used with the DQN use case and converted it to use TD3 instead of DQN. This exercise helped me understand how will I code the variables, done and rewards to drive the car using TD3. Then I ported the concept to a CNN based TD3 by removing sensors and passing image of the car and its surroundings to the model.

**Folders and Files in the projects:**

1. Code Archives: Copy of codes which were subsequently changed and replaced by newer codes like the code relating to sensors, my approaches were i was initially drawing car on the image

2. Images: has all the images used in the project

3. logs: It has the logs for all the runs which were recorded, to check my progress. The final file for the submission is - 

4. tain_final.py is the final file which was used for training and recording the video for the progress

5. train_CropImgRotatn: Uses the approach of crop and rotation

6. train_smoothPenalty: Uses heavy living penalty, introduces smooth action penalty

7. train_FrameSkip: uses logic of frameskip to continue with a predicted action for those number of frames, and also to push transitions in replay buffer

   My commit history can be checked to get better understanding of all the changes I did in my approach.

### Current Progress:

I have experimented multiple strategies but have not been able successfully solve this problem yet. My car currently goes from start point to both the goals without staying on the road.

[]: https://youtu.be/5cIad8gBSAM	"Link to the video"

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
   
    <img src="https://i.imgur.com/1yeRFjG.png" width="400">

3. **Combining 1 and 2**: This is the final approach I used, I am drawing the car as well on the cropped and rotated road

#### Pre-training filling up Replay buffer with  transitions

1. **Random Actor weights**: filling up using the random weights from the actor target.

   Problem: similar actions were getting spitted out most the time

2. **Random uniform actions** between max action and -max action

   It gave more variety to the buffer

#### Network architecture

1. Deciding on cnn architecture: started with very heavy architecture for baseline, slowly streamlined the architecture. Finally went ahead with a simple architecture which gave me 99% accuracy on MNIST under 20 epochs
2. (8, 3x3, Relu, BN), (12, 3x3, Relu, BN), (16, 3x3, stride=2, Relu, BN), (8, 1x1, Relu, BN), (12, 3x3, Relu, BN), (16, 3x3, Relu, BN), (AdaptiveAveragePool),  Flatten, Dense layer of latent dim 16 to 8 to action_dim* - refer code archives and push history for details of the architectures tried

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

3. Based on the discussion in the forum, I decided to monitor sand penalty vs living penalty to check if my sand penalty is overshadowing the living penalty. With some ideas checking the logs, i figured out that my sand penalty is high compared to my living penalty. I decided to increase living penalty based on the  counter of number of rotation the car was taking. I plotted the penalties after running for 100,000 steps. 

![plot](https://i.imgur.com/ngDEj0I.png)



**Final Reward strategy:**

for each step, reward = rs + rd + rb + ra

**rs** = -15 reward, the agent was punished by 10 points after every 5 continuous steps on the sand

**rd** = +20 reward when last covered distance from the goal< the current covered distance in the step; -10 for living penalty, +40 when only 100 distance units left to the goal, +50 when covered first goal, +70 when covered final goal

**rb** = -30 when the agent slammed on the boundaries

**ra** = -0.1*(action^2), agent penalised for steep steering angles, to come out of rotation frenzy and smooth driving



<u>**Episode Done Strategy**:</u>

Experimented with very aggressive episode completions with very specific episode completions on achievement of the goal by the agent.

1. Sand counter episode completions - finished episodes very quickly, episode completions confused agent about whether episode completed 
2. Max episode steps completions - difficult hyper parameter to set, as with more value, the training iterations increase a lot and as noticed with many timesteps nothing was learnt
3. I tried with having episode completion only when the agent closes to the boundary or the goal. But these led to car going in rotation frenzy for eternity in the initial trained steps.

**Final Strategy**:

My final submission is based on the following episode done strategy:

1. ​      if episode steps == 1500 and total steps<10,000
2. ​      if episode steps == 2000 and total steps<30,000
3. ​      if episode steps == 5000 and total steps>30,000
4. ​      Hitting the boundaries
5. ​      Reached both the goals

Episode completion had a punishment attached to it, to penalize where goal was not reached but episode finished and reward if reached successfully

<u>**Exploration Noise Strategy**</u>

To enable more exploration for the model, I decided to decay exploration noise linearly every 2000 steps. This exploration noise value was used as standard deviation to generate a random value from the normal distribution.



<u>**Other Hyperparameters**</u>

1. Max action: I started with a very high max action of 90 to suggest 90 degrees, I tested it for lower values. I finally settled on 30 as the value, to reduce the steering angles.

2. Episode timesteps
3. sand counter: deciding how much we can travel on sand
4. living penalty counter: deciding how much a car can live without increasing in distance
5. Initializing starting position of the car. I tried random start as well for better transitions
6. Frame Skip: I have implemented frame skip as well, you can check that in the code train_frameSkip.py. I need to check how the results are from this. Based on this paper: https://arxiv.org/pdf/1904.09503.pdf. It works for atari games and this use case of self driving car.  We use the frame skip trick during training, where we keep the action unchanged for k consecutive frames. 

These were all implemented for experimentation and learning how we can better teach the car to learn. Some of these were made inactive for trying different scenarios or concepts at different times.



## Learnings:

I work as a Data Scientist but my job usually deals with structured data. This exercise taught me multiple aspects of software engineering for a complex problem like  self driving a car on a specified road using a novel RL algorithms which has not been much tested on similar problem or documented as algorithms like DQN.

Great discussions with peer and understanding how they are attacking the problem

Brainstorming on the rewards, punishment part was really awesome.

I would have been happier if it started to move on the road but my understanding is I am missing something very trivial still and of course there is no excuse:)













​     



​      
