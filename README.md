# Self Driving on a Kivy Environment using TD3



I approached this problem, with the idea to try different things mainly to learn how we can model environment  rewards and punishments to teach a car to drive.

Things implemented and tried:

1. Setting up environment for Reinforcement learning a task

2. Task: To take a car from one step to other

3. **Replay buffer:** 

   1. **Random Actor weights**: filling up using the random weights from the actor target.

      Problem: similar actions were getting spitted out most the time

   2. **Random uniform actions** between max action and -max action

      It gave more variety to the buffer

4. **TD3:**
   1. Implemented the sensor part first. To understand how to use TD3 in place of DQN for the same task
   2. Replaced sensors by CNN. 
   3. In CNN: 
      1. tried multiple architectures from wide, to deep with bottlenecks - (check code archives)
      2. tried two major approaches of deciding the input image:
         1. **Cropping sand and Drawing car**: Cropping a 160x160 sand image, drawing car over it with gray(fill=128),  rescaling the image to 80x80 and passing it to a deep cnn with bottlenecks
            1. Performance was not very good even after 50,000 timesteps of training
            2. It needed deeper architecture to recognize car patterns as well 
            3. Could have worked better with the rewards and punishments i added later
         2. **Cropping sand and rotating**: Cropping a 60x60 sand image, Rotate image based on cars current position by the angle of the car with the x axis of the environment. This made more sense to me, as it was like an actual self driving car where the car can see how is the road currently from its center. This also not needed
      3. settled with a *(5x5, 8, relu)*, followed by 3 * *(3x3, 16, stride = 2, relu)*,  *(3x3, 32, relu)*, *(4x4, 32, relu)*, *Flatten, Dense layer of latent dim 30 to 16 to 8 to action_dim*