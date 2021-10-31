# Deep Reinforcement Learning for Autonomous Driving, with Carla simulator

<p align="center">
![Alt Text](https://github.com/ZuoNicolas/DRL-for_Autonomous_Driving/blob/main/presentation_video/first_learn.gif)
<\p>
    
#### Important this project was realized with Python 3.7.9 and with the last version of the libraries used
#### With the Windows 0.9.12 version of Carla, although there should be no problem with other versions and linux versions under Ubuntu, but not yet tested!



This project was done during 1 weekend, and it still needs a lot of changes and improvements. The improvements and changes for the continuation of this project will be :

1. Change the neural network to a convolution type network (CNN). Which is much more suitable when using image data as input.
    

2. A SSH connection to the faculty computers, rather than a local connection, to avoid latencies (on my PC), because my GPU is not powerful enough to support the Simulator and the model learning with CUDA, which uses all the GPU without limitation. This makes the simulator slow down and create a latency.
    

3. Documentation on a way to have a better method to define the reward functions, because until now, it was defined in an arbitrary way. The EPIC method for Equivalent-Policy Invariant Comparison, seems to be a promising way.

    
4. Then an optimization of the hypers-parameters, with Optuna and also with the NAS method, to automate the design of artificial neural networks, so that it is optimal !
    

5. To test also other type of sensor that Carla proposes, because there we use only the front RGB camera of the vehicle.


![Alt Text](https://github.com/ZuoNicolas/DRL-for_Autonomous_Driving/blob/main/presentation_video/second_learn.gif)
