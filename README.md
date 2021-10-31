# Deep Reinforcement Learning for Autonomous Driving, with Carla simulator

### Result obtained with the first environment :
<p align="center">
  <img src="https://github.com/ZuoNicolas/DRL-for_Autonomous_Driving/blob/main/presentation_video/first_learn.gif" alt="animated" />
</p>

You will find all the code files in the src directory, but if you want to run the code quickly to avoid import errors, because I have not yet tested on other machine than mine with my set up, you can directly run **Project_presentation.ipynb**, where you just have to follow the basic instructions at the beginning and run one by one the code cells

Important this project was realized with **Python 3.7.9** and with the last version of the libraries used.
The project was based on the Windows 0.9.12 version of Carla, but there should be no problem running it with the Linux or Ubuntu versions, although I have not tested it yet !


This project was done during 1 weekend, and it still needs a lot of changes and improvements. The improvements and changes for the continuation of this project will be :

1. Change the neural network to a convolution type network (CNN). Which is much more suitable when using image data as input.

2. A SSH connection to the faculty computers, rather than a local connection, to avoid latencies (on my PC), because my GPU is not powerful enough to support the Simulator and the model training with CUDA, which uses all the GPU without limitation. This makes the simulator slow down and creates a latency.

3. Documentation on a way to have a better method to define the reward functions, because until now, it was defined in an arbitrary way. The EPIC method for Equivalent-Policy Invariant Comparison, seems to be a promising way.
    
4. Then an optimization of the hypers-parameters, with Optuna and also with the NAS method, to automate the design of artificial neural networks, so that it is optimal !

5. To test also other type of sensor that Carla proposes, because there we use only the front RGB camera of the vehicle.

### Result obtained with the second environment which calculates the reward according to the distance.

<p align="center">
  <img src="https://github.com/ZuoNicolas/DRL-for_Autonomous_Driving/blob/main/presentation_video/second_learn.gif" alt="animated" />
</p>

### Evolution of the score over time, with epsilon the exploration rate over time, with a minimum of 0.10 :

<p align="center">
  <img src="https://github.com/ZuoNicolas/DRL-for_Autonomous_Driving/blob/main/data/DQN-EnvDistanceReward-300-1635362488.656119.jpg" />
</p>

## Summary:
The result obtained satisfies my initial expectations, that is to say a very simple model so that the vehicle can advance in its environment without colliding. We can see that it is still a bit unstable, because the vehicle moves in all directions, but this is normal, because the vehicle does not take into account the signs and markings on the ground, which are not yet implemented. But as said before, a lot of improvement can still be made to this model.
