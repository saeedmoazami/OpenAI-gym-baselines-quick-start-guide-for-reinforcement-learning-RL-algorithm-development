<h1 style="text-align: justify;">OpenAI (gym, baselines) quick start guide for reinforcement learning algorithm development</h1>
<p style="text-align: justify;"><br />This document provides the first required steps for getting everything ready to research in reinforcement learning using OpenAI gym simulation environments and PyBullet physics engines. The are a few things that should be noted:</p>
<ul style="text-align: justify;">
<li>The aim of this repository is to save the initial time students spend to start working on reinforcement learning.</li>
<li>You can NOT learn reinforcement learning here, it has been assumed that you have some level of familiarity with&nbsp;reinforcement learning and want to use this document to start tools that require to implement algorithms quickly.</li>
</ul>
<p style="text-align: justify;">This document has been provided as part of my teaching assistant responsibilities for the machine learning course at Lamar University, instructed by Prof. P. Doerschuk.</p>
<p style="text-align: justify;">Please feel free to contact me (<a href="mailto:moazami.iut@gmail.com">moazami.iut@gmail.com</a>) if you have any questions.</p>
<p style="text-align: justify;">&nbsp;</p>
<ul style="text-align: justify;">
<li>
<h2><strong>Python installation:</strong></h2>
</li>
</ul>
<p style="text-align: justify;">I highly recommend the installation of the latest python version using anaconda distribution:</p>
<p style="text-align: justify;"><a href="http://www.anaconda.com/distribution/">www.anaconda.com/distribution/</a></p>
<p style="text-align: justify;">please make sure to download the correct version selecting the appropriate operating system and python 3.X version. You can go through installation steps using the instructed proved by anaconda:</p>
<p style="text-align: justify;"><a href="https://docs.anaconda.com/anaconda/install/">docs.anaconda.com/anaconda/install/</a></p>
<p style="text-align: justify;">&nbsp;</p>
<p style="text-align: justify;">Run jupyter notebook after installation. It will open a browser.</p>
<p style="text-align: justify;"><img src="https://github.com/saeedmoazami/OpenAI-gym-baselines-quick-start-guide-for-reinforcement-learning-RL-algorithm-development/blob/master/Jupyter_Notebook.png" alt="Jupyter_Notebook" width="300" height="38" /></p>
<p style="text-align: justify;">You can start writing your python cone in jupyter now. Direct to a directory and create a new python3 file.</p>
<p style="text-align: justify;">You also can see other installed tools through anaconda navigator installed on your system.</p>
<p style="text-align: justify;">&nbsp;</p>
<p style="text-align: justify;">&nbsp;</p>
<ul style="text-align: justify;">
<li>
<h2><strong>OpenAI gym installation</strong></h2>
</li>
</ul>
<ul style="list-style-type: circle;">
<li>
<h3>Basicinstallation:</h3>
</li>
</ul>
<p>Run this code in your jupyeter notebook:</p>
<p style="text-align: justify;">``` Shell<br />$ pip install gym<br />```</p>
<p style="text-align: justify;">This installs everything you need to start developing a basic reinforcement learning algorithm.</p>
<ul style="list-style-type: circle;">
<li><strong>reinforcement learning algorithm structures using OpenAI:</strong></li>
</ul>
<p style="text-align: justify;">In order to develop and test your deep reinforcement learning algorithms, you will need to build artificial neural networks using a python library such as Keras, TensorFlow, or PyTorch:</p>
``` Shell
pip install keras
pip install tensorflow tensorflow==1.15

```
Please avoid installing tensorflow 2.X at this point.
You can refer to tensorflow installation guide for more information:
tensorflow.org/install/pip

This is the simplest possible path to start implementing your first algorithm in reinforcement learning on your local machine. The structure of the code will be something like the following:


``` Python
# *******************************************************************************************
# Import required libraries.

import numpy as np
import gym

# *******************************************************************************************
# Create the gym environment.

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print("State space size: ",state_size)
print("Action space size: ", action_size,"\n")

n_episodes = 20

# *******************************************************************************************
# Define the agent class, Depending on the algrithm this can have many methods (functions)
# act() function is the policy that receives an observation and returns action. 

class Agent:
    
    def __init__(self, state_size, action_size):
        self.state_size    = state_size
        self.action_size   = action_size        
               
    def act(self, state):
        act = np.random.randint(self.action_size)
        return act

# *******************************************************************************************
# Instanciate an agent

agent = Agent(state_size, action_size)    

done = False                      # done is a boolean variable and is set to true when the algorithm terminates

for episode in range(n_episodes): # This is the total number of episodes loop
    
    state = env.reset()           # We should reset() the environment to start from an initial random ...
                                  # state at the beginning of every episode 
        
    for time_step in range(200):  # 200 is the maximum time step for CartPole problem
        
        action = agent.act(state) # The agent receives the observation (state) and takes actions accordingly
        
        env.render()              # Comment out to prevet rendering to save time.

        next_state, reward, done, _ = env.step(action) # step() receives action and returns next step, reward, done,
                                                       # info. it actually runs the dynamics of the env for one step
    
        state = next_state
        
        if done:                  # done indicates termination of an episode
            
            print("Episode: {}/{}, Score: {}".format(episode+1, n_episodes, time_step))
            
            break    
```
