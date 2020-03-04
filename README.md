<h1 style="text-align: center;"> OpenAI (gym, baselines) quick start guide for reinforcement learning algorithm development </h1>
<p style="text-align: justify;"><br/>This document provides the first required steps for getting everything ready to research in reinforcement learning using OpenAI gym simulation environments and PyBullet physics engines. The are a few things that should be noted:</p>
<ul style="text-align: justify;">
<li>The aim of this repository is to save the initial time students spend to start working on reinforcement learning.</li>
<li>You can NOT learn reinforcement learning here, it has been assumed that you have some level of familiarity with&nbsp;reinforcement learning and want to use this document to start tools that require to implement algorithms quickly.</li>
</ul>
<p style="text-align: justify;">This document has been provided as part of my teaching assistant responsibilities for the machine learning course at Lamar University, instructed by Prof. P. Doerschuk.</p>
<p style="text-align: justify;">Please feel free to contact me (<a href="mailto:moazami.iut@gmail.com">moazami.iut@gmail.com</a>) if you have any questions.</p>
<ul style="text-align: justify;">
<li>
<h2><strong>Python installation:</strong></h2>
</li>
</ul>
<p style="text-align: justify;">I highly recommend the installation of the latest python version using anaconda distribution:</p>
<p style="text-align: justify;"><a href="http://www.anaconda.com/distribution/">www.anaconda.com/distribution/</a></p>
<p style="text-align: justify;">please make sure to download the correct version selecting the appropriate operating system and python 3.X version. You can go through installation steps using the instructed proved by anaconda:</p>
<p style="text-align: justify;"><a href="https://docs.anaconda.com/anaconda/install/">docs.anaconda.com/anaconda/install/</a></p>
<p style="text-align: justify;">Run jupyter notebook after installation. It will open a browser.</p>
<p style="text-align: justify;">You can start writing your python cone in jupyter now. Direct to a directory and create a new python3 file.</p>
<p style="text-align: justify;">You also can see other installed tools through anaconda navigator installed on your system.</p>
<ul style="text-align: justify;">
<li>
<h2><strong>OpenAI gym installation</strong></h2>
</li>
</ul>
<ul style="list-style-type: circle;">
<li>
<h3>Basic Installation:</h3>
</li>
</ul>
<p>Run this code in your jupyeter notebook:</p>

``` Shell
$ pip install gym
```

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

<ul>
<li>
<h2><strong>Setup on a remote machine :</strong></h2>
</li>
</ul>

<p>This section is based on lamar university&rsquo;s cluster computer but general instructions can be used in other settings too.</p>
<p>You need to arrange with the computer science department in order to access the cluster computer</p>
<p>Once you loged in:</p>
<p><img src="https://github.com/saeedmoazami/OpenAI-gym-baselines-quick-start-guide-for-reinforcement-learning-RL-algorithm-development/blob/master/power%20shell.png" alt="" /></p>

``` Shell
$ module load anaconda3/3.7
```
<p>Please note that you need to type module add ana.. then use Tab key to auto complete the rest </p>

``` Shell
$ conda create -n NewEnv
```
<p>You can list all available environments using:</p>

``` Shell
$ conda info &ndash;-envs
```

<p>Activate the created environments using :</p>
``` Shell
$ source activate NewEnv
```
<p>Or</p>
``` Shell
$ conda activate NewEnv
```
<p>Then install python and TensorFlowfor your environment</p>
``` Shell
conda install python=3.6
$ pip install tensorflow==1.14
```
<p>To verify the installation, run python, then use this code to check installation and verion (you should see 1.14.0) </p>

``` Python
import tensorfflow as tf
print(tf.__version__)
```
<p>The following libraries are required for gym and baselines toi run properly:</p>
``` Shell
conda install python=3.6
$ pip install pystan
$ pip install joblib
$ pip install click
$ pip install tqdm
```

<p>Finally run the following code to see if everything works properly:</p>
``` Shell
conda install python=3.6
$ python baselines/deepq/experiments/train_cartpole.py
```

<p>To see rendering envs (or any graphical windows from remove computer) you will need to install x11:</p>
<p>Download and install xming using this link and make sure that it is running in your computer before connecting</p>
<p><a href="https://sourceforge.net/projects/xming/">https://sourceforge.net/projects/xming/</a></p>

<ul style="text-align: justify;">
<li>
<h2><strong>Advanced installation:</strong></h2>
</li>
</ul>

<p>Up to this point you have everything you need to do simple projects in reinforcement learning, But, if you need to go deeper you can follow the following steps:</p>

<ul style="list-style-type: circle;">
<li>
<h3>OpenAI Baselines:</h3>
</li>
</ul>

<p>OpenAI baselines is a set of high-quality standardized algorithms in reinforcement learning developed by OpenAI to provide a baseline for newly developed algorithms to be compared with.</p>
<p>To install baselines, you can follow this instruction:</p>
<p>You will need to install anaconda as it is explained in the previous section.</p>
<p>A C++ compiler is needed to be installed. You can find the &ldquo;community&rdquo; version from the following:</p>
<p><a href="https://visualstudio.microsoft.com/downloads/">visualstudio.microsoft.com/downloads/</a></p>
<p>run Anaconda Power Shell, on your machine</p>
``` Shell
$ conda create -n NewEnv
```
<p>please select a convenient name for NewEnv</p>
<p>you can list all environments using:</p>
``` Shell
$ conda&nbsp;info --envs
```
<p>activate the environment you have created:</p>
``` Shell
$ conda activate NewEnv
```
<p>Install git</p>
``` Shell
$ conda install git
```
<p>Follow the following:</p>
``` Shell
$ pip install git+https://github.com/Kojoley/atari-py.git
```
<p>You can use pip install gym, or clone to inspect the source code</p>
``` Shell
$ git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
<p>These are required libraries. They are not required for all environments BTW:</p>
``` Shell
$ conda install pystan
$ pip install joblib
$ pip install click
$ pip install tqdm
```
<p>Install the baselines:</p>
``` Shell
$ git clone https://github.com/openai/baselines.git
$ cd baseline
$ pip install -e .
```
<p> After finishing the installation, you can use the following commands to run baselines algorithms.To train a cartpole using DQN, run </p>
``` Shell
$ python baselines/deepq/experiments/train_cartpole.py
```
<p>To see a trained cartpole run:</p>
``` Shell
$ python baselines/deepq/experiments/enjoy_cartpole.py
```
<p>Please refer to the following for the detailed explanation:</p>
<p><a href="https://arztsamuel.github.io/en/blogs/2018/Gym-and-Baselines-on-Windows.html">arztsamuel.github.io/en/blogs/2018/Gym-and-Baselines-on-Windows.html</a></p>
<p>Please ignore any errors at this point. Specifically errors related to MuJoCo. MuJoCo is a library for multibody dynamics analysis that is NOT FREE. You will not need to use it now. A limited free license is available for students here, but I recommend using the free license when you are ready to implement codes on continuous state-action spaces. You can find more information here:<a href="http://www.mujoco.org/">http://www.mujoco.org/</a>Also, Pybyllet is a great FREE library that you can use after becoming more proficient in reinforcement learning.<a href="https://pybullet.org/wordpress/">pybullet.org/wordpress/</a></p>
<p>Running baseline:</p>
<p>OpenAI baselines provides an interface to run different algorithms using different setting:</p>
``` Shell
$ python -m baselines.run --alg=&lt;name of the algorithm&gt; --env=&lt;environment_id&gt; [additional arguments]
```
<p>Here are some examples:</p>
``` Shell
$ python -m baselines.run --alg=deepq --env=CartPole-v1 --network=mlp --num_timesteps=5e6 --num_hidden=32&nbsp; --save_path=~/models/deepq5e6_01 --log_path=~/logs/deepq5e6_01/ --lr=1e-3&nbsp; --buffer_size=50000 --seed=0</p>
```
<p>This command, runs trains an DQN (deepq) agent, on CartPole-v1 environment. The structure of the network will be mlp(multi layer perceptron), the algorithm wili be run for 5e6 time steps, number of hidden nodes of mlp is 32, saves the model (the agent&rsquo;s model) in save path, logs the results on log_path, uses learning rate 1e-3, and replay buffer size 5000, and random seed 0.</p>
<p>You should be familiar with DQN before understanding most of these terms.</p>
<p>You can refer to this documents for Saving the mode, Logging, and visyualizing the results:</p>
<p><a href="https://github.com/openai/baselines">https://github.com/openai/baselines</a></p>
<p>and here for visualization:</p>
<p><a href="https://github.com/openai/baselines/blob/master/docs/viz/viz.ipynb">https://github.com/openai/baselines/blob/master/docs/viz/viz.ipynb</a></p>


