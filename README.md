This document provides the first steps of getting everything ready for research in reinforcement learning using OpenAI gym simulation environments and PyBullet physics engines.
The aim of this repository is to save the initial time students spend to start working on reinforcement learning. 
Please keep in mind that you can not learn reinforcement learning here, you can study RL and use this document to start tools that require to implement algorithms
I have provided this as part of the teaching assistant responsibilities for the machine learning course at Lamar University, instructed by Prof. P. Doerschuk.
lease feel free to contact me (moazami.iut@gmail.com) if you have any questions.

''' python
# *******************************************************************************************
# Environment Initialization

env = gym.make('CartPole-v0')

random.seed(43)
np.random.seed(19)
tf.set_random_seed(96)
env.seed(37)

state_size  = env.observation_space.shape[0]
action_size = env.action_space.n

print("Obseravtion spapce size: {}".format(state_size))
print("Action space size: {}".format(action_size))
'''
