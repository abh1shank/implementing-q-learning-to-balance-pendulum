import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from qlearning import Q_learning
env=gym.make('CartPole-v1')
(state,_)=env.reset()
UB=env.observation_space.high
LB=env.observation_space.low
cart_min_vel=-3
cart_max_vel=3
angle_min=-10
angle_max=10
UB[1]=cart_max_vel
UB[3]=angle_max
LB[1]=cart_min_vel
LB[3]=angle_min

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

alpha = 0.1
gamma = 1
epsilon = 0.2
numberEpisodes = 15000
Q1=Q_learning(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,LB,UB)
Q1.simulateEpisodes()
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()
plt.figure(figsize=(12, 5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()
plt.savefig('convergence.png')
env1.close()
np.sum(obtainedRewardsOptimal)
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()