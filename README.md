﻿# Inverted-Pendulum-balancing-with-Q-learning
Steps Involved:
1) Defining learning rate alpha, discount factor gamma
2) Defining epsilon for epsilon greedy to derive the next action ( Exploration vs Exploitation rate )
3) the cart pole/ inverted pendulum environment has continuous space, there for it needs to be discretized,
   the discrete state is created by assigning, all state attributes i.e. cart's position, cart velocity, pendulum angle, and pendulum's angular velocity within the range
   0.00 to 0.25 are assigned index 0, 0.26 to 0.50 are assigned index 1, and so on.
4) Simulation of episodes to generate state action matrix

   given below is the sum of rewards vs episode plot:

   ![episode vs reward](https://github.com/abh1shank/Inverted-Pendulum-balancing-with-Q-learning/assets/97939389/b5ccb3d8-0d66-489f-b2a3-06d1c5bc4017)

    After learning for 15000 simulated episodes:

  ![FormatFactory Screen Record20240410_235212](https://github.com/abh1shank/Inverted-Pendulum-balancing-with-Q-learning/assets/97939389/f21ddd44-95c5-43a5-aac4-20e034c4b395)
