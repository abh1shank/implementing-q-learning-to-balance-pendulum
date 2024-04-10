import numpy as np
import gym
import time
class Q_learning:
    def __init__(self, env, alpha, gamma, epsilon, num_eps, number_bins, LB, UB):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_num = env.action_space.n
        self.num_eps = num_eps
        self.num_bins = number_bins
        self.LB = LB
        self.UB = UB
        self.sum_rewards_eps = []
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(number_bins[0], number_bins[1], number_bins[2], number_bins[3], self.action_num))

    def returnIndexState(self, state):
        pos = state[0]
        vel = state[1]
        angle = state[2]
        omega = state[3]

        pendPosBin = np.linspace(self.LB[0], self.UB[0], self.num_bins[0])
        pendVelBin = np.linspace(self.LB[1], self.UB[1], self.num_bins[1])
        pendAngBin = np.linspace(self.LB[2], self.UB[2], self.num_bins[2])
        pendOmegaBin = np.linspace(self.LB[3], self.UB[3], self.num_bins[3])

        indexPosition = np.maximum(np.digitize(state[0], pendPosBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], pendVelBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], pendAngBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3], pendOmegaBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def select_Action(self, state, index):
        if index < 500:
            return np.random.choice(self.action_num)
        randomNumber = np.random.random()
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon
        if randomNumber < self.epsilon:
            return np.random.choice(self.action_num)
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

    def simulateEpisodes(self):
        for indexEpisode in range(self.num_eps):
            rewardsEpisode = []
            (S, _) = self.env.reset()
            S = list(S)

            print("Simulating episode {}".format(indexEpisode))

            terminalState = False
            while not terminalState:
                SIndex = self.returnIndexState(S)
                A = self.select_Action(S, indexEpisode)
                (S_, reward, terminalState, _, _) = self.env.step(A)

                rewardsEpisode.append(reward)
                S_ = list(S_)
                S_Index = self.returnIndexState(S_)

                Qmax_ = np.max(self.Qmatrix[S_Index])

                if not terminalState:
                    error = reward + self.gamma * Qmax_ - self.Qmatrix[SIndex + (A,)]
                    self.Qmatrix[SIndex + (A,)] = self.Qmatrix[SIndex + (A,)] + self.alpha * error
                else:
                    error = reward - self.Qmatrix[SIndex + (A,)]
                    self.Qmatrix[SIndex + (A,)] = self.Qmatrix[SIndex + (A,)] + self.alpha * error

                S = S_

            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sum_rewards_eps.append(np.sum(rewardsEpisode))

    def simulateLearnedStrategy(self):
        env1 = gym.make('CartPole-v1', render_mode='human')
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000
        obtained_rewards = []

        for timeIndex in range(timeSteps):
            print(timeIndex)
            actionInStateS = np.argmax(self.Qmatrix[self.returnIndexState(currentState)])
            currentState, reward, terminated, truncated, info = env1.step(actionInStateS)
            obtained_rewards.append(reward)
            time.sleep(0.05)
            if terminated:
                time.sleep(1)
                break
        return obtained_rewards, env1

    def simulateRandomStrategy(self):
        env2 = gym.make('CartPole-v1')
        (cur_state, _) = env2.reset()
        env2.render()
        episode_num = 100
        time_steps = 1000
        sum_Rewards_eps = []

        for ep_index in range(episode_num):
            reward_single_ep = []
            initial_state = env2.reset()
            print(ep_index)
            for t in range(time_steps):
                random_action = env2.action_space.sample()
                obs, reward, term, trunc, info = env2.step(random_action)
                reward_single_ep.append(reward)
                if term: break
            sum_Rewards_eps.append(sum(reward_single_ep))

        return sum_Rewards_eps, env2