"""
Author: John Mansfield

Blackjack wrapper that modifies the observation space and creates a transition/reward matrix P.

"""

import gymnasium as gym
import os
import pickle
import numpy as np


class CustomTransformObservation(gym.ObservationWrapper):
    def __init__(self, env, func, observation_space):
        """
        Parameters
        ----------------------------
        env {gymnasium.Env}:
            Blackjack base environment to be wrapped

        func {lambda}:
            Function that converts the observation

        observation_space {gymnasium.spaces.Space}:
            New observation space
        """
        super().__init__(env)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def observation(self, observation):
        """
        Parameters
        ----------------------------
        observation {Tuple}:
            Blackjack base environment observation tuple

        Returns
        ----------------------------
        func(observation) {int}
        """
        return self.func(observation)

class MountainCartWrapper(gym.Wrapper):
    def __init__(self, env, discrete_step=20):
        """
        Parameters
        ----------------------------
        env {gymnasium.Env}:
            Blackjack base environment

        Explanation of convert_state_obs lambda:
        Lambda function assigned to the variable `self._convert_state_obs` takes parameter, `state` and
        converts the input into a compact single integer value by concatenating player hand with dealer card.
        See comments above for further information.

        """
        self.discret_step = discrete_step
        def convert_state_obs(obs):
            low = env.observation_space.low
            high = env.observation_space.high
            discrete_step=self.discret_step
            state = (int((obs[0] - low[0]) / (high[0] - low[0]) * discrete_step), int((obs[1] - low[1]) / (high[1] - low[1]) * discrete_step))
            return state

        def inverse_transform_obs(state):
            low = env.observation_space.low
            high = env.observation_space.high
            discrete_step=self.discret_step
            obs = (state[0] / discrete_step * (high[0] - low[0]) + low[0], state[1] / discrete_step * (high[1] - low[1]) + low[1])
            return obs

        def convert_2d_to_1d(state):
            return state[0] * self.discret_step + state[1]

        def convert_1d_to_2d(state):
            print(state // self.discret_step, state % self.discret_step)
            return state // self.discret_step, state % self.discret_step

        self._transform_obs = convert_state_obs
        self._inverse_transform_obs = inverse_transform_obs
        self.convert_2d_to_1d = convert_2d_to_1d
        self.convert_1d_to_2d = convert_1d_to_2d
        env = CustomTransformObservation(env, self._transform_obs, env.observation_space)
        super().__init__(env)
        self._P = self._create_transition_matrix()

    @property
    def P(self):
        """
        Returns
        ----------------------------
        P {numpy array}, shape(nS, nA, nS):
            Transition matrix
        """
        return self._P

    def _create_transition_matrix(self):
        P={}
        #the format of the dictionary is {state: {action: [(probability, next_state, reward, done)]}}
       # velocityt + 1 = velocityt + (action - 1) * force - cos(3 * positiont) * gravity

        #positiont + 1 = positiont + velocityt + 1
        for i in range(self.discret_step):
            for j in range(self.discret_step):
                index= self.convert_2d_to_1d((i, j))
                P[index] = {}
                for a in range(3):
                    # action 0: push left, action 1: no push, action 2: push right
                    P[index][a] = []
                    position, velocity = self._inverse_transform_obs((i, j))
                    velocity += (a - 1) * 0.001 + np.cos(3 * position) * (-0.0025)
                    velocity = np.clip(velocity, -0.07, 0.07)
                    position += velocity
                    position = np.clip(position, -1.2, 0.6)
                    if position == -1.2:
                        velocity = 0
                    done = bool(position >= 0.55)
                    reward = 0
                    if done:
                        reward = 1
                    next_state = self._transform_obs((position, velocity))
                    next_state1d= self.convert_2d_to_1d(next_state)
                    P[index][a].append((1, next_state1d, reward, done))
        return P










    @property
    def transform_obs(self):
        """
        Returns
        ----------------------------
        _transform_obs {lambda}
        """
        return self._transform_obs