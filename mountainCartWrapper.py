"""
Author: John Mansfield

Blackjack wrapper that modifies the observation space and creates a transition/reward matrix P.

# Transitions and rewards matrix from https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/

Given an action, the mountain car follows the following transition dynamics:

velocityt+1 = velocityt + (action - 1) * force - cos(3 * positiont) * gravity

positiont+1 = positiont + velocityt+1

"""

import numpy as np
import gymnasium as gym

from gymnasium.spaces import Tuple, Box

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
    def __init__(self, env, discrete_size=20):
        self.discrete_size = discrete_size
        """
        Parameters
        ----------------------------
        env {gymnasium.Env}:
            MountainCart base environment

        Explanation of convert_state_obs lambda function
        """
        # def convert_state_obs(obs, discrete_size=self.discrete_size):
        #     """
        #     Converts a continuous observation space to a discrete observation space using a grid 
        #     of size discrete_size x discrete_size
    
        #     Parameters
        #     ----------------------------
        #     obs {Tuple}:
        #         MountainCart base environment observation tuple

        #     Returns
        #     ----------------------------
        #     int:
        #         New observation space value using discrete conversion
        #     """
        #     low = env.observation_space.low
        #     high = env.observation_space.high
        #     discrete_os_win_size = (high - low) / discrete_size
        #     return int((obs[0] - low[0]) / discrete_os_win_size[0]), int((obs[1] - low[1]) / discrete_os_win_size[1])
            
        
        def create_uniform_grid(low, high, bins=(20, 20)):

            grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
            print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
            for l, h, b, splits in zip(low, high, bins, grid):
                print("    [{}, {}] / {} => {}".format(l, h, b, splits))
            return grid 
        
        def discretize(sample):
            return list(int(np.digitize(s, g)) for s, g in zip(sample, self.grid))
        
        def undiscretize(sample):
            pass

        
        def convert_state_obs(obs):
            return tuple(discretize(obs))

        self.discretize = discretize
        self.discrete_to_continuous = undiscretize
        self.grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(discrete_size, discrete_size))
        self._transform_obs = convert_state_obs
        env = CustomTransformObservation(env, self._transform_obs, env.observation_space)
        super().__init__(env)
        
        # create transition and reward matrix
        self._P = self.create_P()


    def create_P(self):
        """
        Creates a transition and reward matrix P for the mountainCart environment

        Returns
        ----------------------------
        P {dict}:
            Dictionary of transition and reward matrix
            the format of the dictionary is {state: {action: [(probability, next_state, reward, done)]}}
        """
        P = {}
        for i in range(self.discrete_size):
            for j in range(self.discrete_size):
                P[(i, j)] = {}
                for action in range(3):
                    P[(i, j)][action] = []
                    for action_ in range(3):
                        # convert state to continuous using grid
                        position, velocity = self.discrete_to_continuous((i, j))
                        new_velocity = velocity + (action_ - 1) * 0.001 - np.cos(3 * position) * 0.0025
                        new_velocity = min(max(-0.07, new_velocity), 0.07)
                        new_position = position + new_velocity
                        new_position = min(max(-1.2, new_position), 0.6)
                        new_position = int((new_position + 1.2) * self.discrete_size / 1.8)
                        new_velocity = int((new_velocity + 0.07) * self.discrete_size / 0.14)
                        done = new_position >= self.discrete_size - 1
                        print(new_position, new_velocity)
                        discrete_new_state = self.discretize((new_position, new_velocity))
                        print(discrete_new_state)
                        reward = 0
                        if done:
                            reward = 1
                        P[(i, j)][action].append((1, discrete_new_state, reward, done))

        return P
    
                   
    

    @property
    def P(self):
        """
        Returns
        ----------------------------
        _P {dict}
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns
        ----------------------------
        _transform_obs {lambda}
        """
        return self._transform_obs