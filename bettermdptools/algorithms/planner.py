"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Model-based learning algorithms: Value Iteration and Policy Iteration

Assumes prior knowledge of the type of reward available to the agent
for iterating to an optimal policy and reward value for a given MDP.
"""

import numpy as np
import warnings
from bettermdptools.utils.decorators import print_runtime
import time


class Planner:
    def __init__(self, P):
        self.P = P

    @print_runtime
    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.

        pi_track {list[dict]}:
            Log of policy for each iteration
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        # V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        V_track = None
        pi_track = []
        timings = []
        i = 0
        total_q_updates = 0
        converged = False
        while i < n_iters-1 and not converged:
            start = time.time()
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
                        total_q_updates += 1
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            V = np.max(Q, axis=1)
            # V_track[i] = V
            pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
            pi_track.append(pi)
            timings.append(time.time() - start)
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check theta and n_iters.  ")

        print(f"Total Q updates: {total_q_updates}")
        pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
        return V, V_track, pi, pi_track, timings

    @print_runtime
    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10, seed=42):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        np.random.seed(seed)
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = []
        pi_track = []
        timings = []
        i = 0
        total_updates = 0
        converged = False
        while i < n_iters-1 and not converged:
            start = time.time()
            i += 1
            old_pi = pi
            V, v_updates = self.policy_evaluation(pi, V, gamma, theta)
            V_track.append(V)
            pi, q_updates = self.policy_improvement(V, gamma)
            total_updates += q_updates + v_updates
            pi_track.append(pi)
            timings.append(time.time() - start)
            if old_pi == pi:
                converged = True
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")

        print(f"Total Q updates: {total_updates}")
        return V, V_track, pi, pi_track, timings


    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        v_updates_count = 0
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                    v_updates_count += 1
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V, v_updates_count


    def policy_improvement(self, V, gamma=1.0):
        q_updates_count = 0
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
                    q_updates_count += 1

        new_pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return new_pi, q_updates_count
