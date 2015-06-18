#!/bin/python
#
# My attempt at implementing SCIRL from:
# Klein, Edouard, et al. "Inverse reinforcement learning through structured
# classification." Advances in Neural Information Processing Systems. 2012
# http://papers.nips.cc/paper/4551-inverse
#

import numpy as np
import sys

# === feature expectations === #
# Here we provide expert feature expectations.
# In practice, this means we take in either:
#   - a model of the system 
#     (that is, transition functions, discount factor gamma)
#   OR
#   - trajectories (potentially simply from the expert)
# In deference to the approach taken in Klein et al. (NIPS 2012) I will use the
# latter strategy.

def feature_expectations_heuristic(trajectory, features, gamma):
    """
    Calculate expert feature expectations using the heuristic given by
    equation 5.
    """
    N = trajectory.shape[0]
    S = features.shape[1]
    k = features.shape[0]
    A = max(trajectory[:, 1]) + 1
    feature_expectations = np.zeros(shape=(k, S, A))
    gamma_powers = [gamma**n for n in xrange(N)]
    for (i, (s, a)) in enumerate(trajectory):
        feature_expectations[:, s, a] = 0
        for j in xrange(i, N):
            feature_expectations[:, s, a] += gamma_powers[j-i]*features[:, trajectory[j, 0]]
        for aa in xrange(A):
            if not aa == a:
                feature_expectations[:, s, aa] = gamma*feature_expectations[:, s, a]
    return feature_expectations

# ==== linearly parametrised score-function based multi-class classifier === #
# Use structured large-margin approach and subgradient descent.

class clf_large_margin(object):
    """
    Multi-class classifier
    """
    def __init__(self, features, parameters=None):
        self.features = features            # mu (k x |S| x |A|)
        self.k = self.features.shape[0]
        self.S = self.features.shape[1]
        self.A = self.features.shape[2]
        if parameters is None:
            parameters = np.zeros(self.k)
        else:
            assert parameters.shape[0] == self.k
        self.params = parameters            # theta (k)
    def objective(self, data, lambd=0.5, params=None):
        """ See section 4.2 """
        N = data.shape[0]
        action_indices = data[:, 1]
        state_indices = data[:, 0]
        loss = np.ones(shape=(N, self.A))           # (N x |A|)
        loss[[range(N), action_indices]] = 0        # zero at true actions
        state_features = self.features[:, state_indices, :] # (k x N x |A|)
        if params is None:
            weighted_state_features = np.einsum('i,ijk', 
                                                self.params,
                                                state_features)     # (N x |A|)
        else:
            weighted_state_features = np.einsum('i,ijk',
                                                params,
                                                state_features)     # (N x |A|)
        term1 = np.max(weighted_state_features + loss, axis=1)      # (N)
        term2 = weighted_state_features[[range(N), action_indices]] # (N)
        objective = np.mean(term1 - term2) + \
                    0.5*lambd*np.dot(self.params,self.params)       # (1)
        return objective
    def fit(self, data, alpha=0.1, lambd=0.5, threshold=1e-6, max_iter=10000):
        """ Subgradient descent on the objective function """
        N = data.shape[0]
        delta_objective = -1
        objective = self.objective(data)
        state_indices = data[:, 0]
        action_indices = data[:, 1]
        state_features= self.features[:, state_indices, :]  # (k x N x |A|)
        current_objective = self.objective(data)
        iteration = 0
        while abs(delta_objective) > threshold and delta_objective < 0:
            weighted_state_features = np.einsum('i,ijk',
                                                self.params,
                                                state_features)     # (N x |A|)
            max_action_indices = np.argmax(weighted_state_features, axis=1)
            term1 = np.array([state_features[:, i, max_action_indices[i]] \
                              for i in xrange(N)])                  # (N x k)
                                                            # NOTE TRANSPOSED
            term2 = np.array([state_features[:, i, action_indices[i]] \
                              for i in xrange(N)])                  # (N x k)
            diff_term = np.mean(term1 - term2, axis=0)      # (k)
            subgradient = diff_term + lambd*self.params
            self.params -= alpha*subgradient
            new_objective = self.objective(data)
            delta_objective = new_objective - current_objective
            current_objective = new_objective
            iteration += 1
            if iteration > max_iter:
                print 'WARNING: Hit max iterations before convergence.'
                break
            print iteration, current_objective, delta_objective
        return True


# === loading/saving data === #

def get_training_data(traj_path, basis_path):
    # --- grab trajectory --- #
    try:
        trajectory = np.load(traj_path)
        assert trajectory.shape[1] == 2     # need both actions & state labels
    except IOError:
        sys.exit('ERROR: Please provide a valid .npy file.')
    # --- grab basis functions --- #
    # note, these go from S -> R^k
    try:
        basis_functions = np.load(basis_path)
        k = basis_functions.shape[0]
        S = basis_functions.shape[1]
        assert S >= max(trajectory[:, 0])
    except IOError:
        sys.exit('ERROR: Please provide a valid .npy file.')
    return trajectory, basis_functions

# === SCIRL === #
# Putting it together...

def SCIRL(trajectory, basis_functions, GAMMA=0.9):
    # --- get feature expectations --- #
    mu = feature_expectations_heuristic(trajectory, basis_functions, GAMMA)
    # --- get coefficients (theta) --- #
    clf = clf_large_margin(mu)
    clf.fit(trajectory)
    theta = clf.params
    # --- construct reward function --- #
    reward = np.dot(theta, basis_functions) # (|S|)
    return reward
