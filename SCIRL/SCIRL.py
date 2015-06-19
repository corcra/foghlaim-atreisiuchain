#!/bin/python
#
# My attempt at implementing SCIRL from:
# Klein, Edouard, et al. "Inverse reinforcement learning through structured
# classification." Advances in Neural Information Processing Systems. 2012
# http://papers.nips.cc/paper/4551-inverse
#

from MDP import mdp
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

def feature_expectations_heuristic(trajectory, features, gamma, n_actions):
    """
    Calculate expert feature expectations using the heuristic given by
    equation 5.
    This is mad slow and doesn't seem to correspond to results from the 'exact'
    method.
    """
    N = trajectory.shape[0]
    S = features.shape[0]
    k = features.shape[1]
    A = n_actions
    feature_expectations = np.zeros(shape=(A, S, k))
    gamma_powers = [gamma**n for n in xrange(N)]
    for (i, (s, a)) in enumerate(trajectory):
        feature_expectations[a, s, :] = 0
        for j in xrange(i, N):
            feature_expectations[a, s, :] += gamma_powers[j-i]*features[trajectory[j, 0], :]
        for aa in xrange(A):
            if not aa == a:
                feature_expectations[aa, s, :] = gamma*feature_expectations[a, s, :]
    return feature_expectations

def feature_expectations_exact(mdp, features, gamma):
    """
    Calculate expert feature expectations using the explicit expression in
    section 4.1.
    Note that it requires an mdp.
    (technically, it just requires the transitions and the policy)
    """
    S = mdp.n_states
    k = features.shape[1]
    assert S == features.shape[0]
    A = len(mdp.actions)
    assert mdp.policy.deterministic == True
    policy = mdp.policy.function            # |S|, values are action indices
    P_expert = np.zeros(shape=(S, S))
    for s in xrange(S):
        policy_action = policy[s]
        # each row in P_expert is the transition probability assuming one takes
        # the optimal action in that state
        P_expert[s, :] = mdp.actions[policy_action].function[s, :]
    inverse_term = np.linalg.inv(1 + gamma*P_expert)
    feature_expectations = np.zeros(shape=(A, S, k))
    for a in xrange(A):
        feature_expectations[a, :, :] = np.dot((1 + gamma*mdp.actions[a].function*inverse_term), features)
    return feature_expectations

# ==== linearly parametrised score-function based multi-class classifier === #
# Use structured large-margin approach and subgradient descent.

class clf_large_margin(object):
    """
    Multi-class classifier
    """
    def __init__(self, features, parameters=None):
        self.features = features            # mu (|A| x |S| x k)
        self.A = self.features.shape[0]
        self.S = self.features.shape[1]
        self.k = self.features.shape[2]
        if parameters is None:
            parameters = np.random.normal(size=self.k)
            #parameters = np.zeros(shape=self.k)
        else:
            assert parameters.shape[0] == self.k
        self.params = parameters            # theta (k)
    def objective(self, data, lambd=0.5):
        """ See section 4.2 """
        N = data.shape[0]
        action_indices = data[:, 1]
        state_indices = data[:, 0]
        loss = np.ones(shape=(self.A, N))           # (|A| x N)
        loss[[action_indices, range(N)]] = 0        # zero at true actions
        state_features = self.features[:, state_indices, :] # (|A| x N x k)
        weighted_state_features = np.dot(state_features, self.params) # (|A| x N)
        term1 = np.max(weighted_state_features + loss, axis=0)      # (N)
        term2 = weighted_state_features[[action_indices, range(N)]] # (N)
        objective = np.mean(term1 - term2) + \
                    0.5*lambd*np.dot(self.params,self.params)       # (1)
        return objective
    def fit(self, data, alpha=0.1, lambd=0.5, threshold=1e-6, max_iter=10000, 
            verbose=2):
        """ Subgradient descent on the objective function """
        N = data.shape[0]
        delta_objective = -1
        state_indices = data[:, 0]
        action_indices = data[:, 1]
        state_features= self.features[:, state_indices, :]  # (|A| x N x k)
        current_objective = self.objective(data)
        iteration = 0
        while abs(delta_objective) > threshold and delta_objective < 0:
            if verbose > 1: print iteration, current_objective, delta_objective
            weighted_state_features = np.dot(state_features, self.params) # (|A| x N)
            max_action_indices = np.argmax(weighted_state_features, axis=0)
            term1 = np.array([state_features[max_action_indices[i], i, :] \
                              for i in xrange(N)])                  # (N x k)
                                                            # NOTE TRANSPOSED
            term2 = np.array([state_features[action_indices[i], i, :] \
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
            if verbose > 1: print self.accuracy(data)
        if not iteration > max_iter and verbose > 0:
            print 'Converged after', iteration, 'iterations.'
        print ''
        return True
    def predict(self, data):
        """ Predict the labels associated with a list of states. """
        N = data.shape[0]
        state_features = self.features[:, data[:, 0], :]        # (|A| x N x k)
        scores = np.dot(state_features, self.params)            # (|A| x N)
        labels = np.argmax(scores, axis=0)
        return labels
    def accuracy(self, data):
        labels = self.predict(data)
        accuracy = np.mean(labels == data[:, 1])
        return accuracy

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
        S = basis_functions.shape[0]
        k = basis_functions.shape[1]
        assert S >= max(trajectory[:, 0])
    except IOError:
        sys.exit('ERROR: Please provide a valid .npy file.')
    # --- map trajectory to integers --- #
    trajectory = trajectory.astype(int)
    return trajectory, basis_functions

# === SCIRL === #
# Putting it together...

def SCIRL(trajectory, basis_functions, n_actions, GAMMA=0.9, mdp=None):
    # --- get feature expectations --- #
    if mdp is None:
        mu = feature_expectations_heuristic(trajectory, basis_functions, 
                                            GAMMA, n_actions)
    else:
        mu = feature_expectations_exact(mdp, basis_functions, GAMMA)
    # --- get coefficients (theta) --- #
    clf = clf_large_margin(mu)
    clf.fit(trajectory)
    theta = clf.params
    # --- construct reward function --- #
    reward = np.dot(basis_functions, theta) # (|S|)
    return reward
