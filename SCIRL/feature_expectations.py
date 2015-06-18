#!/bin/python
# This script will provide expert feature expectations.
# In practice, this means we take in either:
#   - a model of the system 
#     (that is, transition functions, discount factor gamma)
#   OR
#   - trajectories (potentially simply from the expert)
# In deference to the approach taken in Klein et al. (NIPS 2012) I will use the
# latter strategy.

import numpy as np

def feature_expectations_heuristic(trajectory, features, gamma):
    """
    Calculate feature expectations using the heuristic given by equation 5.
    """
    N = trajectory.shape[0]
    S = features.shape[1]
    k = features.shape[0]
    A = max(trajectory[:, 1]) + 1
    feature_expectations = np.zeros(shape=(k, S, A))
    gamma_powers = [gamma**n for n in xrange(N)]
    for (i, (state, action)) in enumerate(trajectory):
        feature_expectations[:, state, action] = 0
        for j in xrange(i, N):
            feature_expectations[:, state, action] += gamma_powers[j-i]*features[:, trajectory[j, 0]]
        for a in xrange(A):
            if not a == action:
                feature_expectations[:, state, a] = gamma*feature_expectations[:, state, action]
    return feature_expectations
