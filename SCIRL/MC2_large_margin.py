#!/bin/python
# Describes a linearly parametrised score-function based multi-class 
# classifier. :)
# Uses structured large-margin approach and subgradient descent.

import numpy as np

class clf_large_margin(object):
    def __init__(self, features, parameters):
        self.params = parameters            # theta (k)
        self.features = features            # mu (k x |S| x |A|)
        self.k = self.params.shape[0]
        assert self.features.shape[0] == self.k
        self.S = self.features.shape[1]
        self.A = self.features.shape[2]
    def objective(self, data, lambd=0.5, params=None):
        N = data.shape[0]
        action_indices = data[:, 1]
        state_indices = data[:, 0]
        loss = np.ones(shape=(N, self.A))           # (N x |A|)
        loss[[range(N), action_indices]] = 0            # zero at true actions
        state_features = self.features[:, state_indices, :]              # (k x N x |A|)
        if params is None:
            weighted_state_features = np.einsum('i,ijk', self.params, state_features)   # (N x |A|)
        else:
            weighted_state_features = np.einsum('i,ijk', params, state_features)   # (N x |A|)
        term1 = np.max(weighted_state_features + loss, axis=1)          # (N)
        term2 = weighted_state_features[[range(N), action_indices]]    # (N)
        objective = np.mean(term1 - term2) + \
                    0.5*lambd*np.dot(self.params,self.params)         # (1)
        return objective
    def fit(self, data, alpha=0.1, lambd=0.5, threshold=1e-6, max_iter=10000):
        N = data.shape[0]
        delta_objective = -1
        objective = self.objective(data)
        state_indices = data[:, 0]
        action_indices = data[:, 1]
        state_features= self.features[:, state_indices, :]              # (k x N x |A|)
        current_objective = self.objective(data)
        iteration = 0
        while abs(delta_objective) > threshold and delta_objective < 0:
            weighted_state_features = np.einsum('i,ijk', self.params, state_features)   # (N x |A|)
            local_max_action_indices = np.argmax(weighted_state_features, axis=1)   # (N)
            term1 = np.array([state_features[:, i, local_max_action_indices[i]] for i in xrange(N)])    # (N x k) NOTE TRANSPOSED
            term2 = np.array([state_features[:, i, action_indices[i]] for i in xrange(N)])  # (N x k)
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
