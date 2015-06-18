#!/bin/python
# Class/functions for (finite) MDP, with policy-solving via value iteration.

import numpy as np
from agents import PolicyFun

def policy_iteration(n_states, actions, rewards, discount):
    """
    Perform policy iteration to find the optimal policy for a MDP.
    (returns a policy)
    """
    # TODO: all of this
    policy = xrange(nstates)
    return policy

def value_iteration(n_states, actions, rewards, discount, eps):
    """
    Perform value iteration to find the optimal policy for a MDP.
    (returns a policy)
    """
    threshold = eps*(1-discount)/discount
    value_fns = np.zeros(n_states)
    new_value_fns = np.zeros(n_states)
    deltaUs = np.zeros(n_states)
    policy = PolicyFun(deterministic=True, finite=True, fun=None)
    policy_fun = [0]*n_states
    delta = 1
    n_iter = 0
    while delta > threshold:
        n_iter += 1
        # precompute transitions with value_fns
        # WARNING: action.function is liable to change semantically
        nbh_value_fns = [np.dot(a.function, value_fns) for a in actions]
        for s in xrange(n_states):
            new_value_fns[s] = rewards[s] + \
                               discount*np.max([nbh_u[s] for nbh_u in nbh_value_fns])
            deltaUs[s] = abs(new_value_fns[s] - value_fns[s])
        delta = max(deltaUs)
        value_fns[:] = new_value_fns[:]
    print 'Optimal policy obtained after', n_iter, 
    print 'iterations using threshold of', threshold
    # when this is over, we have our optimal value_fns, and can get the policy
    nbh_value_fns = [np.dot(a.function, value_fns) for a in actions]
    for s in xrange(n_states):
        policy_fun[s] = np.argmax([nbh_u[s] for nbh_u in nbh_value_fns])
    policy.function = policy_fun
    return policy

class MDP:
    def __init__(self, n_states, actions, rewards, discount):
        """
        Define a (finite) Markov decision process.
        Assume discounted rewards.
        """
        self.n_states = n_states        # integer
        self.actions = actions          # ActionFun
        self.rewards = rewards          # vector length n_states
        self.discount = discount        # float
        self.policy = None              # PolicyFun
                                        # for now, this is deterministic
    def solve(self, rewards=None, eps=0.01, method='value_iteration'):
        """
        Obtain optimal policy using value iteration (for now).
        """
        # these assertions might be a bit much
        assert not self.rewards is None
        assert not self.discount is None
        if method == 'value_iteration':
            optimal_policy = value_iteration(self.n_states,
                                             self.actions,
                                             self.rewards,
                                             self.discount,
                                             eps)
        else:
            print 'ERROR: method', method, 'not implemented.'
            return False
        self.policy = optimal_policy
        return True
