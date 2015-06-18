#!/bin/python

import numpy as np

class Environment(object):
    def __init__(self, state):
        self.state = state
        self.intrinsic_dynamics = lambda x: x
    def evolve(self, evolution_function=None):
        if evolution_function == None:
            evolution_function = self.intrinsic_dynamics
        self.state = evolution_function(self.state)

class Agent(object):
    def __init__(self, actions, policy):
        self.actions = actions
        self.policy = policy
    def select_action(self, state):
        which_action = self.policy(state)
        return which_action, self.actions[which_action]

class PolicyFun:
    """
    This is a class of functions which take in a state
        (state can be continuous or discrete)
    and output a CHOICE of action.
        (can be deterministic or probabilistic)
    TODO: reduce boilerplate (possibly merge with ActionFun)
    """
    def __init__(self, deterministic, finite, fun):
        self.deterministic = deterministic
        self.finite = finite
        self.function = fun
    def __call__(self, state):
        if self.finite:
            if self.deterministic:
                return self.function[state]
            else:
                # the 'function' is actually a matrix of transition probs
                probs = self.function[state, :]
                action = np.random.choice(xrange(len(probs)), p=probs)
                return action
        else:
            if self.deterministic:
                return self.function[state]
            else:
                return self.function(state)
                
class ActionFun:
    """
    This is a class of functions which take in a state
        (state can be continuous or discrete)
    and output a new state.
        (can be deterministic or probabilistic)
    """
    def __init__(self, deterministic, finite, fun):
        self.deterministic = deterministic
        self.finite = finite
        self.function = fun
    def __call__(self, state):
        if self.finite:
            if self.deterministic:
                return self.function[state]
            else:
                # the 'function' is actually a matrix of transition probs
                probs = self.function[state, :]
                newstate = np.random.choice(xrange(len(probs)), p=probs)
                return newstate
        else:
            if self.deterministic:
                return self.function[state]
            else:
                return self.function(state)
