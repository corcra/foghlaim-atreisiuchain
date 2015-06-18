#!/bin/python
# Generate a trajectory from a random, finite MDP.

from MDP.agents import ActionFun, Environment, Agent, generate_trajectory
from MDP.mdp import MDP
import numpy as np

# length of trajectory
N = 100
# number of states
S = 4
# number of actions
A = 4
# rewards (state-specific)
rewards = np.random.normal(size=S)

# --- construct actions --- #
actions = []
for a in xrange(A):
    T = abs(np.random.normal(size=(S, S)))
    T /= np.sum(T, axis=1).reshape(-1,1)
    act = ActionFun(deterministic=False, finite=True, fun=T)
    actions.append(act)

# --- get optimal policy --- #
discount = 0.8
mdp = MDP(S, actions, rewards, discount)
mdp.solve()
optimal_policy = mdp.policy

# --- construct environment --- #
s0 = np.random.randint(S)
environment = Environment(s0)

# --- construct agent --- #
agent = Agent(actions, optimal_policy)

# --- generate trajectory --- #
trajectory = generate_trajectory(agent, environment, N)

# --- save --- #
fname = './test/test_N'+str(N)
np.save(fname+'_MDP', mdp)
np.save(fname+'_traj', trajectory)
