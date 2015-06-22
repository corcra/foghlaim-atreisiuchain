#!/bin/python
# Run SCIRL on a trajectory generated from an MDP.
# Load the true MDP, compare the SCIRL-derived reward function.

from SCIRL import SCIRL
from MDP import mdp
import numpy as np
from copy import deepcopy

traj_path = './data/test_N100_traj.npy'
# using trivial basis functions (identity)
basis_path = './data/test_N100_basis_S4k1.npy'
mdp_path = './data/test_N100_MDP.npy'

# --- get data --- #
trajectory, basis_functions = SCIRL.get_training_data(traj_path, basis_path)

# --- load true MDP --- #
mdp = np.load(mdp_path).item()
MDP_reward = mdp.rewards

# --- run SCIRL --- #
SCIRL_reward_modelfree = SCIRL.SCIRL(trajectory, basis_functions, n_actions=4)
SCIRL_reward_model = SCIRL.SCIRL(trajectory, basis_functions, n_actions=4,mdp=mdp)

# --- compare rewards --- #
print 'SCIRL rewards: (no model)', SCIRL_reward_modelfree
print 'SCIRL rewards: (model)', SCIRL_reward_model
print 'MDP rewards:', MDP_reward

# --- compare optimal poicies according to these rewards --- #
print ''
mdp_SCIRL_modelfree = deepcopy(mdp)
mdp_SCIRL_modelfree.policy.function = SCIRL_reward_modelfree
mdp_SCIRL_modelfree.solve()
mdp_SCIRL_model = deepcopy(mdp)
mdp_SCIRL_model.policy.function = SCIRL_reward_model
mdp_SCIRL_model.solve()
print ''

print 'SCIRL policy: (no model)', mdp_SCIRL_modelfree.policy.function
print 'SCIRL policy: (model)', mdp_SCIRL_model.policy.function
print 'ground truth policy:', mdp.policy.function
