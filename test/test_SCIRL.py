#!/bin/python
# Run SCIRL on a trajectory generated from an MDP.
# Load the true MDP, compare the SCIRL-derived reward function.

from SCIRL import SCIRL
from MDP import mdp
import numpy as np

traj_path = './data/test_N100_traj.npy'
basis_path = './data/test_basis_S4k3.npy'
mdp_path = './data/test_N100_MDP.npy'

# --- get data --- #
trajectory, basis_functions = SCIRL.get_training_data(traj_path, basis_path)

# --- run SCIRL --- #
SCIRL_reward = SCIRL.SCIRL(trajectory, basis_functions)

# --- load true MDP --- #
mdp = np.load(mdp_path).item()
MDP_reward = mdp.rewards

# --- compare --- #
print 'SCIRL rewards:', SCIRL_reward
print 'MDP rewards:', MDP_reward
