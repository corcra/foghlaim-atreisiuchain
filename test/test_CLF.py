#!/bin/python
# just for testing the large-margin classifier

from SCIRL.SCIRL import clf_large_margin
import numpy as np

# -- make features -- #
A = 3
S = 10
k = 6
# not very reasonable
features = np.random.normal(size=(A, S, k))

# -- define clf -- #
clf = clf_large_margin(features)

# -- make data -- #
N = 1000
states = np.random.randint(low=0, high=S, size=N)
# trivial sort of classification
labels = states % 2
#labels = (states + np.random.binomial(S, p=0.05, size=N)) % A
data = np.vstack((states, labels)).transpose()

# -- fit ! -- #
clf.fit(data)
