#!/bin/python
# just for testing the large-margin classifier

from SCIRL.SCIRL import clf_large_margin
import numpy as np

# -- make features -- #
A = 3
S = 10
k = 7
# not very reasonable
features = np.random.normal(size=(A, S, k))         # |A| x |S| x k (duh)

# -- define clf -- #
clf = clf_large_margin(features)

# -- make data -- #
N = 100000
states = np.random.randint(low=0, high=S, size=N)
# fake some 'true' parameters
true_params = np.random.normal(size=k)
true_score_fn = np.dot(features, true_params)       # |A| x |S|
# use to predict labels
scores = true_score_fn[:, states]                   # |A| x N
labels = np.argmax(scores, axis=0)
# combine
data = np.vstack((states, labels)).transpose()

# -- fit ! -- #
clf.fit(data)

# --- compare true params --- #
print 'True objective:', clf.objective(data, lambd=0.5, params=true_params)
print 'Fitted objective:', clf.objective(data, lambd=0.5)
print 'True parameters:', true_params
print 'Fitted parameters:', clf.params
