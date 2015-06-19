#!/bin/python
# just for testing the large-margin classifier

from SCIRL.SCIRL import clf_large_margin
import numpy as np

# -- make features -- #
A = 7
S = 10
k = 7
# not very reasonable
features = np.random.normal(size=(A, S, k))         # |A| x |S| x k (duh)

# -- define clf -- #
clf = clf_large_margin(features)
# -- make data -- #
N = 1000
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
initial_objective = clf.objective(data)
clf.fit(data)

# --- compare true params --- #
print '\t\t\t\ttrue objective: %.3f' % clf.objective(data, lambd=0.5, params=true_params)
print 'initial objective: %.3f' % initial_objective, '===> fitted objective: %.3f' % clf.objective(data, lambd=0.5)
print '\nParameters:'
print 'True\tFitted'
for i in xrange(k):
    print '%+.3f' % true_params[i],'\t%+.3f' % clf.params[i]
