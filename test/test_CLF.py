#!/bin/python
# just for testing the large-margin classifier

from SCIRL.SCIRL import clf_large_margin
import numpy as np
from copy import deepcopy

# -- make features -- #
A = 7
S = 10
k = 3
true_k = 6
# not very reasonable
true_features = np.random.normal(size=(A, S, true_k))         # |A| x |S| x k (duh)
true_params = np.random.normal(size=true_k)
true_score_fn = np.dot(true_features, true_params)
# give the classifier a subset of the real parameters (-> features)
k_picks = np.array(sorted(list(np.random.choice(true_k, k, replace=False))))
features = true_features[:, :, k_picks]

# -- define clf -- #
clf = clf_large_margin(features)
true_clf = clf_large_margin(true_features)
true_clf.params = true_params

# -- make data -- #
N = 1000
states = np.random.randint(low=0, high=S, size=N)
# predict labels using true score function
scores = true_score_fn[:, states]                   # |A| x N
labels = np.argmax(scores, axis=0)
# combine
data = np.vstack((states, labels)).transpose()

# -- fit ! -- #
initial_objective = clf.objective(data)
clf.fit(data, verbose=1)

# --- describe results --- #
print 'RESULTS:'
print 'True accuracy:  ', true_clf.accuracy(data)
print 'Fitted accuracy:', clf.accuracy(data)
print '\t\t\t\ttrue objective: %.3f' % true_clf.objective(data)
print 'initial objective: %.3f' % initial_objective, '===> fitted objective: %.3f' % clf.objective(data)
print '\nParameters:'
print 'True\tFitted'
for i in xrange(true_k):
    if i in k_picks:
        print '%+.3f' % true_params[i],'\t%+.3f' % clf.params[list(k_picks).index(i)]
    else:
        print '%+.3f' % true_params[i],'\t-'
