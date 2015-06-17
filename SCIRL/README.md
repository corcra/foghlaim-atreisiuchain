### Structured Classification-based Inverse Reinforcement Learning
My attempt at implementing the SCIRL algorithm as introduced in  

[Klein, Edouard, et al. "Inverse reinforcement learning through structured
classification." *Advances in Neural Information Processing Systems*. **2012**.](http://papers.nips.cc/paper/4551-inverse)

Given a training set of state, action (from expert deterministic policy on that
state) pairs, the algorithm requires two primary components:
- an estimate of the expert feature expectations
- a linearly parametrised score function-based multi-class classification
  algorithm

The reward function is assumed to be a linear combination of some basis 
functions on the state space (these must be given). Feature expectations are 
then essentially action-value functions, assuming the reward function is given
by only one of these basis functions. Then, the full action-value functions are
linear combinations of these feature expectations.

Structured classification comes into it as follows: consider a score function
over input/label pairs, which is a linear function of some basis functions
(assumed given, not the same as those described in previous paragraph). We 
learn the coefficients of this linear combination to minimise the 
classification error on a training dataset. Classification is achieved by 
selecting the label (given an input) which maximises the score function.

This is related to IRL as follows: By defining the basis functions as the 
*expert feature expectations* (functions of both states and labels, remember),
the score function (linear function of the basis functions) is equivalent to
the action-value function (with some parametrisation). A greedy policy will 
select the action maximizing the value of the action-value function in a given
state, so we see that the classifier is learning to imitate the optimal policy.
Thus, the parameters learned by minimizing the classification error should
recapture the action-value function of the expert. Knowing then the parameters
of the action-value function, we obtain the reward function because it shares
these parameters.

*This all would have been a lot easier to describe if I could use LaTeX...*
