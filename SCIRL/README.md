### Structured Classification-based Inverse Reinforcement Learning
My attempt at implementing the SCIRL algorithm as introduced in  

[Klein, Edouard, et al. "Inverse reinforcement learning through structured
classification." *Advances in Neural Information Processing Systems*. **2012**.](http://papers.nips.cc/paper/4551-inverse)

Given a training set of state, action (from expert deterministic policy on that
state) pairs, the algorithm requires two primary components:
- an estimate of the **expert feature expectations**
- a linearly parametrised score function-based multi-class **classification
  algorithm**

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

So this leaves two questions:

#### 1. Getting expert feature expectations

This is tricky because one would like to know the underlying dynamics of the
system (state-action-state transition functions: a *model* of the system) to
calculate the expected value of anything. This isn't always available. If it
is, the problem is a standard policy-evaluation problem (since the feature
expectations are the action-value functions for special choices of reward
function).

If you *don't* have a model of the system handy, things are harder. You can use
temporal difference learning algorithms: they mention least-squares temporal 
differences, but actually use a heuristic method: (eq 5)

For state-action pairs *observed* in the available expert trajectories, one
empirically calculates the feature expectation (with the discount factor).

For state-action pairs *not* observed, one assumes that the non-expert will
make the correct choice on their next move, (*'applying a non-expert action
just delays the effect of the expert action'*), so the feature expectation is
multiplied by the discount factor.

#### 2. Multi-class classification algorithm

They use the structured large margin approach: ([Taskar, Ben, et al. "Learning
structured prediction models: A large margin approach." *Proceedings of the 22nd
international conference on Machine learning*. ACM,
**2005**.](https://dl.acm.org/citation.cfm?id=1102464))

This basically means choosing an appropriate loss function (section 4.2 in the
paper) and minimizing using subgradient descent. The expert feature
expectations are involved, here.
