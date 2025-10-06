---
title: "Reinforcement Learning: Eligibility Traces and TD(lambda)"
date: 2017-11-02 15:18:00 -0200
description: Eligibility Traces in Temporal Difference Methods
tags: reinforcement-learning mdp optimal-control machine-learning
categories: ml reinf-learn
layout: post
author: Alister Machado
---

In the last post of this series, we talked about _temporal difference_ methods.
These allow us to learn online – at the same time we interact with an environment –
and are based on the notion of _bootstrapping_. This means that we use our current
approximation for the value of a state (which might be wrong) to update our estimated value for another state.

Everything goes well as long as all of the approximations get better with time.
This method is called TD(0), and is _biased_, while having reduced _variance_.
A Monte-Carlo estimation method is not biased, but has a lot of variance since it
uses the outcome of a complete episode to perform an update. The variance comes
from the fact that, at each interaction, there's randomness involved in picking
an action – in the case of a stochastic policy – and in the fact that the environment's
dynamics are also random (remember, we have a distribution over the possible next states,
that depends on the current state and the action taken).

One problem with TD(0) is that it uses information from only **one** step to perform
an update. Typically, the action that caused a reward to be seen might have happened
several timesteps in the past. This means that using only the most recent information
can lead to slow convergence.

To solve this problem, we might think that using more than one step to perform an
update is enough. But then the following question rises: how do we pick the number
of steps to use?
Instead of doing that, we'll instead employ a mathematical trick to use _all_ relevant
timesteps, weighted by a factor that reflects the _chance_ that said timestep caused
us to see the reward we're seeing. These methods are called TD(\\(\lambda\\)) methods.

## This can be mathematically heavy, so let's put some context here.

We're _still_ (sorry) trying to learn a way to estimate \\(v_\pi(s)\\). In the last
post we saw the notion of _targets_, which are values towards which we shift our estimate
by a small amount. For example, the TD(0) target is
\\(r_{t+1} + \gamma \hat{v_\pi}(s_{t+1})\\)
and \\(\delta_t = r_{t+1} + \gamma \hat{v_\pi}(s_{t+1}) - \hat{v_\pi}(s_t)\\) is called
the **TD(0) error**.

Let's build up the intuition for TD(\\(\lambda\\)) by expanding the TD(0) error. For this,
we'll talk about the _n-step returns_ of a trajectory. The \\(n = 1\\) return is the TD(0)
target.

$$
n = 1 \rightarrow G_t^{(1)} = r_{t+1} + \gamma \hat{v_\pi}(s_{t+1}) \\
n = 2 \rightarrow G_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 \hat{v_\pi}(s_{t+2}) \\
\vdots \\
\forall n \rightarrow G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n \hat{v_\pi}(s_{t+n})
$$

So, in order to perform _n-step TD-learning_, we can replace the TD(0) target, which
is the 1-step return, with the n-step return! Our update for the state-value function
becomes:

$$
\hat{v_\pi}(s_t) \gets \hat{v_\pi}(s_t) + \alpha (G_t^{(n)} - \hat{v_\pi}(s_t))
$$

## Avoiding the problem...

Picking \\(n\\) can be tough, and most certainly won't generalize to different environments,
so let's find a way to avoid picking \\(n\\) altogether! We'll do this by picking
_all_ different values of \\(n\\) at once. How? Bear with me:

The intuition is to _average_ all of the possible n-step returns into a single return.
If we average these values in a smart way, we can do it super quick and still have
something that intuitively makes sense.

We'll weight the n-step return \\(G_t^{(n)}\\) using a weight that that decays
exponentially with time. This is done by introducing a factor \\(\lambda \in [0, 1]\\) and
weighting the nth return with \\(\lambda^{n-1}\\). Since we want all of these weights to
sum to one (to have a weighted average), we need to normalize them. The normalization constant is easy to derive:

$$
\sum_{n=1}^\infty \lambda^{n-1} = \sum_{n=0}^{\infty} \lambda^n = \frac{1}{1 - \lambda}
$$

Therefore, the normalization constant we're looking for is \\((1-\lambda)\\). This
gives rise to the definition of the \\(\lambda\\)-return:
\\[
G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
\\]


<style type="text/css">

#container {
  height:150px;
 line-height:150px;
}

#container img {
  vertical-align: middle;
  max-height: 50%;
}
img[src*='#center'] {
    width: 40em;
    display: block;
    margin: auto;
}
</style>


![Lambda weighting decay over time](/assets/lambda_return_weight.png#center)
This graph shows the value of \\((1-\lambda) \lambda^{n}\\) for different values
of \\(n\\). We can see how different values of lambda affect the initial value of a return
and the way this value decays over time. Bigger values of lambda lead to slower decay
(information from the past is given a non-negligible importance).


## ... by introducing a new problem

Now we can use the lambda-return as target for our state-value function updates.
But wait a second... the lambda-return involves _all_ of the possible n-step returns
and, as a consequence, involves information from _every_ time step of our trajectory.
We're back to episodic updates: we need to wait for a trajectory to be completed,
and then we will be able to calculate every possible n-step return and combine them
using the lambda-weighted average. (for completeness, the flavor of TD(lambda) we saw
above is called the _forward view_)

So far it seems like we've accomplished very little, if anything at all. But never
fear! We did not introduce this lambda weighting scheme just because it's pretty.
In fact, it will allow us to use a trick that enables online updates while being
virtually equivalent to the intuition we derived above. To do this, we'll use
something called an _eligibility trace_.

### Eligibility traces and who the hell is to blame for this reward

We're now going to switch to the _backward view_ of TD(lambda). The name gives us
a hint on how it works: instead of waiting for what is going to happen _next_ to be
able to perform an update, we will remember what happened in the _past_ and use current
information to update the state-values for every state we've seen so far.

To do that, we'll employ _eligibility traces_, a nifty idea that allows us to do just
that. An eligibility trace is defined as:

$$
e_0(s) \gets 0 \quad \forall s \in \mathcal{S} \\
e_{t}(s) \gets \lambda \gamma e_{t-1}(s) + \mathbf{I}(S_t = s)\quad \forall s \in \mathcal{S}
$$

where \\(\mathbf{I}\\) is the indicator function, which is equals to 1 when the
condition inside it is true and 0 otherwise.

This eligibility trace will be used as a scaling factor for the TD error. So, for
example, if we're interacting with the environment and see a reward \\(r_{t+1}\\),
our update to the state-value function will be:

$$
\delta_t = r_{t+1} + \gamma \hat{v_\pi}(s_{t+1}) - \hat{v_\pi}(s_t) \\
\hat{v_\pi}(s) \gets \hat{v_\pi}(s) + \alpha \delta_t e_t(s) \quad \forall s \in \mathcal{S}
$$

Ok, there's a lot to explain here:

* The eligibility traces combine two things: both how _frequent_ and how _recent_
a state is. This implies that we must update our state-value function estimates for
**every state** at each time step. States that are not visited or haven't been visited
in a while will have an eligibility value equals or close to zero (resp.), which
means that our estimate for those states won't change (much) – and this is exactly what we want.
* We now use the eligibility value as a scaling factor for a TD(0) error. The trick here
is hidden in the fact that every state is updated at once. In other words: we _propagate_
current error information into the states we visited in the _past_. This allows us to
combine the n-step returns in an online fashion.
* This formulation of the eligibility trace is called an _accumulating eligibility trace_.
An alternative to it is to use a _replacing eligibility trace_ that, instead of summing 1
to the eligibility when visiting a state, _resets_ the eligibility back to 1.

## How does this look in code?

Using a vector to keep the eligibility of each state, and taking advantage of numpy's vectorized operations, the code is quite simple. We're assuming there is already a variable that represents the step length (alpha).

Algorithm 1: TD(\\(\lambda\\)) - estimating state-value function with eligibility traces.
```python
import numpy as np

state_values = np.zeros(n_states) # initial guess = 0 value
eligibility = np.zeros(n_states)

lamb = 0.95 # the lambda weighting factor
state = env.reset() # start the environment, get the initial state
# Run the algorithm for some episodes
for t in range(n_steps):
  # act according to policy
  action = policy(state)
  new_state, reward, done = env.step(action)
  # Update eligibilities
  eligibility *= lamb * gamma
  eligibility[state] += 1.0

  # get the td-error and update every state's value estimate
  # according to their eligibilities.
  td_error = reward + gamma * state_values[new_state] - state_values[state]
  state_values = state_values + alpha * td_error * eligibility

  if done:
    state = env.reset()
  else:
    state = new_state
```

## Wrapping up

I could go on and on and prove why the backward view of TD(\\(\lambda\\)) is equivalent
to the forward view – and maybe I'll do it in the future – but I don't want to
overextend this post. The key takeaways are:

* There are ways to use information from _every_ time step to perform state-value
updates that do not necessarily involve waiting until the end of an episode.
* One such way is to use the backward view of TD(\\(\lambda\\)), which involves
eligibility traces.
* Eligibility traces are ways to keep a history of what happened in the past and
how the states we've visited affected the reward we're seeing. It allows us to update
multiple state-value function estimates at once, in a way that is weighted by recency
and frequency.
* There is a different type of eligibility trace called a replacing eligibility trace,
which does not _sum_ one, but instead _sets_ the eligibility back to one when a state
is visited.
* This is _still_ a Temporal Difference method. It is still _biased_ because of the bootstrapping. It should **intuitively** have less variance than a Monte Carlo method because of the averaging of low-variance information, but I have no theory to back up this claim.

## References

* Again, David Silver's class on TD(\\(\lambda\\)) ([link here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf))
