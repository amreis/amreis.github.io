---
title: "Reinforcement Learning: Evaluating Behavior"
layout: post
date: 2017-05-28 17:33:20 -0700
categories: ml reinf-learn
tags: policy-evaluation monte-carlo
---

This is the second post of a series I'm writing on Reinforcement Learning, giving an overview on the subject and trying to stay away from strong theory. This way, hopefully, you can better understand concepts as explained in Sutton & Barto's book, for example.

In this post, I'll be talking about something called **policy evaluation**. In the context of Reinforcement Learning, evaluating a policy means that we want to know how good a certain behavior is in a given environment. But first, let's remember what we got ourselves into:

## Let's recap!

Reinforcement learning is the field that studies how to learn **behavior** while only receiving (probably delayed) **reward** information from the environment. This problem is very different from traditional Machine Learning scenarios.

In the last post we saw some examples of problems that can be solved using RL and some of the theory that is behind the setting. We also talked about the **credit assignment** problem, which consists of determining which actions caused us to receive a reward from the environment.

Before we dive into policy evaluation, we need to build some concepts concerning our notion of "performing well" in an environment. Namely, we need a more formal definition of the Markov Decision Process, of a policy function and also of the goal our agent will be pursuing.

## Defining a concrete goal

Throughout our agent's interaction with the environment, it will see a sequence of **states**, take **actions** and receive **rewards**. If we have an environment with the notion of _termination_, every time we reach a terminal state, we say that an _episode_ has ended. Our **trajectory** in this episode is usually represented as \\(\tau = \left\langle s_0, a_0, r_1, s_1, a_1, r_2, \dots, a_{T-1}, r_T \right\rangle\\).

<style type="text/css">

#container {
  height:300px;
 line-height:300px;
}

#container img {
  vertical-align: middle;
  max-height: 100%;
}
img[src*='#center'] {
    display: block;
    margin: auto;
}
</style>


![Agent-environment interaction](/assets/state-env-interaction.svg#center)


The fact that we index the rewards on the _next_ time step represents that we have already interacted with the environment, so we consider that the reward we see is already a part of the "next step".

Looking at our trajectory, we can choose to maximize different goals. Two central examples of those are:

* Average return: \\(\frac{1}{T} \sum_{i=1}^{T} R_i\\)
* Discounted return: \\(G_t = \sum_{i=0}^T \gamma^{i} R_{t+i+1}\\), where \\(\gamma \in [0, 1]\\)

We are going to focus on the _second_ one, which values immediate reward more strongly than late rewards. This can be related to the concepts in finance, for example, where immediate gain is often more valuable than average gain. Also, the fact that we have this factor of \\(\gamma\\) multiplying the instantaneous reward lets us treat infinite-horizon problems as if they were finite. That happens because
\\[
\sum_{i=0}^{\infty} \gamma^i = \frac{1}{1-\gamma} \quad ,\gamma \in [0,1]
\\]

So, we will say that our agent behaves well if its policy maximizes the **expected discounted return** \\(G_t\\)

## Some more details...

### Markov Decision Processes, revisited

To be able to write some of the equations I need to show you, we need to better specify two things in our MDP: how we **change states** and how we **get rewarded**.

An MDP possesses two devices to do just that:

* A **transition probability** function \\(\mathcal{P}_{a}^{s,s'} = \mathbb{P}(S'=s' | S=s, A=a)\\)
. In words, it's the probability of reaching state \\(s'\\) after taking action \\(a\\) in state \\(s\\).
* A **reward** function \\(\mathcal{R}_{a}^{s}\\) which is the reward we get for taking action \\(a\\) in state \\(s\\).

### Policies

A policy is a function (deterministic or not) that tells us the action to take (or a distribution over actions) for each state. We'll often see something like:
\\[
a \sim \pi(\cdot | s) \quad ; \quad \pi(a|s) = \mathbb{P}(A_t = a\, | \,S_t = s)
\\]
Which means that the action we pick at state \\(s\\) is drawn from a probability distribution determined by that state. If the policy is deterministic, we may write \\(a = \pi(s)\\). Also, it is supposed to be stationary (i.e. it doesn't change over time).

It makes sense, then, to relate rewards to states and actions. This is done through two functions: the _state-value_ function and the _action-value_ function. The only one that matters to us now is the _state-value_ function, the _action-value_ will show its worth when we attack the planning problem.

### The State-Value function \\(v_\pi\\)

The aim of the state value function is to tell us what is our **expected discounted return** starting from a given state \\(s\\) and following the policy \\(\pi\\) from there onwards. And that's, in essence, all it is: an expectation (a weighted average) over what can happen given that we follow the behavior defined by \\(\pi\\) when interacting with the environment. In math:

$$
v_\pi(s) = \mathbb{E}_\pi \left[ G_t \,| S_t = s \right] =
\mathbb{E}_\pi \left[ R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} +\dots | S_t = s \right]
$$

This can also be expressed recursively as:

$$
v_\pi(s) = \mathbb{E}_\pi \left[ R_t + \gamma (R_{t+1} + \gamma R_{t+2} + \dots) | S_t = s \right] = \mathbb{E}_\pi \left[ R_t + \gamma v_\pi(S_{t+1})\right | S_t = s] \\
= \sum_{a} \pi(a | s) \left(\mathcal{R}_a^s + \gamma \sum_{s'} \mathcal{P}_a^{s, s'} v_\pi(s') \right)
$$

This tutorial should not confuse you with math. What this equation is saying is: "the overall _value_ of a state is a weighted average of the _immediate reward_ we'll get from taking each action, _plus_ the discounted _average value_ of the states we might end up in". Take a deep breath. Read it again.

It's basically just a big average. Considering every possible outcome (every action, every next state) and using their probabilities to weight their values. It's **recursive** (defined in terms of itself), and while that might seem confusing, it's actually very helpful (we'll see why in the next post).

<!--

### The Action-Value function \\(q_\pi\\)

We can also express the expected return \\(G_t\\) in terms of both a _state_ and an _action_, which gives rise to the action-value function. It is very very similar to the state-value function:

$$
q_\pi(s,a) = \mathbb{E}_\pi \left[G_t | S_t = s, A_t = a \right]
$$

It also admits a recursive form, which is:

$$
q_\pi(s, a) = \mathbb{E}_\pi \left[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots | S_t = s, A_t = a \right] = \\
\mathbb{E}_\pi \left[ R_t + \gamma q_\pi (S_{t+1}, A_{t+1}) | S_t=s, A_t=a\right]
$$

-->
## How well are we behaving?

Remember: the problem at hand is estimating how well a given policy makes us behave in an environment. One way of looking at this is to say that we want to know the _state-value function_ for a given policy. In other words, for any given state, we want to know the expected return we'll have if we follow a policy \\(\pi\\).

While I could talk (and you should probably read) about methods that are used when we **know** the underlying MDP to our environment, Reinforcement Learning typically occurs in scenarios where we _don't_. Let's try to build an intuition on how to do it using _only_ the agent's experience.

### Monte-Carlo Method for State-Value Function

Monte Carlo Methods have one central idea: sampling. When we don't know a quantity, we try to sample it and estimate it using an average, for example. In our case, we want to know the state-value function for a policy. That means we should sample this function many times and approximate its value using its average (since the function is an expectation).

But what does it mean to sample the state-value function? In fact, what we will do is sample many trajectories in an environment, calculate the reward we get from each state and accumulate these values using an average.

As for many things in Computer Science, this is a good idea to start with, but what we implement in practice will be different. Instead of using an average, which in theory would keep all the information from the past, we will actually use something a bit different. Let me put some code here:

Algorithm 1: Every-Visit Monte Carlo State-Value Approximation
```python
state_values = [0.0 for _ in range(n_states)] # initial guess = 0 value

# We run this algorithm for a given number of episodes
for episode in range(number_episodes):
  # we get the trajectory from the episode. we only care
  # about the states we've seen and the rewards we've got.
  # `policy` is a function or a table that tells us how to
  # behave in each state, not shown here.
  states, rewards = run_episode(policy)
  for i, state in enumerate(states):
    # calculate G_t using the gamma discounts
    value = get_discounted_return(rewards[i:])
    # shift our current estimate a little bit towards
    # the value we got from the last simulation
    state_values[state] += alpha * (value - state_values[state])
```


This should be in line with what you were expecting from code that does what I described in the post so far. Except for the last line. That's where our average should have been, but it's not. While we can calculate an average incrementally, there comes a point where the next value we're adding to it has almost no influence (think of the case where we have 1 million values and add another one to an average).

Using this `alpha` value, which is between zero and one, we can think of that last line in the following way:

We have a sample from the value function in the `value` variable. This can be considered as a "noisy truth". What we do is _shift_ our current prediction by a small amount `alpha` in the direction of the "true" value (the one we sampled). That way, if we gradually reduce `alpha`, our estimate will converge to the true state-value function. Also, later when we will be doing _planning_, having this type of "shifting by alpha" behavior will allow us to forget the distant past, which means we can adapt to new behavior we find while exploring the environment.

## Wrap-up

This was a _long_ post (I apologize) with some math (I apologize some more, but I swear it was needed) and some code. No apologies for the code. We covered just a part of policy evaluation techniques, and the next post will be dedicated to what we call **temporal difference** methods (namely, TD(0) and TD\\((\lambda)\\)).

### Important concepts

* Transition probability function, reward function: express how the environment behaves, including its uncertainty, and how we are rewarded.
* Discounted return: our agent's goal is to maximize the expected value of this variable. It can give higher value to immediate rewards.
* The state-value function: a policy's expected discounted return, given a starting state.
* Monte Carlo algorithm: sampling trajectories and refining estimates.

### Reducing alpha

In order for this algorithm to converge, we need to _reduce_ the value of alpha. We can't reduce it too fast, because we would fall short of the correct value function. We can't reduce it too slow, because it may make us diverge. How to we pick alpha, you ask? Well, _in theory_, any sequence of alphas that follow these rules will work:

\\[
\sum_{t=0}^\infty \alpha_t = \infty \, ; \, \sum_{t=0}^\infty \alpha_t^2 < \infty
\\]

One sequence of alphas that respects these conditions is, for example, \\(\alpha_t = 1/t\\).

## References

* Lectures 2-3 of David Silver's course ([here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf) and [here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf))

<!--

* Sutton & Barto's book (link, chapter)
* THE survey paper

-->
