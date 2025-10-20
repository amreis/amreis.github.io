---
title: "On Optimal Value Functions"
layout: post
date: 2017-12-21 18:22:20 -0200
categories: ml reinf-learn
tags: reinforcement-learning value-function bellman optimality
description: "Definitions and devices used to calculate optimal value functions"
---

For the last posts, we have been talking a lot about ways to determine the _value_
of a policy (in other words, the expected return that an agent will experience when
acting according to that policy). But what good is that? Sure, it's nice to know
_if_ a policy is bad or not – which we call the _prediction_ problem –, but what
about actually _finding_ good policies? This is referred to as the **control** problem,
and to start talking about it, we need to define what it means to be optimal and how
we can discover the optimal policy for an environment.

We will focus on the case where we know both the _dynamics_ and the _rewards_ of
our environment. This makes the analysis easier and enables us to reach some interesting
results.

## What is "optimal"?

Let's begin by formalizing the best expected return possible that we can achieve
in an environment. This is nothing more than the optimal _state-value function_:

$$
v_*(s) = \max_\pi v_\pi(s), \forall s
$$

Another concept that will be useful in our analysis is that of the optimal
_action-value function_. This is analogous to the state-value function, but instead
we also pick the action, not only the state. The action-value function is defined
as:

$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]\\

= \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a]
$$

And the optimal action-value function is:

$$
q_*(s, a) = \max_\pi q_\pi(s, a)
$$

There are also special forms of the Bellman equations for the optimal value functions (and only for the _optimal_ ones),
called the **Bellman Optimality Equations**. These are:

$$
v_*(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t = a]\\
= \max_a q_*(s, a)\\
= \max_a \sum_{s'} p(s' | s, a) [r + \gamma v_*(s')]
$$

and

$$
q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') | S_t = s, A_t = a]\\
= \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1})]\\
= \sum_{s'} p(s'|s,a)[r + \gamma v_*(s')]
$$

## How do we get there?

### First step: Exact Policy Evaluation

Remember the definition of state-value function for a given policy:

**Definition 1: State-Value function**

$$
v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]\\
= \sum_{a} \pi(a | s) \sum_{s'} p(s'|s, a) [r + \gamma v_\pi(s')]
$$

This is a fact. The true value function for a policy respects this equation, the left
side can be rewritten as the right side. This is important, very important! If we know
the full dynamics of the system, this definition of state value function is a linear system!
It has as many equations as the number of states, and just as many variables.
In practice, we'll solve this by turning the above definition into an update rule,
or an operator if you're more comfortable with that:

$$
v_\pi^{t+1}(s) = \sum_{a} \pi(a | s) \sum_{s'} p(s' | s,a) [r + \gamma v_\pi^{t}(s')]
$$

Or, using operator notation, we have:

$$
T^\pi v(s) = \sum_{a} \pi(a|s) \sum_{s'} p(s'|s,a) [r + \gamma v(s')]\\

v_\pi^{t+1} = T^\pi v_\pi^{t}
$$

The true value function is then a **fixed point** of the operator (and of the
equation) defined above, because it's the only case in which we'll have
equality of \\(v_\pi^{t+1}\\) and \\(v_\pi^t\\).

Bottom line: we have a very simple way to do _policy evaluation_ under known dynamics.
The methods explored in this blog before this post (TD(0), Monte Carlo...) are ways
of doing this without knowing the full dynamics of the environment.

### Second step: Policy Improvement

Knowing how to evaluate a policy allows us to make it _better_. We can do that
because of a result called the _policy improvement theorem_.

Suppose we have a new policy \\(\pi'\\) such that \\(\pi'(s) \neq \pi(s)\\). Now
assume that we choose _a single action_ according to \\(\pi'\\) and follow \\(\pi\\)
for the rest of the episode. If doing this leads to a better outcome, then we might expect
that the new behavior upon seeing state \\(s\\) is one that we should take for _every_
occurrence of \\(s\\).

Formally, if the new policy \\(\pi'\\) satisfies \\(q_\pi(s, \pi'(s)) \geq v_\pi(s) \forall s\\),
then we have that \\(v_{\pi'}(s) \geq v_\pi(s) \forall s\\), and policy \\(\pi'\\) is an _improvement_
over \\(\pi\\).

Now suppose we build a new policy in the following way:

$$
\pi'(s) = \text{argmax}_a q_\pi(s, a)\\
= \text{argmax}_a \sum_{s'} p(s'|s,a) [r + \gamma v_\pi(s')]
$$

This is called a _greedy_ policy. It aims to maximize immediate expected gain. In
decent wording, this means it looks ahead one step into the future and picks the
action that gives it the biggest _expected_ (weighted average) return.

Notice that a greedy policy \\(\pi\\) can be looked at from two perspectives:
 it is a function that maps a state to an action \\(\pi(s) = a\\) but also a
 probability distribution over actions that focuses all of its mass on a single action
 \\(\pi(a|s) = 1\\), if \\(a\\) is the greedy action, and \\(0\\) otherwise.

The new policy satisfies the policy improvement theorem by definition, and so is
an improvement on the old policy. When the value of the new policy is not an improvement
for at least one state, _we are guaranteed to have found the optimal deterministic policy_.
This might give you a hint on what we're about to do to reach the optimal policy.

### Third step: Value Iteration

Let's not focus on the _name_ of the technique now, but instead look at the bigger picture.
We have a way of _precisely evaluating_ the value of a policy. And given the value
of a policy, we have a way of _improving_ the policy. Okay then, let's just do that
over and over again: evaluate, improve, evaluate, improve, evaluate, improve...

Listing 1: Policy Iteration Algorithm
```python
policy_stable = False
while not policy_stable:
  old_policy = copy(policy)
  state_values = evaluate(policy, environment_dynamics)
  new_policy = greedy_policy_with_respect_to(state_values)
  if new_policy == old_policy:
    policy_stable = True
  policy = new_policy
```

The evaluation step (line 4) follows
what is described in "First step", iteratively applying the \\(T^\pi\\) operator
to the value function until it converges.

I wanted to keep the code at a very, very high level there, because we won't do
that in practice, but the idea is still important.

You might notice that I (well, the people who created this, in fact) called this
**Policy Iteration**, not Value Iteration as is the name of the section.
That's because we can improve it, and the improvement has
a different, but equally cute name. _(hint hint: it's Value Iteration)_

Notice that we need to evaluate the policy _until convergence_ of the state values,
_only to throw it away one line later_. That seems like a waste of time, and
we don't like wasting time! We can do better. Much better. In fact, it is known
that you only need to apply the \\(T^\pi\\) operator _once_, improving the state
value estimates, but without caring to let them fully converge. We can then derive
a new policy from these non-converged values by acting greedily towards them.

Being formal, what we'll do is:

$$
\hat{v}_0(s) = 0\\
\pi^0(s) = \text{argmax}_a \sum_{s'} p(s'|s,a) [r + \gamma \hat{v}_0(s')]\\
\hat{v}_{1}(s) = \sum_{a} \pi^0(a | s) \sum_{s'} p(s' | s,a) [r + \gamma \hat{v}_0(s')]\\
\pi^1(s) = \text{argmax}_a \sum_{s'} p(s'|s, a) [r + \gamma \hat{v}_1(s')]\\
\hat{v}_2(s) = \sum_{a} \pi^1(a | s) \sum_{s'} p(s'|s,a)[r+\gamma \hat{v}_1(s')]\\
$$

But here's the thing... the generated policies are all _greedy_.
With this in hand, we can rewrite the equation that
determines \\(\hat{v}_{k}(s)\\). Let's start:

$$
\hat{v}_{1}(s) = \sum_{a} \pi^0(a | s) \sum_{s'} p(s' | s,a) [r + \gamma \hat{v}_0(s')]\\
$$

But since the policy is greedy, this collapses to only one action (because the others
have probability zero). And not only that,
since the _argmax_ is taken over the expression that appears in the summand above, we have
that the only action that matters is the one that _maximizes_ the expression. Therefore,
the new value for state \\(s\\) is nothing but the \\(\max\\) value among the actions,
with respect to the previous approximation of the value function.

$$
= \max_a \sum_{s'} p(s' | s, a) [r + \gamma \hat{v}_0(s')]
$$

Now we're left with a simple and elegant iteration scheme for the value function
(hence the name, **Value Iteration**):

$$
\hat{v}_{k+1} (s) = \max_a \sum_{s'} p(s' | s, a) [r + \gamma \hat{v}_k(s')]
$$

Using operator notation, again, we have a form of the Bellman operator, in this
case called \\(T^*\\) (note that this is nothing but transforming the Bellman optimality equation
described in the beginning of this text into an update rule):

$$
(T^* v)(s) = \max_a \sum_{s'} p(s' | s, a) [r + \gamma v(s')]\\
v_{k+1}(s) = T^* v_k(s)
$$

Applying this Bellman operator to our value function estimate over and over again
will _inevitably_ lead us to the true, optimal value function for this environment.
Since we know the dynamics of the system, we can then derive the optimal policy from
this value function.

And this is how we solve MDPs for which we completely know the dynamics. Now for some extras!

## Appendices

### Exact general value function using Linear Algebra

Assume we construct the matrices/vectors as follows:

$$
\mathcal{R} = \begin{bmatrix}
R(s_1) \\
R(s_2) \\
\vdots \\
R(s_n)
\end{bmatrix}\\

\mathcal{P}_{i, j} = \sum_{a} \pi(a |s_i) p(s_j | s_i, a)\\

$$

Then we can solve exactly for a given policy's value function starting from a
Bellman equation in matrix notation (note that this is nothing but a rewriting
of the \\(T^\pi\\) operator):

$$
V_\pi = \mathcal{R} + \gamma \mathcal{P} V_\pi\\
V_\pi - \gamma \mathcal{P} V_\pi = \mathcal{R}\\
(I - \gamma \mathcal{P}) V_\pi = \mathcal{R}\\
V_\pi = (I - \gamma \mathcal{P})^{-1} \mathcal{R}
$$

Which is a linear system of \\(n\\) equations in \\(n\\) variables. Plug in a solver,
and be happy. In fact we usually prefer to use iterative methods like repeated
application of the \\(T^\pi\\) operator insted of going through the hassle of
inverting a matrix.

### Convergence speed of the Bellman operator

It can be shown that the Bellman operator \\(T^*\\) is a _contraction mapping_ in the
space of value functions. What this means is:

$$
\lVert T^* V_1 - T^* V_2 \rVert \leq \gamma \lVert V_1 - V_2 \rVert
$$

But we also have that applying the \\( T^* \\) operator to the _optimal_ value function
does not change it (it's a fixed point of the operator). Therefore \\( (T^* )^{n} {V}^* = T^* V^* = V^* \\).

So, applying the Bellman operator _once_ to any value function estimate _reduces_ its distance
to the optimal value function by a multiplicative factor of \\(\gamma\\). Repeated application
of this operator gives us:

$$
\lVert (T^*)^n V - V^* \rVert = \lVert T^* ((T^*)^{n-1} V) - T^* ((T^*)^{n-1} V^*) \rVert \\
\leq \gamma^1 \lVert (T^*)^{n-1} V - (T^*)^{n-1} V^* \rVert \leq \dots \\
\leq \gamma^n \lVert V - V^* \rVert
$$

So the error between our initial estimate and the optimal value function decreases
exponentially fast with successive applications of the Bellman operator.

### Deriving the optimal policy

From the optimal state-values, under known dynamics, the process of coming up with
the corresponding optimal policy is immediate, but I'll mention it here for completeness.
It is formed in the following way:

$$
\pi(s) = \text{argmax}_a \sum_{s'} p(s' | s, a) [r + \gamma V^*(s')]
$$

## Wrap-up

We covered a lot of ground on a very important subject in this post. Here are the
main highlights:

* The notion of optimal policies and optimal state- and action-value functions;
* Exact policy evaluation by the successive application of a Bellman operator;
* Policy improvement by acting greedily with respect to a value function;
* Interleaving these two operations to reach the optimal value function (and as a consequence, the optimal policy);
* Improving that algorithm so that we can have the optimal policy faster, which
in turn gives us a very nifty update rule for the state-value estimates;
* And two asides on the speed of convergence of this technique, and on a different perspective on the problem.

## References

* David Silver's course ([here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf))
* The RL Book (for which we now have the full draft!): [here](http://incompleteideas.net/book/the-book-2nd.html), Chapters 3-4.
