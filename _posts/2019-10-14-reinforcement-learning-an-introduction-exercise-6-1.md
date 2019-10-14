---
title: "Reinforcement Learning: An Introduction – Exercise 6.1"
date: 2019-10-14 9:00:00 -0300
description: Resolution of the exercise 6.1 in Sutton and Barto's book, 2nd edition
tags: reinforcement-learning mdp machine-learning exercises rlbook
categories: ml reinf-learn
layout: post
---

It's been a while since I posted here... in fact, so long I'm not sure I even know my way around this thing anymore.

Regardless, I've been going through the second edition of Richard Sutton and Andrew Barto's book "Reinforcement Learning: An Introduction" while trying my best to solve every exercise. One that had me taking a particularly long time was Exercise 6.1, in page 121. This is in Chapter 6: Temporal-Difference Learning (by far the most exciting Chapter in the book until now).

While trying to come up with ideas to solve the exercise, I searched around the web but found nothing that could give me a clue. So here, this is my attempt at sharing my solution and the rationale for it.

## First of all, _notation_!

This chapter talks about TD-Learning, so a few notions need to be established:

$$
G_t \dot{=} R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ... + \gamma^{T-t-1} R_T \\
= R_{t+1} + \gamma G_{t+1}
$$

 is the return (weighted sum of rewards) observed after time \\(t\\). The notation here implicitly assumes that we're dealing with _episodic_ environments.

\\(V_t(S_t)\\) is the current approximation to a given state's value. In other words, \\(V_t\\) is our current approximation to the _value function_.

And last but not least:

$$
\delta_t \dot{=} R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t)
$$

is what's called the _TD error_ at time t. In the book's definition, \\(V\\) shows up _without_ the index \\(t\\). That's because up until page 121, we're asked to work as if all Value Function updates are performed _after_ the end of the episode. Since exercise 6.1 is exactly about imagining what happens to the error between the Return \\(G_t\\) and its difference to \\(V_t(S_t)\\) _as it changes throughout the episode_, my definition already introduces the index.

You can see that this definition is correct (or at least that it _makes sense_) since at the time that we're calculating \\(\delta_t\\), we've only just performed action \\(A_t\\) in state \\(S_t\\), being rewarded with \\(R_{t+1}\\) and observing the next state, \\(S_{t+1}\\). That means that we haven't updated our value estimates yet, which means the error involves \\(V_t\\), not \\(V_{t-1}\\) or \\(V_{t+1}\\).

## What's in the book:

In the book, we see that the difference between the return \\(G_t\\) and \\(V({S_t})\\) can be neatly written as a sum of TD-errors, like this:

$$
G_t - V(S_t) = R_{t+1} + \gamma G_{t+1} - V(S_t) + (\gamma V(S_{t+1}) - \gamma V(S_{t+1})) \\
= R_{t+1} + \gamma V(S_{t+1}) - V(S_t) + \gamma G_{t+1} - \gamma V(S_{t+1}) \\
= \delta_t + \gamma (G_{t+1} - V(S_{t+1})) \\
= \delta_t + \gamma \delta_{t+1} + \gamma^2 (G_{t+2} - \gamma V(S_{t+2})) \\
= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \cdots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T - t}(G_T - V(S_T))
$$

Now, we need to pause. This is so we can understand exactly what \\(G_T\\) and \\(V(S_T)\\) are equal to. Let's start with the return:

Remember the definition of \\(G_t\\) in terms of all the rewards that come after it:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T\\
= \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}
$$

We can apply the same formula for \\(G_T\\):

$$
G_T = \sum_{k=T}^{T-1} \gamma^{k-T} R_{k+1}
$$

But \\(T > T - 1\\), so this is _an empty sum_, which means \\(G_T = 0\\).

As for \\(V(S_T)\\), remember that the value of a state is the Expected Return that can be obtained -- under a given policy -- after visiting that state. If the state is terminal (which is the case of \\(S_T\\)), then the episode _ends_ when we land on it. This means that we can't obtain _any_ reward after visiting that state, and consequently that the state's value is _zero_!

So, continuing the book's proof:

$$
= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \cdots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t}(0 - 0)\\
= \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k
$$

Now we're ready to account for a value function estimate that _changes within the episode_.

## When estimates change within the episode

When we perform a TD update within the episode, it looks like this:

$$
V_{t+1}(S_t) \dot{=} V_t(S_t) + \alpha [R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t)]\\
= V_t(S_t) + \alpha \delta_t \text{ , and}\\
V_{t+1}(s') \dot{=} V_t(s') \forall s' \neq S_t
$$

Let's look at how to express the difference between the return at time \\(t\\) and the value estimate of state \\(S_t\\), which is exactly what Exercise 6.1 asks us to do:

$$
G_t - V_t(S_t) = R_{t+1} + \gamma G_{t+1} - V_t(S_t) + (\gamma V_{t+1}(S_{t+1}) - \gamma V_{t+1}(S_{t+1}))\\
= R_{t+1} - V_t(S_t) + \gamma V_{t+1}(S_{t+1})  + \gamma (G_{t+1} - V_{t+1}(S_{t+1}))
$$

We _almost_ have everything that we need to make \\(\delta_t\\) appear. But instead of \\(R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t)\\), we have the same expression with \\(V_{t+1}(S_{t+1})\\) in place of \\(V_t(S_{t+1})\\). So we'll artificially introduce one into the expression by _adding zero in a clever way_.

$$
= R_{t+1} - V_t(S_t) + \gamma V_{t+1}(S_{t+1}) + \gamma (G_{t+1} - V_{t+1}(S_{t+1})) + \gamma V_t(S_{t+1}) - \gamma V_t(S_{t+1})\\
= R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t) + \gamma(V_{t+1}(S_{t+1}) - V_t(S_{t+1})) + \gamma (G_{t+1} - V_{t+1}(S_{t+1}))\\
= \delta_t + \gamma (V_{t+1}(S_{t+1}) - V_t(S_{t+1})) + \gamma(G_{t+1} - V_{t+1}(S_{t+1}))\\
$$

We will now look closely at \\(V_{t+1}(S_{t+1}) - V_t(S_{t+1})\\), since it's the only difference between this last step in the proof and the same thing for the case where estimates don't change within the episode.

The only change between \\(t\\) and \\(t+1\\) is made in the estimate for \\(S_t\\), because that's the only state we have information for. What I'm saying is, from the definition of the update in \\(V\\), after observing \\(R_{t+1}\\) and \\(S_{t+1}\\), we _only_ update the estimate for state \\(S_t\\), leaving _all the other estimates untouched_. This effectively implies that \\(V_t(S_{t+1})\\) and \\(V_{t+1}(S_{t+1})\\) are _the same thing_, except for \\(S_{t+1} = S_t\\)! When \\(S_{t+1} = S_t\\), we have:

$$
V_{t+1}(S_{t+1}) - V_t(S_{t+1}) = V_{t+1}(S_t) - V_t(S_t) = (V_t(S_t) + \alpha \delta_t) - V_t(S_t) =\\
\alpha \delta_t
$$

We can express both cases -- where \\(S_{t+1}\\) and \\(S_t\\) are/are not equal -- compactly as:

$$
V_{t+1}(S_{t+1}) - V_t(S_{t+1}) = \mathbf{I}( S_{t+1} = S_t ) (\alpha \delta_t)
$$

where \\(\mathbf{I}\\) is the _indicator function_, which returns \\(1\\) when the predicate inside it is true, and \\(0\\) otherwise. We can then introduce this in our proof:


$$
= \delta_t + \gamma (\mathbf{I}( S_{t+1} = S_t ) (\alpha \delta_t)) + \gamma(G_{t+1} - V_{t+1}(S_{t+1}))\\
= (1 + \alpha\gamma \mathbf{I}( S_{t+1} = S_t )) \delta_t + \gamma (G_{t+1} - V_{t+1}(S_{t+1}))\\
= (1 + \alpha\gamma \mathbf{I}( S_{t+1} = S_t )) \delta_t
+ \gamma (1 + \alpha\gamma \mathbf{I}(S_{t+2} = S_{t+1}))\delta_{t+1} + \gamma^2 (G_{t+2} - V_{t+2}(S_{t+2}))\\
= (1 + \alpha \gamma\mathbf{I}( S_{t+2} = S_t )) \delta_t +
\gamma (1+ \alpha \gamma\mathbf{I}( S_{t+2} = S_{t+1}))\delta_{t+1} + \cdots +
\gamma^{T-t-1}(1 + \alpha \gamma \mathbf{I}( S_T = S_{T-1} ))\delta_{T-1}\\
= \sum_{k=t}^{T-1} \gamma^{k-t} (1 + \alpha \gamma \mathbf{I}( S_{k+1} = S_k))\delta_k
$$

(again, \\(G_T\\) and \\(V_t(S_T), 0 \leq t \leq T\\) are equal to zero for the same reasons as before).

Which means that, even though our state estimates change within an episode, the error between them and the corresponding returns can be neatly written as a sum of the TD errors, plus an additional factor that is equal to \\(\alpha\\) everytime \\(S_{t+1}\\) is equal to \\(S_t\\).

Thanks for bearing with me through what I think is a very nice and rewarding proof!

## References

* Sutton and Barto's book, 2nd edition, page 121
