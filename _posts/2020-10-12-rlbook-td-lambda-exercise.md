---
title: "Reinforcement Learning: An Introduction – Exercise 12.5"
date: 2020-10-12 9:00:00 -0300
description: Resolution of the exercise 12.5 in Sutton and Barto's book, 2nd edition
tags: reinforcement-learning mdp machine-learning exercises rlbook td td-lambda
categories: ml reinf-learn
layout: default
---

Damn, I have to post things more regularly... it's been tough to keep writing
things these days though. Nonetheless, I'm still going through Sutton and Barto's
"Reinforcement Learning: an introduction" book, on its 2nd edition. I'm now
working through the Eligibility Traces chapter (Ch. 12), and there is this
particular exercise that took me a few tries to get just right.

For this post, more than just sharing with you the final answer and how to get
there, I want to walk through how I struggled with this exercise, and key
learnings that enabled me to finally reach the solution. I hope you'll find
this interesting!

# Quick Recap: where are we?

We're currently studying Eligibility Traces and the associated \\(\text{TD}(\lambda)\\)
algorithm. Eligibility Traces are not themselves the subject of this post, though,
so I won't go too deep in there. Anyway, I should say _a few_ things. Firstly,
the \\(\text{TD}(\lambda)\\) algorithm and its variants are based on _averaging_
every _n-step return_ and using it as a target for learning. The _n-step return_
is defined as:

$$
G_{t:t+n} \dot{=} R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n \widehat{v}(S_{t+n})
$$

which means: "collect the \\(n\\) available rewards as if we're building a regular
return \\(G_t\\), and truncate after that using the current estimate for the
value of the \\(t+n\\)-th state." **Important:** in the book, \\(\widehat{v}\\) is
an approximation to the true value function and has a dependency on a _parameter
vector_ \\(\mathbf{w}\\). We can safely refrain from explicitly representing
that dependency -- and I will -- for the sake of simplicity because _this particular exercise_, as you'll see, keeps \\(\mathbf{w}\\) constant.

The _full \\(\lambda\\)-return_ is then defined on top of the _n-step return_ as:

$$\begin{aligned}
G_t^\lambda \dot{=}& (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n} \\
=& (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t
\end{aligned}
$$

For this exercise, we'll be looking at the _truncated \\(\lambda\\)-return_,
which takes on a very similar form, replacing the episode ending time \\(T\\)
with a _horizon_ \\(h\\):

$$
G_{t:h}^\lambda \dot{=} (1-\lambda) \sum_{n=1}^{h-t-1}\lambda^{n-1} G_{t:t+n} + \lambda^{h-t-1} G_{t:h},\quad 0\leq t < h \leq T.
$$

All of these are used as _targets_ for methods that estimate state- or action-value
functions. Read the book for more details, there's a link to it at the bottom.

# The Exercise

> Several times in this book (often in exercises) we have established that
> returns can be written as sums of TD errors if the value function is held constant. Why
> is (12.10) another instance of this? Prove (12.10).

Where (12.10) is the following equation:

$$\begin{aligned}
&G_{t:t+k}^\lambda = \widehat{v}(S_{t}, \mathbf{w}_{t-1}) + \sum_{i=t}^{t+k-1} (\gamma\lambda)^{i-t} \delta_i' \quad (12.10) \\
&\text{where} \\
&\delta_t' \dot{=} R_{t+1} + \gamma \widehat{v}(S_{t+1}, \mathbf{w}_t) - \widehat{v}(S_t, \mathbf{w}_{t-1})
\end{aligned}
$$

# The "why" part

It's important to write things as sums of TD-Errors because it is something that
can be calculated at each step. If the update target is a sum of TD-errors, it
is easy to keep track of it without having to hold a large history of the interactions
with the environment in memory.

# The Proof

Let's get our hands dirty! I'll comment these with my thought process so you can
keep up.

First of all, the exercise tells us to consider that the parameter vector will be
fixed throughout the episode, so there's no need to keep track of it. This means
we can drop the explicit dependency on it from the equations. From that, we get:

$$
\delta_t' = R_{t+1} + \gamma \widehat{v}(S_{t+1}) - \widehat{v}(S_t) = \delta_t
$$

, where \\(\delta_t\\) is the regular TD-error.

The first thing I thought of doing was just playing around with the definition
for truncated \\(\lambda\\)-return and seeing what happened.

$$\begin{aligned}
G_{t:t+k}^\lambda =& (1-\lambda)\sum_{n=1}^{k-1} \lambda^{n-1} G_{t:t+n} + \lambda^{k-1} G_{t:t+k} \\
=& (1-\lambda)\sum_{n=1}^{k} \lambda^{n-1} G_{t:t+n} \\
=& (1-\lambda)[G_{t:t+1} + \sum_{n=2}^k \lambda^{n-1} G_{t:t+n}] \quad &\text{(pulling one term out of the sum)} \\
=& (1-\lambda)[R_{t+1} + \gamma \widehat{v}(S_{t+1})] + (1-\lambda)\sum_{n=2}^k \lambda^{n-1} G_{t:t+n} \\
=& (1-\lambda)[R_{t+1} + \gamma \widehat{v}(S_{t+1}) - \color{red}{\widehat{v}(S_t)}] + (1-\lambda)[\sum_{n=2}^k \lambda^{n-1} G_{t:t+n} + \color{red}{\widehat{v}(S_t)}] \quad &\text{(adding 0 in a convenient way)} \\
=& (1-\lambda)\delta_t + (1-\lambda)[\sum_{n=2}^k \lambda^{n-1}G_{t:t+n} + \widehat{v}(S_t)]
\end{aligned}
$$

And then I got a little stuck. It's weird that there's a \\((1-\lambda)\widehat{v}(S_t)\\) term in there.
At the same time, the equality I'm trying to prove involves a standalone \\(\widehat{v}(S_t)\\) term...
My only hope is to expand it all and hope that things cancel out once I work the
deeper levels. The only thing that is in place so far is the \\(\delta_t\\) term,
which kind of fits with what we're trying to prove, except for the leading \\((1-\lambda)\\) coefficient.
Also, you might have already spotted where I made a big mistake that will bite my ass in the short future.

Let's keep going:

$$\begin{aligned}
=& \delta_t - \lambda \delta_t + \widehat{v}(S_t) - \lambda\widehat{v}(S_t) +
  \underbrace{\sum_{n=2}^k \lambda^{n-1}G_{t:t+n} -\lambda\sum_{n=2}^k \lambda^{n-1}G_{t:t+n}}_{\text{we'll look into this specific part for answers}} \quad \color{blue}{(A)}
\end{aligned}
$$

Let's look into that subtraction between two sums more closely, by examining
one of its terms. For the case of \\(n=2\\):

$$\begin{align}
&\lambda G_{t:t+2} - \lambda^2 G_{t:t+2} \\
&= \lambda (R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2}) - \lambda(R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2}))) \\
&= \lambda [(1-\lambda) (R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2}))] \\
&= \lambda [(1-\lambda) (R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2})) \color{blue}{- \delta_t - \widehat{v}(S_t)}] \quad \text{(bringing some terms from } \color{blue}{(A)}\text{ into the expression to see if they help)} \color{red}{(B)} \\
&= \lambda [(1-\lambda) (R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2})) - R_{t+1} - \gamma\widehat{v}(S_{t+1}) \color{red}{+ \widehat{v}(S_t) - \widehat{v}(S_t)}] \color{red}{(C)} \\
&= \lambda (\color{red}{R_{t+1}} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2}) - \lambda(R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2})) \color{red}{- R_{t+1}} - \gamma \widehat{v}(S_{t+1})) \\
&= \lambda (\gamma \color{red}{(R_{t+2} + \gamma \widehat{v}(S_{t+2}) - \widehat{v}(S_{t+1}))} - \lambda(R_{t+1} + \gamma R_{t+2} + \gamma^2 \widehat{v}(S_{t+2}))) \\
&= \color{blue}{\lambda\gamma\delta_{t+1}} - \lambda^2 (\dots)
\end{align}
$$

At this point, I got hopelessly **stuck**. _But_ look at what we've achieved.
We have a term of the form \\(\lambda\gamma\delta_{t+1}\\) which is _exactly_ one
of the things we're looking for. And also, steps B and C in that development gave us
something very interesting... this might spark more ideas:

$$
\delta_t + \widehat{v}(S_t) = R_{t+1} + \gamma \widehat{v}(S_{t+1}) - \widehat{v}(S_t) + \widehat{v}(S_t) \\
= R_{t+1} + \gamma \widehat{v}(S_{t+1}) = G_{t:t+1}
$$

This means that (A) could be rewritten as

$$\begin{aligned}
&= \delta_t + \widehat{v}(S_t) - \lambda G_{t:t+1} + \sum_{n=2}^k \lambda^{n-1}G_{t:t+n} -\lambda\sum_{n=2}^k \lambda^{n-1}G_{t:t+n} \\
&= \delta_t + \widehat{v}(S_t) - \underbrace{\lambda G_{t:t+1}}_{\text{merge with last summation}} + \sum_{n=2}^k \lambda^{n-1}G_{t:t+n} - \sum_{n=2}^k \color{red}{\lambda^{n}}G_{t:t+n} \\
&= \delta_t + \widehat{v}(S_t) + \underbrace{\sum_{n=2}^k \lambda^{n-1}G_{t:t+n}}_{\text{has k - 1 terms}} - \underbrace{\sum_{\color{red}{n=1}}^k \lambda^{n}G_{t:t+n}}_{\text{has k terms}} \\
&= \delta_t + \widehat{v}(S_t) + \underbrace{\sum_{n=2}^k \lambda^{n-1}G_{t:t+n}}_{\text{has k - 1 terms}} - \underbrace{\sum_{n=1}^{\color{red}{k-1}} \lambda^{n}G_{t:t+n}}_{\text{has k - 1 terms}} - \lambda^k G_{t:t+k} \\
&= \delta_t + \widehat{v}(S_t) + \sum_{i=1}^{k-1} \lambda^i(G_{t:t+i+1} - G_{t:t+i}) - \lambda^k G_{t:t+k} \quad \text{(group terms by } \lambda\text{'s exponent)}
\end{aligned}
$$

Now, that difference between _n-step returns_ looks interesting. I wonder if
there's a closed formula for it...

$$\begin{align}
G_{t:t+i+1} - G_{t:t+i} &= (\underbrace{R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{i-1} R_{t+i}}_{\text{this}} +  \gamma^i R_{t+i+1} + \gamma^{i+1} \widehat{v}(S_{t+i+1}))
  - (\underbrace{R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{i-1} R_{t+i}}_{\text{is equal to this, and cancels out}} + \gamma^{i} \widehat{v}(S_{t+i})) \\
&= \gamma^i R_{t+i+1} + \gamma^{i+1} \widehat{v}(S_{t+i+1}) - \gamma^i \widehat{v}(S_{t+i}) \\
&= \gamma^i (R_{t+i+1} + \gamma \widehat{v}(S_{t+i+1}) - \widehat{v}(S_{t+i})) = \gamma^i \delta_{t+i}
\end{align}
$$

Plugging this back into our equation:

$$
= \delta_t + \widehat{v}(S_t) + \sum_{i=1}^{k-1}\lambda^i (\gamma^i \delta_{t+i}) - \lambda^k G_{t:t+k} \\
= \delta_t + \widehat{v}(S_t) + \sum_{i=1}^{k-1} (\lambda\gamma)^i \delta_{t+i} - \lambda^k G_{t:t+k}
$$

But hang on... this is _so close_ to the actual answer! If you're paying close
attention, you'll notice that I made a mistake **right at the beginning**, when
I brought what was a standalone \\(\lambda^{k-1}G_{t:t+k}\\) into a summation
that _had a coefficient_ of \\((1-\lambda)\\) in front of it. This means I introduced
a value of \\(- \lambda \lambda^{k-1} G_{t:t+k}\\) that _wasn't there_. That's the
extra term showing up in the proof! If we, armed with all the things we know,
go back to the beginning, deriving the correct answer is **now** easy. And it is
so only because we scratched our heads and hit dead ends while figuring things
out.

$$\begin{aligned}
G_{t:t+k}^\lambda &= (1-\lambda) \sum_{n=1}^{k-1} \lambda^{n-1} G_{t:t+n} + \lambda^{k-1} G_{t:t+k} \\
&= \sum_{n=1}^{k-1} \lambda^{n-1} G_{t:t+n} - \sum_{n=1}^{k-1} \lambda^n G_{t:t+n} + \lambda^{k-1}G_{t:t+k} \\
&= G_{t:t+1} + \underbrace{\sum_{n=2}^{k-1} \lambda^{n-1} G_{t:t+n} - \sum_{n=1}^{k-2} \lambda^n G_{t:t+n}}_{\text{same number of terms, same coefficients}} - \lambda^{k-1} G_{t:t+k-1} + \lambda^{k-1} G_{t:t+k} \\
&= G_{t:t+1} + \sum_{i=1}^{k-2} \lambda^i(G_{t:t+i+1} - G_{t:t+i}) + \lambda^{k-1} (G_{t:t+k} - G_{t:t+k-1}) \\
&= G_{t:t+1} + \sum_{i=1}^{k-2} \lambda^i (\gamma^i \delta_{t+i}) + \lambda^{k-1} \gamma^{k-1} \delta_{t+k-1} \\
&= R_{t+1} + \gamma \widehat{v}(S_{t+1}) + \sum_{i=1}^{k-1} (\lambda\gamma)^i \delta_{t+i} \\
&= (R_{t+1} + \gamma \widehat{v}(S_{t+1}) - \widehat{v}(S_t)) + \widehat{v}(S_t) + \sum_{i=1}^{k-1} (\lambda\gamma)^i \delta_{t+i} \\
&= \delta_t + \widehat{v}(S_t) + \sum_{i=1}^{k-1} (\lambda\gamma)^i \delta_{t+i} = \widehat{v}(S_t) + \sum_{i=0}^{k-1} (\lambda\gamma)^i \delta_{t+i} \\
&= \widehat{v}(S_t) + \sum_{i=t}^{t+k-1} (\lambda\gamma)^{i-t} \delta_i \\
&& \blacksquare
\end{aligned}
$$

This was a **really** long proof, but only because I didn't know the path to
the solution and made a few mistakes on top of that. I think we're more used to
the flawless path being shared, but the failures teach us so much more. This is
the motivation behind this post. I have found myself thinking I was incapable of
deriving some proofs, but keeping at it and noticing the nice results you get
along the way eventually leads to the solution. I hope this encourages every
student that struggles with this type of thing.

# References
* Sutton and Barto's book, 2nd edition: [link](http://incompleteideas.net/book/RLbook2018.pdf)
