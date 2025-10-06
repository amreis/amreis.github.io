---
title: A Taste of Reinforcement Learning
date: 2017-05-28 17:20:00 -0700
description: An introduction to Reinforcement Learning
tags: reinforcement-learning mdp optimal-control machine-learning
categories: ml reinf-learn
layout: post
author: Alister Machado
---

This post also appeared in *dev.to* as [A Taste of Reinforcement Learning](https://dev.to/amreis/a-taste-of-reinforcement-learning)

For the last few months, I've been hearing more and more about Reinforcement Learning. You might have too. You might have seen DeepMind's [fantastic results](https://youtu.be/V1eYniJ0Rnk) in playing Atari games, or maybe you just stumbled upon the video of [a helicopter](https://www.youtube.com/watch?v=VCdxqn0fcnE) that learned how to best drive itself and do acrobatics.

All of those demonstrations have sparked my interest in the field, so I went on to get to know a little bit better what the fuss is all about. Here are some of my impressions, and why I'm currently in love with the field.

This post will be part of a series where I explain (hopefully in a good way) concepts related to this field. All feedback is appreciated since I'm really new to it and am looking to deepen my knowledge in it.

But less about me, more about Reinforcement Learning! Let's start!

## Why does it have a cute, special name?

You might think that this is only another way of doing Machine Learning, so why even bother giving it a special name? Well, it turns out that the type of problems that Reinforcement Learning aims to solve and the way it solves them do not fit _that well_ into "traditional" Machine Learning categories.

By that, I mean that calling it _supervised learning_, or _unsupervised learning_, or _semi-supervised learning_ does not quite capture what it's about. Here's why:

### The Setting

In Reinforcement Learning, we want to learn **how to behave** in a given **environment**. More specifically, we want to train an **agent** (something that is capable of taking **actions** and reading **observations** from the environment) to perform as well as it can.

But what does it even mean to _perform well_ in this setting? Reinforcement Learning associates how well you're doing with the idea of a **reward**. This is a numerical quantity that the agent gets for interacting with the environment. For example, if you're training an agent that is supposed to balance a sphere on a plate for as long as possible, you might reward it for every second that it manages to keep it on the plate.

The fact that we don't learn directly from point-label pairs, but instead from these rewards, creates some problems. The most immediate of them being the **credit assignment** problem.

### Credit Assignment: what is causing what?

Suppose the agent has been running for some time, and has seen a reward (be it positive or negative) at a given point in time. It has already visited many states (where it may or may not have seen rewards) and taken many actions by then. How do we know which of these previous decisions caused the agent to now receive this reward?

For example, if you are playing a game and exploring the map, did you die because you walked into a monster? Or maybe the problem was choosing to enter a given room where the enemies were too strong? This decision might have been made _several iterations before_ we actually get feedback from the environment.

Still, it is clear that we need to adjust our behavior. If we got negative reward, we should behave differently as to avoid it. Conversely for positive reward: such actions should become more frequent.

The task of adapting our actions using only this _delayed_ reward information is what renders Reinforcement Learning unique, and in my opinion, so special: you only have signals saying "this was good" or "this was bad", and still you can learn to behave in an environment only from that. It is, in a sense, a very general artificial intelligence, since it does not need to know anything about the dynamics or underlying concepts of the environment to learn how to deal with
it, using only its experience.

## A bit of theory

The above description was quite abstract when it comes to the math of this thing. I don't want to dive in too deep about it, so I recommend [David Silver's course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) on Reinforcement Learning. It's where I'm getting the fundamental concepts. There is one part that I feel that I should talk about, though...

### Markov Decision Processes

RL problems typically assume that you're in an environment that respects a formalism known as a **Markov Decision Process** (MDP for short). I will skip the formal definition and focus on how it behaves, maybe that's easier to understand. A finite MDP works as follows:

* It is composed of a _finite_ number of **states** (green circles)
* At each step in time, an agent can choose from a set of **actions** (red circles)
* Choosing an action at a given point in time causes the environment to produce a **reward** (yellow arrows) and transition to a **new state**

<p><a href="https://commons.wikimedia.org/wiki/File:Markov_Decision_Process_example.png#/media/File:Markov_Decision_Process_example.png"><img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Markov_Decision_Process_example.png" alt="Markov Decision Process example.png"></a>
</p>

Some things that are worth noting:

In an MDP, the states have the **Markov property**. This basically says that "the future is conditionally independent of the past, given the present". In practice, it means that every _individual_ state provides us with all the information we need to determine what will happen next (we don't need the full history of states).

Also, the transition to a new state might not be _deterministic_. Taking an action while being in a given state can lead to different following states (you have a _probability_ of landing in a state for each previous state and action, represented by the numeric values on the arrows in the above diagram).

## Wrapping up

This is just a very general idea on what this field is about, and I hope I was able to give you at least a flavor of it.

One thing I haven't talked about are **continuous** state/action spaces. These introduce some new problems that I want to cover in the next post where I'll talk about **Policy Evaluation** techniques. Evaluating a policy means that, given a function that determines our behavior in the environment, we want to know how good it is (in terms of the total reward we expect to get by following it).

I've also only cited the case where the environment behaves as an MDP. One question that might arise is "what happens if the states don't have the Markov property?". In theory, you could then create a state that has the Markov property by accumulating your whole history. This would incur other problems, such as the probability of seeing each individual state becoming really tiny.

Another possible variation is when we don't have access to the state information  itself, but only to some observation of the environment that is derived from it. This is quite common when we're dealing with "real world" data. We don't have the complete state information, but we can use the **observations** to reason about what's happening (we say the environment is a Partially Observable MDP in this case).

That's it for today! I hope you enjoyed the post and if you have feedback on it, it's highly appreciated. See you next time! 


