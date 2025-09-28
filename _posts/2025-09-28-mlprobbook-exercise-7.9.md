---
title: "Machine Learning: A Probabilistic Perspective, Exercise 7.9"
date: 2025-09-27 09:00:00 +0200
description: Resolution of the exercise 7.9 in Murphy's book
tags: machine-learning exercises probability
categories: ml prob-ml
layout: post
---

## Hello again!

Hello there, if anyone's reading this. It's officially been a while since I've last written on
this blog. Lots of stuff changed: from 2020 until now I left Google Brazil, became a PhD
Candidate with Utrecht University in the Netherlands (2021) and am now officially entering the
**last year** of my PhD. This means by this time next year, I'll be a doctor! Exciting!!

I am working with Dimensionality Reduction for Data Visualization. My approach is one based on
Neural techniques, that is to say, I am mostly employing Neural Networks to create, control,
distort, and explain Dimensionality Reduction plots. I am also working on sanely _inverting_ the
DR process, and using that for Explainable AI. You can get a good overview of my work by accessing
my [Google Scholar](https://scholar.google.com.br/citations?user=WVXX6mYAAAAJ&hl=en) page.

Don't hesitate to shoot me an email if you'd like to know more!

## Why am I back?

Well, I'm going through some classic Machine Learning bibliography as part of my studies for what
I intend to do after my doctorate. The plan is to become a university professor, hopefully in my
home country (Brazil). For that, I need to take a (quite difficult) test, for which some broad
and deep knowledge is required. Right now, I'm going through Kevin P. Murphy's "Machine Learning:
a probabilistic approach."

The book covers lots of different subjects in ML from a probabilistic perspective, including lots
of Bayesian inference. It's slightly above my level in terms of Maths, which means I _do struggle_
with the exercises. In every single chapter there's at least one exercise that throws me for a loop.
Sometimes that means not even understanding exactly what the exercise is asking.

I'd like to share some **solutions** to some exercises in this blog here and there, which is
why I'm writing today. In particular, although I *have found* the solution to this exercise online, I felt it was quite lacking in the level of detail I wanted to have and was somewhat handwavy. I hope my explanation will provide the reader with a deeper understanding of what the exercise requests.

## Exercise 7.9 - Generative Model for Linear Regression

In this blog post, I'll walk through the solution and reasoning for Exercise 7.9 in the book
(that's page 244). I'll copy the exercise here for practical reasons:

> Linear regression is the problem of estimating \\(\mathbb{E}[Y\|\mathbf{x}]\\) using a linear function of the form \\(w_0 + \mathbf{w}^T \mathbf{x}\\). Typically we assume that the conditional distribution of \\(Y\\) given \\(\mathbf{X}\\) is Gaussian. We can either estimate this conditional Gaussian directly (a discriminative approach), or we can fit a Gaussian to the joint distribution of \\(\mathbf{X}, Y\\) and then derive \\(\mathbb{E}[Y | \mathbf{X} = \mathbf{x}]\\).
> <br />
> In Exercise 7.5 we showed that the discriminative approach leads to these equations
> <br />
> $$\begin{aligned}\mathbb{E}[Y | \mathbf{x}] &= w_0 +  \mathbf{w}^T \mathbf{x} \\ w_0 &= \bar{y} - \bar{\mathbf{x}}^T \mathbf{w} \\ \mathbf{w} &= (\mathbf{X}_c^T \mathbf{X}_c)^{-1} \mathbf{X}_c^T \mathbf{y}_c\\\end{aligned}$$
> <br />
> where \\(\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{X}}\\) is the centered input matrix, and \\(\bar{\mathbf{X}} = \mathbf{1}_n \bar{\mathbf{x}}^T\\) replicates \\(\bar{\mathbf{x}}\\) across the rows. Similarly, \\(\mathbf{y}_c = \mathbf{y} - \bar{\mathbf{y}}\\) is the centered output vector, and \\(\bar{\mathbf{y}} = \mathbf{1}_n \bar{y}\\) replicates \\(\bar{y}\\) across the rows.
> <br />
> a. By finding the maximum likelihood estimates of \\(\mathbf{\Sigma} _{XX}, \mathbf{\Sigma} _{XY}, \mathbf{\mu}_X\\), and \\(\mathbf{\mu}_Y\\), derive the above equations by fitting a joint Gaussian to \\(\mathbf{X}, Y\\) and using the formula for conditioning a Gaussian (see Section 4.3.1). Show your work.

OKAY, after this LONG exercise description, we can finally get started.

## What it's asking of us

So far in this Chapter, we have been fitting the regression coefficients directly to the data by working with the likelihood function and maximizing it.

That is, we have been assuming a model of the form
$$
\begin{equation}
p(y | \mathbf{x}) = \mathscr{N}(y | w_0 + \mathbf{w}^T \mathbf{x}, \sigma^2).
\end{equation}
$$
We then perform maximization of this likelihood (or better, of its log form, since the log-likelihood has the same optimum as the likelihood), assuming a dataset of entries has been seen. This leads to the minimization of the negative log-likelihood function (NLL), which has the form

$$
J(\mathbf{w}) = (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w}),
$$

assuming the data has been centered and we are not using regularization. This leads to the values of \\(w_0\\) and \\(\mathbf{w}\\) mentioned in the exercise description.

Now, what we are aiming to do, is to fit a full *joint* model of \\(\mathbf{X}\\) and \\(y\\). From that model, we can easily derive a discriminative model with Bayes' theorem, since

$$
p(\mathbf{X}, Y) = p(Y | \mathbf{X}) p(\mathbf{X}).
$$

The process for building the model is quite different, but should lead us to the same result, which is what we will show below.

## Fitting a joint model

To fit a joint model to X and y, we simply consider that we have a distribution over the *concatenated* \\(\mathbf{x}\\) and \\(y\\) values. That is, assume a model, which will be a Gaussian, of the form:

$$
p\left(\begin{bmatrix} \mathbf{x} \\ y\end{bmatrix}\right) = \mathscr{N}\left(\begin{bmatrix} \mathbf{x} \\ y\end{bmatrix} \Bigg| \mu, \Sigma \right).
$$

In this model, we can identify:

$$
\mu = \begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma _{XX} & \Sigma _{XY} \\ \Sigma _{YX} & \Sigma _{YY} \end{bmatrix},
$$

where additionally, we know that both \\(\mu_Y\\) and \\(\Sigma _{YY}\\) are *scalars*, since y is a scalar.

Fitting such a model is theory we already know at this point in the book. The MLE estimates for *any* multivariate normal distribution are:

$$
\begin{align}
\widehat{\mu} &=  \frac{1}{n} \sum_{i=1}^n \mathbf{v}_i \\
\widehat{\Sigma} &= \frac{1}{n} \sum_{i=1}^n (\mathbf{v}_i - \widehat{\mu})(\mathbf{v}_i - \widehat{\mu})^T,
\end{align}
$$

where I'm using \\(\mathbf{v}_i\\) to represent each observation (so as to not cause confusion with our use of \\(\mathbf{x}, y\\)). In this exercise, \\(\mathbf{v_i} = [\mathbf{x}_i^T y_i]^T\\).

## MLE for the Joint Mean

Using the formula described above for the MLE of the mean and adapting it to our joint distribution over X and y, we have:

$$
\widehat{\mu} = \frac{1}{n} \sum_{i = 1}^n \begin{bmatrix} \mathbf{x}_i \\ y_i \end{bmatrix} = \begin{bmatrix} \frac{1}{n}\sum_{i=1}^n x_i \\ \frac{1}{n} \sum_{i=1}^n y_i\end{bmatrix} = \begin{bmatrix}\bar{\mathbf{x}} \\ \bar{y}\end{bmatrix},
$$

so the MLE for the joint mean is the concatenation of the MLEs for each individual variable over which we have built the joint. This means we already have \\(\mu_X = \bar{\mathbf{x}}\\) and \\(\mu_Y = \bar{y}\\).

## MLE for the Joint Covariance Matrix

Using again the formulas above, this time the one for calculating \\(\widehat{\Sigma}\\), we have:

$$
\begin{align}
\widehat{\Sigma} = \frac{1}{n} \sum_{i=1}^n \left(\begin{bmatrix} \mathbf{x}_i \\ y_i \end{bmatrix} - \begin{bmatrix}\bar{\mathbf{x}} \\ \bar{y}\end{bmatrix}\right)\left(\begin{bmatrix} \mathbf{x}_i \\ y_i \end{bmatrix} - \begin{bmatrix}\bar{\mathbf{x}} \\ \bar{y}\end{bmatrix}\right)^T = \frac{1}{n}\sum_{i=1}^n\left(\begin{bmatrix} \mathbf{x}_i - \bar{\mathbf{x}} \\ y_i - \bar{y}\end{bmatrix}\right)\left(\begin{bmatrix} \mathbf{x}_i - \bar{\mathbf{x}} \\ y_i - \bar{y}\end{bmatrix}\right)^T
\end{align}
$$

We will (and kind of have to) work with this block-wise.

$$
= \begin{bmatrix}
\frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T & \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(y_i - \bar{y})^T \\
\frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})^T (y_i - \bar{y}) & \frac{1}{n} \sum_{i=1}^n(y_i - \bar{y})(y_i - \bar{y})^T
\end{bmatrix},
$$

but since the \\(y\\)'s are scalars:

$$
\widehat{\Sigma} = \begin{bmatrix}
\frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T & \frac{1}{n}\sum_{i=1}^n (y_i - \bar{y}) (\mathbf{x}_i - \bar{\mathbf{x}})\\
\frac{1}{n}\sum_{i=1}^n (y_i - \bar{y})(\mathbf{x}_i - \bar{\mathbf{x}})^T & \frac{1}{n} \sum_{i=1}^n(y_i - \bar{y})^2
\end{bmatrix}.
$$

But remember that the covariance matrix \\(\Sigma\\) can be block-partitioned as

$$
\Sigma = \begin{bmatrix} \Sigma _{XX} & \Sigma _{XY} \\ \Sigma _{YX} & \Sigma _{YY} \end{bmatrix},
$$

meaning its MLE can be partitioned the same way. Therefore, we read from the MLE for \\(\Sigma\\) the MLEs for each one of its submatrices:

$$
\begin{align}

\widehat{\Sigma} _{XX} &= \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T = \frac{1}{n} \mathbf{X}_c^T \mathbf{X}_c\\
\widehat{\Sigma} _{XY} &= \widehat{\Sigma} _{YX}^T = \frac{1}{n}\sum_{i=1}^n (y_i - \bar{y}) (\mathbf{x}_i - \bar{\mathbf{x}}) = \frac{1}{n} \mathbf{X}_c^T \mathbf{y}_c \\
\widehat{\Sigma} _{YY} &= \widehat{\sigma}_Y^2 = \frac{1}{n} \sum_{i=1}^n(y_i - \bar{y})^2 = \frac{1}{n} \mathbf{y}_c^T \mathbf{y}_c

\end{align}
$$

## Where the magic happens: Conditioning

Now, to find the distribution \\(p(y \| \mathbf{X})\\), we will use the conditioning formulas for Gaussians stated in Section 4.3.1, a subset of which I copy below:

If \\(\mathbf{x}_1, \mathbf{x}_2\\) are jointly Gaussian, with mean and covariance matrix

$$
\mu = \begin{bmatrix}\mu_1 \\ \mu_2\end{bmatrix} \\
\Sigma = \begin{bmatrix} \Sigma _{11} & \Sigma _{12} \\ \Sigma _{21} & \Sigma _{22} \end{bmatrix},
$$

the following holds:

$$
\begin{align}
p(\mathbf{x}_1 | \mathbf{x_2}) & = \mathscr{N}(\mathbf{x}_1 | \mu _{1|2}, \Sigma _{1|2}) \\
\mu _{1|2} &= \mu_1 + \Sigma _{12} \Sigma _{22}^{-1}(\mathbf{x}_2 - \mu_2) \\
\Sigma _{1|2} &= \Sigma _{11} - \Sigma _{12} \Sigma _{22}^{-1} \Sigma _{21} \\
\end{align}
$$

Now we just need to plug in the equivalent submatrices and vectors for computing the distribution of \\(y\\) given \\(\mathbf{x}\\). In particular, we are interested in \\(\mathbb{E}[y \| \mathbf{x}]\\) (see exercise description), so we should compute \\(\mu _{Y \| X}\\).

From the formula above, replacing \\(1 \leftrightarrow y, 2 \leftrightarrow \mathbf{x}\\):

$$
\begin{align}
\mu_{y | \mathbf{x}} &= \mu_Y + \Sigma _{YX} \Sigma _{XX}^{-1} (\mathbf{x} - \mu_X) \\
&= \bar{y} - \Sigma _{YX} \Sigma _{XX}^{-1} \bar{\mathbf{x}} + \Sigma _{YX} \Sigma _{XX}^{-1} \mathbf{x}.
\end{align}
$$

In the expression above, we have one term that is independent of \\(\mathbf{x}\\) and one that is multiplying \\(\mathbf{x}\\). According to our model, the independent term is \\(w_0\\) and the term multiplying \\(\mathbf{x}\\) is \\(\mathbf{w}^T\\) (since the linear term is \\(\mathbf{w}^T \mathbf{x}\\)). Hence:

$$
\begin{align}
\mathbf{w}^T &= \Sigma _{YX} \Sigma _{XX}^{-1} = (\Sigma _{XY}^T) (\frac{1}{n} \mathbf{X}_c^T \mathbf{X}_c)^{-1} = n (\frac{1}{n} \mathbf{X}_c^T \mathbf{y}_c)^T (\mathbf{X}_c^T \mathbf{X}_c)^{-1} \\
\mathbf{w} &= ((\mathbf{X}_c^T \mathbf{y}_c)^T(\mathbf{X}_c^T\mathbf{X}_c)^{-1})^T = (\mathbf{X}_c^T \mathbf{X}_c)^{-T} (\mathbf{X}_c^T \mathbf{y}_c) = (\mathbf{X}_c^T\mathbf{X}_c)^{-1}\mathbf{X}_c^T \mathbf{y}_c
\end{align}
$$,

where we use the fact that the inverse of a symmetric matrix is also symmetric (*i.e.*, \\((\mathbf{X}_c^T \mathbf{X}_c)^{-T} = ((\mathbf{X}_c^T \mathbf{X}_c)^{-1})^T = (\mathbf{X}_c^T \mathbf{X}_c^T)^{-1}\\)).

As for \\(w_0\\), we have:

$$
w_0 = \bar{y} - \Sigma _{YX} \Sigma _{XX}^{-1} \bar{\mathbf{x}},
$$

where we recognize the term multiplying \\(\bar{\mathbf{x}}\\) as \\(\mathbf{w}^T\\), meaning:

$$
w_0 = \bar{y} - \mathbf{w}^T \bar{\mathbf{x}}.
$$

This concludes the proof.

## Second part of the Exercise

> What are the advantages and disadvantages of this approach compared to the standard discriminative approach?

Well, let's see... the most obvious advantage is that now we have a full-blown generative model for \\(\mathbf{X}, Y\\). This means we can compute the conditional distribution *in the other direction* just as easily!

$$
\begin{align}
p(\mathbf{x} | y) &= \mathscr{N}(\mathbf{x} | \mu _{X | Y}, \Sigma _{X|Y}) \\
\mu _{X|Y} &= \mu_X + \Sigma _{XY} \Sigma _{YY}^{-1} (y - \mu_Y) \\
\Sigma _{X|Y} &= \Sigma _{XX} - \Sigma _{XY} \Sigma _{YY}^{-1} \Sigma _{YX}
\end{align}
$$

This allows us to easily sample data points that are associated with a given value for the regressed variable, for example. We can also easily derive marginals for both \\(\mathbf{x}\\) and \\(y\\).

As for the disadvantages, honestly I do not see any so far. Typically, when fitting a joint directly, it might be harder to find samples for all relevant regions of the data space. But here, it's not the case: the computations are all the same for the full joint and for the discriminative model. We still have to compute \\(\mathbf{w}\\) and \\(w_0\\), just as before. We just have some more machinery to sample for conditionals and marginals, if we choose to do so.


## References

- MURPHY, Kevin P. "Machine Learning: A Probabilistic Perspective". 2012. The MIT Press.
