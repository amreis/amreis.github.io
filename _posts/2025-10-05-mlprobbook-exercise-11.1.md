---
title: "Machine Learning: A Probabilistic Perspective, Exercise 11.1"
date: 2025-10-05 22:00:00 +0200
description: Resolution of Exercise 11.1 in Kevin P. Murphy's book
tags: machine-learning exercises probability
categories: ml prob-ml
layout: post
author: Alister Machado
---


## Exercise 11.1 - Student's T Distribution as Infinite Mixture of Gaussians

In this blog post, I'll walk through the solution and reasoning for Exercise 11.1 in the book
(that's page 376). The exercise description is quite simple:

> Derive Equation 11.61. For simplicity, assume a one-dimensional distribution.

Equation 11.61 is the following cutesy thing:

$$
\newcommand{\D}{\mathrm{d}}

\mathcal{T}(\mathbf{x}_i | \mu, \Sigma, \nu) = \int \mathcal{N}(\mathbf{x}_i | \mu, \Sigma / z_i) \text{Ga}(z_i | \nu/2, \nu/2) \,\D z_i,
$$

where \\(\mathcal{T}\\) is the PDF (probability density function) of the Student's T distribution with location \\(\mu\\), scale \\(\Sigma\\), and \\(\nu\\) degrees of freedom.

Remember that in this book this means a _definite_ integral over the entirety of the domain of the integration variable. Since we are integrating over \\(z_i\\), this means integrating in the interval \\([0, \infty)\\). Also, the Gamma distribution is parametrized with shape and _rate_, as opposed to shape and _scale_. Additionally, note that we are using the _location-scale_ T distribution. Typically, a T distribution only has a single parameter: the degrees of freedom \\(\nu\\). It is easy to [extend](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Location-scale_t_distribution) this to any location and scale.

## Context

Chapter 11 is talking about fitting Mixture Models and using the EM algorithm. A simple example of a mixture model is one where the likelihood is given by a Mixture of Gaussians, for instance:

$$
p(\mathbf{x} | \mu_{1\dots K}, \Sigma_{1\dots K}, \pi) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \mu_k, \Sigma_k).
$$

Such models are useful in many ways, and considerably more expressive than using a single distribution to fit some data. In this exercise, we aim to show that if we create a mixture of _infinitely many Gaussians_ with appropriately selected weights and appropriately modified parameters, we will obtain a Student's T distribution.

In order to convert the mixture above to a mixture of _infinitely many_ elements, we replace the sum with an integral

$$
p(\mathbf{x} | \theta) = \int \pi(z) \mathcal{N}(\mathbf{x} | \mu(z), \Sigma(z)) \D z.
$$

Where the importance weights \\(\pi_k\\) become functions  \\(\pi(z)\\), and something similar happens for both parameters of the Gaussian. The exercise says that we should use
* \\(\pi(z) = \textrm{Ga}(z \| \nu/2, \nu/2)\\)
* \\(\mu(z) = \mu\\)
* \\(\Sigma(z) = \Sigma / z\\)

## What the exercise is asking of us

The exercise tries to show us that computing the integral in Equation 11.61 we will arrive at the PDF of the Student's T distribution. In the end, this is quite some busy work manipulating formulas, but I did find it satisfying so I'm sharing my resolution here.

We want to arrive at the following expression, which is the PDF of a Student's T distribution:

$$
\mathcal{T}(\mathbf{x} | \mu, \tau, \nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2}) \tau \sqrt{\pi \nu}}
\left(1 + \frac{1}{\nu} \left( \frac{x - \mu}{\tau}\right)^2\right)^{\frac{-(\nu + 1)}{2}}.
$$

I have replaced \\(\Sigma\\) for \\(\tau\\) here since that is the notation in the book and we are considering the one-dimensional case (so \\(\tau\\) is a scalar standard deviation parameter). Note that in the integral expression for the Student's T distribution given in the exercise it is the _covariance matrix_ that is being scaled by \\(1/z_i\\). This means our _variance_ parameter in the one-dimensional case will be scaled by \\(z_i\\), i.e., the scalar equivalent to \\(\Sigma / z_i\\) is \\(\tau^2 / z_i\\).

Let's recall the PDF of the Normal and Gamma distributions:


$$
\begin{align}
\mathcal{N}(\mathbf{x} | \mu, \sigma^2) &= \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left[ -\frac{1}{2 \sigma^2} \left(x - \mu\right)^2\right] \\
\text{Ga}(z | \alpha, \beta) &= \frac{\beta^\alpha}{\Gamma(\alpha)} z^{\alpha - 1} \exp\left(-\beta z\right).
\end{align}
$$

Now let's substitute in the parameters we need to use for the proof:

$$
\begin{align}
\mathcal{N}(\mathbf{x} | \mu, \tau^2 / z) &= \frac{1}{\sqrt{\frac{2\pi \tau^2}{z}}} \exp\left[ -\frac{1}{2 \frac{\tau^2}{z}} \left(x - \mu\right)^2\right] \\
\text{Ga}(z | \frac{\nu}{2}, \frac{\nu}{2}) &= \frac{\left(\frac{\nu}{2}\right)^{\frac{\nu}{2}}}{\Gamma\left(\frac{\nu}{2}\right)} z^{\frac{\nu}{2} - 1} \exp\left(\frac{-\nu}{2} z \right)
\end{align}
$$

## Let's get our hands dirty!

Now it's time to compute this beast of an integral and _hope_ (yes, I said hope) we reach something that we can simplify or a structure that we recognize.

$$
\begin{align}

&\int \mathcal{N}(\mathbf{x} | \mu, \tau^2 / z) \text{Ga}(z | \frac{\nu}{2}, \frac{\nu}{2}) \D z = \\

&\int \frac{1}{\sqrt{\frac{2\pi \tau^2}{z}}} \exp\left[ -\frac{1}{2 \frac{\tau^2}{z}} \left(x - \mu\right)^2\right] \frac{\left(\frac{\nu}{2}\right)^{\frac{\nu}{2}}}{\Gamma\left(\frac{\nu}{2}\right)} z^{\frac{\nu}{2} - 1} \exp\left(\frac{-\nu}{2} z \right) \D z = \\

&\int \frac{\sqrt{z}}{\sqrt{2\pi}\tau} \exp\left[ -\frac{z}{2\tau^2}\right] \frac{\left(\frac{\nu}{2}\right)^{\frac{\nu}{2}}}{\Gamma\left(\frac{\nu}{2}\right)} z^{\frac{\nu}{2} - 1} \exp\left(\frac{-\nu}{2}\right) \D z = \\

&\frac{1}{\sqrt{2\pi} \tau} \frac{\left(\frac{\nu}{2}\right)^\frac{\nu}{2}}{\Gamma\left(\frac{\nu}{2}\right)} \int \exp\left[-\frac{z}{2} \left(\left(\frac{x - \mu}{\tau}\right)^2 + \nu \right)\right] z^{\frac{1}{2} + \frac{\nu}{2} - 1} \D z =\\

&\frac{1}{\sqrt{2\pi} \tau} \frac{\left(\frac{\nu}{2}\right)^\frac{\nu}{2}}{\Gamma\left(\frac{\nu}{2}\right)} \int \exp\left[-\frac{z\nu}{2}\left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2 \right)\right] z^{\frac{\nu - 1}{2}} \D z.
\end{align}
$$


WHEW that already took quite a lot of work just grouping similar terms and pushing constants out of the integral. Now, the _key step here_ is to recognize that the integrand is of the form
$$
z^{A} \exp (- B z)
$$
and the integration is over the entire domain of \\(z\\). Luckily, there is a distribution that looks _just like that_. It's the **Gamma** distribution again. Its PDF is an exponential times a power of z.

Now we just need to match parameters. We know two crucial things from the expression for the Gamma PDF:
* The exponent of \\(z\\) in a Gamma distribution is \\(\alpha - 1\\);
* The term inside the exponential is \\(-\beta z\\).

Therefore, the integrand is the _unnormalized_ PDF of a \\(\text{Ga}\left(z \| \alpha = \frac{\nu+1}{2}, \beta = \frac{\nu}{2}\left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2 \right)\right)\\). If you want to, check for yourself! You just need to substitute the parameters into the expression for the Gamma distribution (scroll up the page).

## Oh yeah, it's all coming together

Since we have the _unnormalized_ PDF in the integrand and we are integrating over the entire support of the distribution, the integral is equals to the _normalization constant_ of the distribution. For the Gamma distribution, that is:

$$
\begin{align}
\text{Ga}(z | \alpha, \beta) &= \frac{\beta^\alpha}{\Gamma(\alpha)} z^{\alpha - 1} \exp\left(-\beta z\right) =  \frac{1}{Z} z^{\alpha - 1} \exp\left(-\beta z\right) \\
Z &= \frac{\Gamma(\alpha)}{\beta^\alpha}.
\end{align}
$$

This means that

$$
\begin{align}
&\int \exp\left[-\frac{z\nu}{2}\left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2 \right)\right] z^{\frac{\nu - 1}{2}} \D z =  \frac{\Gamma\left( \frac{\nu+1}{2} \right)}{\left(\frac{\nu}{2}\left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2 \right)\right)^{\frac{\nu+1}{2}}}
\end{align}
$$

Plugging that back into the expression we were working with...

$$
\begin{align}
&\frac{1}{\sqrt{2\pi} \tau} \frac{\left(\frac{\nu}{2}\right)^\frac{\nu}{2}}{\Gamma\left(\frac{\nu}{2}\right)} \int \exp\left[-\frac{z\nu}{2}\left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2 \right)\right] z^{\frac{\nu - 1}{2}} \D z = \\
&\frac{1}{\sqrt{2\pi} \tau} \frac{\left(\frac{\nu}{2}\right)^\frac{\nu}{2}}{\Gamma\left(\frac{\nu}{2}\right)} \frac{\Gamma\left( \frac{\nu+1}{2} \right)}{\left(\frac{\nu}{2}\left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2 \right)\right)^{\frac{\nu+1}{2}}} = \\

&\frac{1}{\sqrt{2\pi}\tau} \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)} \frac{\left(\frac{\nu}{2}\right)^{\frac{\nu}{2}}}{\left(\frac{\nu}{2}\right)^{\frac{\nu+1}{2}}} \left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2\right)^{\frac{-(\nu+1)}{2}} = \\
&\frac{1}{\sqrt{2\pi}\tau} \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)} \left(\frac{\nu}{2}\right)^{\frac{\nu - (\nu+1)}{2}} \left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2\right)^{\frac{-(\nu+1)}{2}} = \\
&\frac{1}{\sqrt{2\pi}\tau} \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)} \left(\frac{\nu}{2}\right)^{-\frac{1}{2}} \left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2\right)^{\frac{-(\nu+1)}{2}} = \\

&\frac{\sqrt{2}}{\sqrt{2\pi}\tau \sqrt{\nu}} \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2\right)^{\frac{-(\nu+1)}{2}} = \\
&\frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\tau\sqrt{\pi\nu}} \left(1 + \frac{1}{\nu}\left(\frac{x - \mu}{\tau}\right)^2\right)^{\frac{-(\nu+1)}{2}} \blacksquare
\end{align}
$$

So, indeed, our result holds. Student's T distribution is equivalent to an *infinite* mixture of Gaussian distributions with scaled covariance matrices. I found this result so impressive and it felt like it would be so complicated to prove, but it's just nice how things work out!


## References

- MURPHY, Kevin P. "Machine Learning: A Probabilistic Perspective". 2012. The MIT Press.
