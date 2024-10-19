---
layout: page
title: About
permalink: /about/
---



Hi, I'm Alister Machado dos Reis, a <span id="age"></span>-years old PhD Student from Porto Alegre, Brazil, currently living in Utrecht, the Netherlands. I obtained my B.Sc. in Computer Science ([Institute of Informatics](https://www.inf.ufrgs.br/site/)) at the Federal University of Rio Grande do Sul ([UFRGS](http://www.ufrgs.br/ufrgs/inicial)) in March 2018.
I was also part of the BRAFITEC exchange program between the years of 2015 and 2017. It is one of the most competitive exchange programs of my country.

As a part of this program, I was also enrolled at the INP Grenoble (Institut National Polytechnique de Grenoble) in Grenoble, France. There, I was a student at the École Nationale Supérieure d'Informatique et Mathématiques Apliquées de Grenoble (ENSIMAG). I was enrolled as an Information Systems Engineering student between 2015 and 2016, and later I was a student in the Master's Program in Data Science (2016-2017).

You can find my Master's Thesis online here (it has a short introduction in Portuguese, but the main text is in English): [Incremental Learning Applied to Streaming Environments](https://lume.ufrgs.br/handle/10183/175067), [PDF](https://lume.ufrgs.br/bitstream/handle/10183/175067/001065053.pdf?sequence=1&isAllowed=y).

My exchange program entailed a double degree, which means I have obtained degrees from both the Brazilian and French institutes.

Currently I'm a PhD Student at the [Utrecht University](https://www.uu.nl/) with the [Visualization and Graphics Group](https://vig.science.uu.nl/). You can read my publications on [Google Scholar](https://scholar.google.com.br/citations?user=WVXX6mYAAAAJ&hl=en).

When I started this blog, I was doing mostly Reinforcement Learning. Since then, my interests have shifted towards Multidimensional Projections and High-Dimensional Data Visualization, especially with the aid of Machine Learning methods.

You can find me on [LinkedIn](https://www.linkedin.com/in/alistermachado/), [Github](https://github.com/{{site.github_username}}/) and [Twitter](https://twitter.com/{{site.twitter_username}}), and here is my [Resume](/assets/cv/CV_AlisterReis.pdf), updated in October 2024.

Hope you enjoy the content!

<!--
This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

You can find the source code for the Jekyll new theme at:
{% include icon-github.html username="jekyll" %} /
[minima](https://github.com/jekyll/minima)

You can find the source code for Jekyll at
{% include icon-github.html username="jekyll" %} /
[jekyll](https://github.com/jekyll/jekyll)
-->

<script type='text/javascript'>
    let today = new Date();
    let myAge = today.getFullYear() - 1995 - 1;
    if (today.getMonth() >= 12 && today.getDate() >= 19) {
        myAge += 1;
    }
    document.getElementById("age").innerHTML = myAge;
</script>