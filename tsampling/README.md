# tsampling
Thompson Sampling Multi-Armed Bandit for Python

A package that implements the Thompson Sampling approach for a Multi-Armed Bandit.
The purpose of this project is to help people simply create and maintain Thompson Sampling experiments that have Bernoulli and Poisson distributions.

## Usage

### Setting up the experiment:

The following method will instantiate the experiment with the default priors.
```python
from tsampling.bernoulli import BernoulliExperiment

experiment = BernoulliExperiment(arms=2)
```

You can also set your custom priors by using the Priors module:
```python

from tsampling.bernoulli import BernoulliExperiment
from tsampling.priors import BetaPrior

pr = BetaPrior()
pr.add_one(mean=0.5, variance=0.2, effective_size=10, label="option1")
pr.add_one(mean=0.6, variance=0.3, effective_size=30, label="option2")
experiment = BernoulliExperiment(priors=pr)
```

### Performing an action:
You can randomly choos which arm to "pull" in the multi-armed bandit:
```python
experiment.choose_arm()
```

### Updating reward:
You can update the different arms information by adding reward information:

```python
rewards = [{"label":"option1", "reward":1}, {"label":"option2", "reward":0}]
experiment.add_rewards(rewards)
```

### Plotting Posterior Distribution:
You can plot the posterior distribution 

```python
experiment.plot_posterior()
```

## Installation

### Pip 
```
pip install tsampling
```

## License
 Free software: MIT license

## Published Link

https://pypi.org/project/tsampling/0.0.1/

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

_Cookiecutter: https://github.com/audreyr/cookiecutter
`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

## References
* Roberts, S. (2021, December 16). Thompson Sampling - Towards Data Science. Medium. https://towardsdatascience.com/thompson-sampling-fc28817eacb8
* Russo. (n.d.). A Tutorial on Thompson Sampling. Stanford. Retrieved December 15, 2022, from https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
* Marmerola, G. D. (2017, November 21). Introduction to Thompson Sampling: the Bernoulli bandit. Guilherme’s Blog. https://gdmarmerola.github.io/ts-for-bernoulli-bandit/
* https://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf
* GeeksforGeeks. (2021, October 19). Introduction to Thompson Sampling | Reinforcement Learning. https://www.geeksforgeeks.org/introduction-to-thompson-sampling-reinforcement-learning/
* Solving Poisson multi-armed bandit with Thompson sampling. (n.d.). Gist. https://gist.github.com/eigenfoo/e9a9933d94524e6dee717276c6b6f732
* Verstraeten, T. (2020, April 21). Multi-Agent Thompson Sampling for Bandit Applications with Sparse Neighbourhood Structures. Nature. https://www.nature.com/articles/s41598-020-62939-3?error=cookies_not_supported&code=f4c773ab-c5f9-44ee-bd58-ab0b6f465c3c
* Foo, K. (2021, December 13). Multi-Armed Bandit Analysis of Thompson Sampling Algorithm. Medium. https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-thompson-sampling-algorithm-6375271f40d1



