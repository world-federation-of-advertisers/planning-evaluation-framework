"""Sampling algorithms for distributions needed in simulations."""

import math
import typing

import numpy as np
from scipy import stats


def sample_discrete_laplace(param: float, rng: np.random.Generator) -> int:
    """Generate discrete Laplace with given parameter."""
    # Note that we have this custom function because the scipy.stats.dlaplace is
    # somehow very slow.

    # The discrete Laplace random variable can be generated as the difference of
    # two independent geometric random variables.
    p_geometric = 1 - math.exp(-param)
    y1 = rng.geometric(p=p_geometric)
    y2 = rng.geometric(p=p_geometric)
    return y1 - y2


def sample_discrete_gaussian(sigma: float, rng: np.random.Generator) -> int:
    """Samples from the discrete Gaussian random variable with parameter sigma."""
    # Use rejection sampling of discrete Laplace distribution (Algorithm 3 in
    # Canonne et al.) to sample a discrete Gaussian random variable.
    # References:
    # Canonne, Kamath, Steinke. "The Discrete Gaussian for Differential Privacy".
    # In NeurIPS 2020.
    t = math.floor(sigma) + 1

    while True:
        y = sample_discrete_laplace(1 / t, rng)

        sigma_sq = sigma ** 2
        p_bernoulli = math.exp(-((abs(y) - sigma_sq / t) ** 2) * 0.5 / sigma_sq)
        if rng.binomial(1, p_bernoulli) == 1:
            return y


def sample_noise(
    rng: np.random.Generator,
    epsilon: typing.Optional[float] = None,
    num_queries: typing.Optional[int] = None,
    noise_parameter: typing.Optional[float] = None,
    noise_type: str = "dgaussian",
) -> float:
    """Samples noise based on given parameters and the type of noise.

    Args:
      rng: the random generator.
      epsilon: the epsilon parameter in differential privacy.
      num_queries: the number of queries which share the (epsilon, delta) budget.
      noise_parameter: the parameter of the noise to sample from.
      noise_type: there are three types possible; (1) 'dgaussian' for discrete
        Gaussian noise, (2) 'dlaplace' for discrete Laplace noise and (3)
        'dlaplace_basic' for discrete Laplace noise but with basic accounting.
        Furthermore, when set to 'no_noise', no noise will be added (same effect
        as setting epsilon to None). For the case of 'dgaussian' and 'dlaplace',
        noise_parameter must be specified.

    Returns:
      A random noise sampled from the distribution that satisfies (epsilon, delta)
      -differential privacy even after num_queries executions.
    """
    if epsilon is None or noise_type == "no_noise":
        return 0
    if noise_type == "dlaplace_basic":
        # By basic composition, the epsilon is split equally among queries.
        return sample_discrete_laplace(epsilon / num_queries, rng)
    if noise_type == "dlaplace":
        if noise_parameter is None:
            raise ValueError("Noise parameter must be specified for dlaplace.")
        return sample_discrete_laplace(noise_parameter, rng)
    if noise_type == "dgaussian":
        if noise_parameter is None:
            raise ValueError("Noise parameter must be specified for dgaussian.")
        return sample_discrete_gaussian(noise_parameter, rng)
    raise TypeError("Specified noise type not found.")


def noise_sampler_factory(
    rng: np.random.Generator,
    epsilon: typing.Optional[float] = None,
    num_queries: typing.Optional[int] = None,
    noise_parameter: typing.Optional[float] = None,
    noise_type: str = "dgaussian",
) -> typing.Callable[[], float]:
    """Returns a function that samples from the given noise distribution."""

    def sample_noise_no_input() -> float:
        return sample_noise(
            rng,
            epsilon=epsilon,
            num_queries=num_queries,
            noise_parameter=noise_parameter,
            noise_type=noise_type,
        )

    return sample_noise_no_input


def get_population_histogram(max_freq: int, n: int, distribution: str) -> np.ndarray:
    """Generates population histogram for a specified distribution.

    Args:
      max_freq: maximum frequency, above which we snap to the max_freq-th entry.
      n: cardinality.
      distribution: population frequency distribution to generate; can be one of
          (i) 'uniform': the uniform distribution over max_freq buckets,
          (ii) 'zipf': the Zipf distribution with parameter 2 with mass,
          (iii) 'poisson': the Poisson distribution with parameter 5 shifted by
            one.

    Returns:
      The population histogram for the given distribution and cardinality.
    """
    if distribution == "poisson":
        h = stats.poisson.pmf(list(range(0, max_freq - 1)), 5)
    elif distribution == "zipf":
        h = stats.zipf.pmf(list(range(1, max_freq)), 2)
    elif distribution == "uniform":
        h = np.array([1 / max_freq] * (max_freq - 1))
    else:
        raise ValueError(f"Distribution unsupported: {distribution}")
    h = (h * n).astype("int")
    return np.append(h, n - sum(h))


def sample_histogram(
    population_histogram: typing.Sequence[int],
    sample_size: int,
    rng: np.random.Generator,
) -> typing.Sequence[int]:
    """Samples histogram from a population histogram.

    Due to a limitation of rng.multivariate_hypergeometric, this method can be
    very slow if the population histogram is of size more than one billion.

    Args:
      population_histogram: Frequency histogram of all items.
      sample_size: number of active registers.
      rng: the random generator.

    Returns:
      The frequency histogram of the items in the active registers.
    """
    # rng.multivariate_hypergeometric only supports "marginals" method for
    # sampling when the total population is at most 1 billion. Otherwise, we need
    # to use the (slower) count method.
    method = "count" if sum(population_histogram) >= 1_000_000_000 else "marginals"
    return rng.multivariate_hypergeometric(
        population_histogram, sample_size, method=method
    )
