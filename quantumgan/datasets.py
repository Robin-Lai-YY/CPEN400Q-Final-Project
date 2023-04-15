"""Utilities for generating data from known distributions.
"""
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.linalg import sqrtm
from jaxtyping import Array, Float


def generate_grayscale_bar(
    key: jr.PRNGKeyArray, n: int
) -> Float[Array, "n 4"]:
    """Sample the 2x2 grayscale bar distribution from the paper.

    By convention, we use row-major order for the pixels:
    x[0] ~ unif(0.4, 0.6)
    x[1] = 0
    x[2] = 1 - x[0]
    x[3] = 0

    Args:
      key: The input PRNG key.
      n: Number of samples to draw.

    Returns:
      The (n,4) array of samples.
    """
    noise = jr.uniform(key, (n,), minval=0.4, maxval=0.6)
    d = jnp.stack((noise, jnp.zeros(n), 1 - noise, jnp.zeros(n)), axis=1)
    return d


def frechet_distance(S1: Float[Array, "n v"], S2: Float[Array, "m v"]):
    """Estimate the Fréchet distance (2-Wasserstein) between two
    multidimensional Gaussian distributions (on R^v), given many (>1000)
    samples from each.

    Note that this only works if the distributions can be reasonably
    approximated by a multidimensional gaussian.  We would otherwise need to
    use an optimal transport solver.

    References:
      [1] https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance
      [2] https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance

    Args:
      S1: Distribution 1 (n samples, v variables).
      S2: Distribution 2 (m samples, v variables).
    """

    mu1, mu2 = jnp.mean(S1, axis=0), jnp.mean(S2, axis=0)
    sigma1, sigma2 = jnp.cov(S1, rowvar=False), jnp.cov(S2, rowvar=False)

    sigma1_sqrt = sqrtm(sigma1)
    M = sigma1 + sigma2 - 2 * sqrtm(sigma1_sqrt @ sigma2 @ sigma1_sqrt)
    dist = jnp.dot(mu1 - mu2, mu1 - mu2) + jnp.trace(M)

    return jnp.abs(dist)
