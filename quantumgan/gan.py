"""Interface to GANs, classical and quantum.
"""
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float, PyTree
import equinox as eqx
from abc import abstractmethod, ABCMeta


class GAN(eqx.Module, metaclass=ABCMeta):
    """A generic generative adversarial network.  We provide these methods
    because the intermediate state of a quantum batch GAN cannot be used
    separately while training with generated data.

    The public parameter attributes are used to filter which parameters we take
    gradients with respect to.  (See Equinox's filter_grad).

    Attributes:
      gen_params: PyTree of parameters for the generator.
      dis_params: PyTree of parameters for the discriminator.
    """

    gen_params: PyTree
    dis_params: PyTree

    @abstractmethod
    def random_latent(
        self, key: PRNGKeyArray, batch: int
    ) -> Float[Array, "batch latent"]:
        """Generate a random latent space vector.

        Args:
          key: A JAX PRNG key (determinism).
          batch: The batch size, which will be the number of rows.

        Returns:
          An array of random vector (rows) in the latent space of the GAN.
        """
        raise NotImplementedError

    @abstractmethod
    def train_fake(
        self, latent: Float[Array, "batch latent"]
    ) -> Float[Array, " batch"]:
        """Generate an image from the given latent space vector, feed it to the
        discriminator, and compute a probability that the discriminator thinks
        the data is real (0.0 -> fake, 1.0 -> real).

        Args:
          latent: An array of vectors in the GAN latent space (generated by
            random_latent).

        Returns:
          Probabilities from 0 to 1.
        """
        raise NotImplementedError

    @abstractmethod
    def train_real(
        self, features: Float[Array, "batch minibatch feature"]
    ) -> Float[Array, " batch"]:
        """Compute a probability that the discriminator thinks a training
        example is real (0.0 -> fake, 1.0 -> real).

        Args:
          features: An array of feature vectors draw from the training set.
            The minibatch dimension groups examples that return a joint
            probability.

        Returns:
          Probabilities from 0 to 1.
        """
        raise NotImplementedError
