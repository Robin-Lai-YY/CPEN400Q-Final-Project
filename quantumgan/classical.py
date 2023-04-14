"""Classical GANs with the hyperparameters specified in the paper for
comparison."""
import jax
import jax.random as jr
from jaxtyping import Float, Array
import equinox as eqx
from quantumgan.gan import GAN
from typing import Sequence


class BarMLPGAN(GAN):
    """Our best guess for the multi-layer perceptron GAN described in figure
    10.

    Has a batch size of 1.
    """

    def __init__(
        self,
        key: jr.PRNGKeyArray,
        gen_hidden: int = 5,
        dis_hidden: Sequence[int] = (5, 2),
    ):
        """Create a BarMLPGAN.

        Args:
          gen_hidden: The number of neurons in the generator's single hidden
            layer.
          dis_hidden: A length-2 sequence of the number of neurons in the
            discriminators two hidden layers.
        """
        key1, key2, key3, key4, key5 = jr.split(key, 5)

        self.gen_params = [
            eqx.nn.Linear(2, gen_hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(gen_hidden, 4, key=key2),
            jax.nn.softmax,  # Same normalization (sum=1) as batch GAN
        ]

        self.dis_params = [
            eqx.nn.Linear(4, dis_hidden[0], key=key3),
            jax.nn.relu,
            eqx.nn.Linear(dis_hidden[0], dis_hidden[1], key=key4),
            jax.nn.relu,
            eqx.nn.Linear(dis_hidden[1], 1, key=key5),
            jax.nn.sigmoid,
        ]

    def random_latent(
        self, key: jr.PRNGKeyArray, batch: int
    ) -> Float[Array, "batch latent"]:
        return jr.normal(key, (batch, 2))

    def train_fake(
        self, latent: Float[Array, "batch latent"]
    ) -> Float[Array, ""]:
        x = latent[0]
        for layer in self.gen_params:
            x = layer(x)
        y = x
        for layer in self.dis_params:
            y = layer(y)
        return y[0]

    def train_real(
        self, features: Float[Array, "batch feature"]
    ) -> Float[Array, ""]:
        x = features[0]
        for layer in self.dis_params:
            x = layer(x)
        return x[0]

    def generate(
        self, latent: Float[Array, "batch latent"]
    ) -> Float[Array, "batch feature"]:
        x = latent[0]
        for layer in self.gen_params:
            x = layer(x)
        return x
