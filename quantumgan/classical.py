"""Classical GANs with the hyperparameters specified in the paper for
comparison."""
import jax
import jax.random as jr
from jaxtyping import Float, Array
import equinox as eqx
from quantumgan.gan import GAN
from typing import Sequence


class BarClassicalGAN(GAN):
    """The common parent class for all classical Bar GANs."""

    def __init__(
        self,
        key: jr.PRNGKeyArray,
        latent_shape: Sequence[int],
        gen_params: Sequence[eqx.Module],
        dis_hidden: Sequence[int] = (5, 2),
    ):
        """Create a classical Bar GAN.

        Args:
          latent_shape: The shape of the latent space input.
          gen_params: The architecture of the generator.
          dis_hidden: A length-2 sequence of the number of neurons in the
            discriminators two hidden layers.
        """
        key1, key2, key3 = jr.split(key, 3)

        self.latent_shape = latent_shape
        self.gen_params = gen_params

        self.dis_params = [
            eqx.nn.Linear(4, dis_hidden[0], key=key1),
            jax.nn.relu,
            eqx.nn.Linear(dis_hidden[0], dis_hidden[1], key=key2),
            jax.nn.relu,
            eqx.nn.Linear(dis_hidden[1], 1, key=key3),
            jax.nn.sigmoid,
        ]

    def random_latent(
        self, key: jr.PRNGKeyArray, batch: int
    ) -> Float[Array, "batch latent"]:
        return jr.normal(key, (batch,) + self.latent_shape)

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
        def f(x):
            for layer in self.gen_params:
                x = layer(x)
            return x

        return jax.vmap(f)(latent)


class BarMLPGAN(BarClassicalGAN):
    """Our best guess for the multi-layer perceptron GAN described in figure
    10a.

    Has a batch size of 1.
    """

    def __init__(
        self,
        key: jr.PRNGKeyArray,
        latent_shape: int = 2,
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

        key1, key2, key3 = jr.split(key, 3)
        gen_params = [
            eqx.nn.Linear(latent_shape, gen_hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(gen_hidden, 4, key=key2),
            jax.nn.softmax,  # Same normalization (sum=1) as batch GAN
        ]

        super(BarMLPGAN, self).__init__(
            key3, (latent_shape,), gen_params, dis_hidden
        )


class BarCNNGAN(BarClassicalGAN):
    """Our best guess for the CNN GAN described in figure 10b.

    Has a batch size of 1.
    """

    def __init__(
        self,
        key: jr.PRNGKeyArray,
        channels_hidden: int = 2,
        dis_hidden: Sequence[int] = (5, 2),
    ):
        """Create a BarCNNGAN.

        Args:
          channels_hidden: Number of channels in the hidden layer
          dis_hidden: A length-2 sequence of the number of neurons in the
            discriminators two hidden layers.
        """

        key1, key2, key3 = jr.split(key, 3)

        """
            Archtecture based on the following paragraph from the paper. 
            We assume no padding.
                In the generator of GAN CNN [Fig. 10(b)], the convolutional 
                kernels with shape "(1 x 2)" and "(2 x 1)" are applied to 
                the input noise and hidden features, respectively. Giving 
                a sampled noised as input, the CNN generator can directly 
                output a 2 x 2 gray-scale bar image.
        """
        in_channels = 2
        gen_params = [
            eqx.nn.ConvTranspose2d(in_channels, channels_hidden, (1, 2), key=key1),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(channels_hidden, 1, (2, 1), key=key2),
            jax.numpy.ravel,
            jax.nn.softmax,  # Same normalization (sum=1) as batch GAN
        ]

        super(BarCNNGAN, self).__init__(
            key3, (in_channels, 1, 1), gen_params, dis_hidden
        )
