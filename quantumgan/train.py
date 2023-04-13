"""Generic training algorithms for GANs.  This is abstracted out here so allow
for the same training routines to compare classical and quantum GANs.
"""
import jax.numpy as jnp
from jaxtyping import Array, Float
from optax import GradientTransformation


def bce_loss(x: Float[Array, " batch"], target: float):
    """Compute the binary cross entropy between x and the target probabilities.

       Args:
         x: Array of input probabilities (entire batch)
         target: Target probability (label)

       Returns:
    cross entropy loss (float).
    """
    return -jnp.mean(
        target * jnp.clip(jnp.log(x), a_min=-100)
        + (1 - target) * jnp.clip(jnp.log(1 - x), a_min=-100)
    )


def train_gan(
    gen_optimizer: GradientTransformation,
    init_gen_params,
    dis_optimizer: GradientTransformation,
    init_dis_params,
    checkpoint_freq: int = 50,
):
    """Train a GAN"""
    checkpoints = []
    g_loss_history = []
    d_loss_history = []

    gen_optimizer.init(init_gen_params)
    dis_optimizer.init(init_dis_params)
