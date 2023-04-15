"""Generic training algorithms for GANs.  This is abstracted out here so allow
for the same training routines to compare classical and quantum GANs.
"""
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
import equinox as eqx
from optax import GradientTransformation
from typing import Callable
from dataclasses import dataclass
from tqdm.auto import tqdm

from quantumgan.gan import GAN


LossFn = Callable[[Float[Array, " batch"], float], float]


def bce_loss(x: Float[Array, " batch"], target: float) -> Float[Array, ""]:
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


@dataclass
class TrainResult:
    checkpoints: list[tuple[int, GAN]]
    g_loss_history: list[int]
    d_loss_history: list[int]


def train_gan(
    key: jr.PRNGKeyArray,
    gan: GAN,
    gen_optimizer: GradientTransformation,
    dis_optimizer: GradientTransformation,
    train_data: Float[Array, "n_batches batch feature"],
    loss_fn: LossFn = bce_loss,
    checkpoint_freq: int = 1000,
    show_progress: bool = False,
    jit: bool = True,
) -> TrainResult:
    """Train a GAN.

    Completely generic routine for training generative adversarial networks.

    Args:
      gan: The GAN (with initial parameters).
      gen_optimizer: An optax optimizer for the generator.
      dis_optimizer: An optax optimizer for the generator.
      train_data: An array of batches of features for real training examples.
        The batch size must be compatible with the GAN's supported batch size
        (e.g. unrestricted for a classical GAN, but fixed for a quantum batch
        GAN).
      loss_fn: Loss function (binary cross entropy by default).  The loss gets
        an array of GAN outputs and a label (0.0 = fake, 1.0 = real).
      checkpoint_freq: Number of iterations between saving a copy of the GAN.
      show_progress: Option to show the process in a tqdm progress bar.
      jit: Option to enable JIT.

    Returns:
      checkpoints: A series of trained GANs.  The last entry is the final,
        trained model.
      d_loss_history: Discriminator loss after each iteration.
      g_loss_history: Generator loss after each iteration.
    """
    checkpoints = []
    g_loss_history = []
    d_loss_history = []

    gen_s = gen_optimizer.init(gan.gen_params)
    dis_s = dis_optimizer.init(gan.dis_params)

    def gen_loss(gen, dis, latent):
        gan = eqx.combine(gen, dis)
        return loss_fn(gan.train_fake(latent), 0.0)

    def dis_loss(dis, gen, latent, examples):
        gan = eqx.combine(gen, dis)
        l1 = loss_fn(gan.train_fake(latent), 1.0)
        l2 = loss_fn(gan.train_real(examples), 0.0)
        return (l1 + l2) / 2

    def step(gan, gen_s, dis_s, latent, examples):
        gen, dis = eqx.partition(gan, gan.gen_filter())
        g_loss, g_grad = eqx.filter_value_and_grad(gen_loss)(gen, dis, latent)
        g_updates, gen_s = gen_optimizer.update(g_grad, gen_s, gan)

        dis, gen = eqx.partition(gan, gan.dis_filter())
        d_loss, d_grad = eqx.filter_value_and_grad(dis_loss)(
            dis, gen, latent, examples
        )
        d_updates, dis_s = dis_optimizer.update(d_grad, dis_s, gan)

        gan = eqx.apply_updates(eqx.apply_updates(gan, g_updates), d_updates)

        return gan, gen_s, dis_s, g_loss, d_loss

    if jit:
        step = eqx.filter_jit(step)

    progress = None
    if show_progress:
        progress = tqdm(train_data)
    train_iter = progress if show_progress else train_data
    for i, example in enumerate(train_iter):
        if checkpoint_freq > 0 and i % checkpoint_freq == 0:
            checkpoints.append((i, gan))

        key, latent_key = jr.split(key)
        latent = gan.random_latent(latent_key, example.shape[0])
        gan, gen_s, dis_s, g_loss, d_loss = step(
            gan, gen_s, dis_s, latent, example
        )

        g_loss_history.append(g_loss.item())
        d_loss_history.append(d_loss.item())

        if progress is not None and i % 100 == 0:
            progress.set_postfix({"g": f"{g_loss:.3f}", "d": f"{d_loss:.3f}"})

    checkpoints.append((i + 1, gan))

    return TrainResult(checkpoints, g_loss_history, d_loss_history)
