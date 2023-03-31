"""bar_gan.generate(gen_params, gen_key, 8).reshape(2,4,2,2)
Library code for "Experimental Quantum Generative Adversarial Networks for
Image Generation"
"""
import pennylane as qml
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pennylane.operation import Operation
from jaxtyping import Array, Float, PyTree


def multilayer_pqc(
    weights: Float[Array, "layer qubit"],
    wires,
    rotation: Operation = qml.RY,
    entangle: Operation = qml.CZ,
    layout: str = "staircase",
):
    np.random.seed(12345)       # hack
    for layer in weights:
        for r, w in zip(layer, wires):
            rotation(r, w)
        if layout == "staircase":
            for control, target in zip(wires, wires[1:]):
                entangle((control, target))
        elif layout == "random":
            idxs = np.random.permutation(len(wires))
            wires_permuted = [wires[i] for i in idxs]
            for control, target in zip(wires_permuted, wires_permuted[1:]):
                entangle((control, target))


def format_wires(name: str, num: int):
    return [name + str(i) for i in range(num)]


def bce_loss(x, y):
    return -jnp.mean(
        y * jnp.clip(jnp.log(x), a_min=-100)
        + (1 - y) * jnp.clip(jnp.log(1 - x), a_min=-100)
    )


class BatchGAN:
    def __init__(
        self,
        features: int,
        gen_ancillary: int,
        gen_layers: int,
        dis_ancillary: int,
        dis_layers: int,
        rotations=qml.RY,
        entanglers=qml.CZ,
        layout="staircase",
    ):
        assert jnp.log2(features) == int(
            jnp.log2(features)
        ), "feature dimension must be a power of 2"

        self.gen_layers = gen_layers
        self.dis_layers = dis_layers
        self.postselect_probs = features

        self.rotations = rotations
        self.entanglers = entanglers
        self.layout = layout

        self.gen_ancillary = format_wires("ag", gen_ancillary)
        self.dis_ancillary = format_wires("ad", dis_ancillary)
        self.feature_reg = format_wires("f", int(jnp.log2(features)))
        wires = self.gen_ancillary + self.dis_ancillary + self.feature_reg

        self.qdev = qml.device("default.qubit", wires=wires)
        self.qnode_train_fake = qml.QNode(
            self.circuit_train_fake, self.qdev, interface="jax"
        )
        self.qnode_train_real = qml.QNode(
            self.circuit_train_real, self.qdev, interface="jax"
        )
        self.qnode_gen = qml.QNode(self.circuit_generate, self.qdev, interface="jax")

    def init_params(self, key):
        gen_key, dis_key = jax.random.split(key)
        gen_params = jax.random.uniform(
            gen_key,
            (self.gen_layers, len(self.gen_ancillary) + len(self.feature_reg)),
            jnp.float32,
            0,
            jnp.pi,
        )
        dis_params = jax.random.uniform(
            dis_key,
            (self.dis_layers, len(self.dis_ancillary) + len(self.feature_reg)),
            jnp.float32,
            0,
            jnp.pi,
        )
        return gen_params, dis_params

    def gen_latent(self, key, batch):
        return jax.random.uniform(
            key,
            (batch, len(self.gen_ancillary) + len(self.feature_reg)),
            jnp.float32,
            0,
            jnp.pi / 2,
        )

    def generate(self, gen_params, key, batch):
        latent = self.gen_latent(key, batch)
        probs = jax.vmap(lambda x: self.qnode_gen(gen_params, x))(latent)
        probs = probs[:, 0 : self.postselect_probs]
        return jax.vmap(lambda p: p / jnp.sum(p))(probs)

    def postselect(self, probs):
        # Postselect for ancillary bits all being 0
        probs = probs[0 : self.postselect_probs]
        # Add up all probabilities for the final bit of the discriminator output being 1
        return jnp.sum(probs[0::2]) / jnp.sum(probs)

    def predict(self, gen_params, dis_params, latent, example):
        def batch(f):
            return jax.vmap(lambda x: self.postselect(f(gen_params, dis_params, x)))

        d_real = batch(self.qnode_train_real)(example)
        d_fake = batch(self.qnode_train_fake)(latent)
        return d_real, d_fake

    def dis_loss(self, dis_params, gen_params, latent, example):
        d_real, d_fake = self.predict(gen_params, dis_params, latent, example)
        return (bce_loss(d_real, 0.0) + bce_loss(d_fake, 1.0)) / 2

    def gen_loss(self, gen_params, dis_params, latent, example):
        _, d_fake = self.predict(gen_params, dis_params, latent, example)
        return bce_loss(d_fake, 0.0)

    def circuit_gen(self, gen_params, latent):
        qml.AngleEmbedding(latent, self.gen_ancillary + self.feature_reg, rotation="Y")
        multilayer_pqc(
            gen_params,
            self.gen_ancillary + self.feature_reg,
            self.rotations,
            self.entanglers,
            self.layout,
        )

    def circuit_dis(self, dis_params):
        multilayer_pqc(
            dis_params,
            self.dis_ancillary + self.feature_reg,
            self.rotations,
            self.entanglers,
            self.layout,
        )

    def circuit_train_fake(self, gen_params, dis_params, latent):
        self.circuit_gen(gen_params, latent)
        self.circuit_dis(dis_params)
        return qml.probs()

    def circuit_train_real(self, gen_params, dis_params, features):
        qml.AmplitudeEmbedding(jnp.sqrt(features / jnp.sum(features)), self.feature_reg)
        self.circuit_dis(dis_params)
        return qml.probs()

    def circuit_generate(self, gen_params, latent):
        self.circuit_gen(gen_params, latent)
        return qml.probs()

    def draw(self, gen_params, dis_params):
        return qml.draw_mpl(self.qnode_train_fake)(
            gen_params,
            dis_params,
            jnp.zeros(len(self.gen_ancillary) + len(self.feature_reg)),
        )
