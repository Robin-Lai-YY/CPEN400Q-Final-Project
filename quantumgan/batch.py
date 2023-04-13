"""The 'batch' strategy from 'Experimental Quantum GANs'.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKeyArray
from jaxtyping import Float, Array
import pennylane as qml
from pennylane.operation import Operation

from quantumgan.gan import GAN
from quantumgan.mpqc import MPQC, EntanglerLayer, StaircaseEntangler


class BatchGAN(GAN):
    gen_params: Float[Array, "layers gen_qubits"]
    dis_params: Float[Array, "layers dis_qubits"]

    trainable: Operation
    entangler: Operation
    qdev: qml.Device

    index_reg: tuple[str, ...]
    gen_ancillary: tuple[str, ...]
    dis_ancillary: tuple[str, ...]
    feature_reg: tuple[str, ...]

    qnode_train_fake: qml.QNode
    qnode_train_real: qml.QNode
    mpqc: MPQC

    def __init__(
        self,
        features_dim: int,
        minibatch_size: int,
        gen_params: Float[Array, "layers gen_qubits"],
        dis_params: Float[Array, "layers dis_qubits"],
        trainable: Operation = qml.RY,
        entangler: EntanglerLayer = StaircaseEntangler(),
    ):
        """Create and configure a batch GAN.

        Args:
          features_dim: Training examples and the generated data will be
            vectors of reals of this size.
          minibatch_size: A setting > 1 introduces index register qubits,
            allowing training to happen on more than one training example at a
            time.
          gen_params: The initial parameters for the generator.
        """
        super().__init__(gen_params, dis_params)
        assert is_p2(features_dim), "Feature dimension must be a power of 2"
        assert is_p2(minibatch_size), "Minibatch size must be a power of 2"

        self.gen_params = gen_params
        self.dis_params = dis_params

        self.trainable = trainable
        self.entangler = entangler

        def format_wires(name: str, num: int):
            return tuple(name + str(i) for i in range(num))

        n_features = int(jnp.log2(features_dim))
        gen_ancillary = gen_params.shape[1] - n_features
        dis_ancillary = dis_params.shape[1] - n_features

        self.index_reg = format_wires("i", int(jnp.log2(minibatch_size)))
        self.gen_ancillary = format_wires("ag", gen_ancillary)
        self.feature_reg = format_wires("f", int(jnp.log2(features_dim)))
        self.dis_ancillary = format_wires("ad", dis_ancillary)

        wires = (
            self.gen_ancillary
            + self.dis_ancillary
            + self.index_reg
            + self.feature_reg
        )

        self.qdev = qml.device("default.qubit", wires=wires)
        self.qnode_train_fake = qml.QNode(
            self._circuit_train_fake, self.qdev, interface="jax"
        )
        self.qnode_train_real = qml.QNode(
            self._circuit_train_real, self.qdev, interface="jax"
        )

        self.mpqc = MPQC(trainable, entangler)

    # See the docstrings for GAN for these overriden methods:
    def random_latent(
        self, key: PRNGKeyArray, batch: int
    ) -> Float[Array, "batch latent"]:
        size = (
            len(self.gen_ancillary)
            + len(self.feature_reg)
            + len(self.index_reg)
        )
        return jr.uniform(key, (batch, size), minval=0, maxval=jnp.pi / 2)

    def train_fake(
        self, latent: Float[Array, "batch latent"]
    ) -> Float[Array, " batch"]:
        return jax.vmap(lambda x: self._measure(self.qnode_train_fake(x)))(
            latent
        )

    def train_real(
        self, features: Float[Array, "batch minibatch feature"]
    ) -> Float[Array, " batch"]:
        return jax.vmap(lambda x: self._measure(self.qnode_train_real(x)))(
            features
        )

    def _measure(self, probs: Float[Array, " probs"]):
        """Postselect for ancillary bits all being 0 and return a probability.

        All the ancillary bits are first in the device setup, so we can throw
        away all probs other than the first 2^n, where n is the number of
        qubits NOT being postselected for.
        """
        n = 2 ** (len(self.index_reg) + len(self.feature_reg))
        probs = probs[0:n]
        # Add up all probabilities for the final bit of the discriminator
        # output being 1.
        return jnp.sum(probs[0::2]) / jnp.sum(probs)

    def _circuit_train_fake(self, latent: Float[Array, " latent"]):
        self._circuit_gen(latent)
        self._circuit_dis()
        return qml.probs()

    def _circuit_train_real(
        self, features: Float[Array, "minibatch feature"]
    ):
        embedding_wires = self.index_reg + self.feature_reg
        features_normalized = features / jnp.sum(
            features, axis=1, keepdims=True
        )
        # Because the index register comes first, this put the first training
        # example into the amplitudes where i=0, the second where i=1, and so
        # on.
        features_flatten = features_normalized.reshape(
            2 ** len(embedding_wires)
        )
        qml.AmplitudeEmbedding(jnp.sqrt(features_flatten), embedding_wires)
        self._circuit_dis()
        return qml.probs()

    def _circuit_gen(self, latent: Float[Array, " latent"]):
        wires = self.gen_ancillary + self.index_reg + self.feature_reg
        qml.AngleEmbedding(latent, wires, rotation="Y")
        self.mpqc(self.gen_params, self.gen_ancillary + self.feature_reg)

    def _circuit_dis(self):
        self.mpqc(self.dis_params, self.dis_ancillary + self.feature_reg)

    @staticmethod
    def init_params(
        key: PRNGKeyArray,
        features_dim: int,
        gen_layers: int,
        gen_ancillary: int,
        dis_layers: int,
        dis_ancillary: int,
    ) -> tuple[
        Float[Array, "layers gen_qubits"], Float[Array, "layers dis_qubits"]
    ]:
        """Generate some suitable initial paramters for a batch GAN using
        rotation operators as the trainables.

        Args:
          key: A JAX PRNG key to ensure determinism.
          features_dim: The total number of features (log2(n) feature register
            qubits).
          gen_layers: The number of quantum generator layers.
          gen_ancillary: The number of ancillary qubits for the generator.
          dis_layers: The number of quantum discriminator layers.
          dis_ancillary: The number of ancillary qubits for the discriminator.

        Returns:
          A tuple (gen_params, dis_params)
        """
        feature_bits = int(jnp.log2(features_dim))
        gen_key, dis_key = jr.split(key)
        gen_params = jr.uniform(
            gen_key,
            (gen_layers, gen_ancillary + feature_bits),
            minval=0,
            maxval=jnp.pi,
        )
        dis_params = jr.uniform(
            dis_key,
            (dis_layers, dis_ancillary + feature_bits),
            minval=0,
            maxval=jnp.pi,
        )
        return (gen_params, dis_params)


def is_p2(x: int) -> bool:
    """Returns true if the argument is a power of 2."""
    return jnp.log2(x) == int(jnp.log2(x))
