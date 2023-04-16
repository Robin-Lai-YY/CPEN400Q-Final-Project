"""The 'batch' strategy from 'Experimental Quantum GANs'.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKeyArray
from jaxtyping import Float, Array
import equinox as eqx
import pennylane as qml
from pennylane.operation import Operation

from quantumgan.gan import GAN
from quantumgan.mpqc import MPQC, EntanglerLayer, StaircaseEntangler
from quantumgan.ideal_device import IdealDeviceJax


class BatchGAN(GAN):
    """Batch GANs: quantum generators AND discriminators.

    The batch strategy is useful when the entire feature vector fits onto our
    quantum device.  We can use a batch_size > 1 with an index register to
    train entire batches of examples at once.
    """

    gen_params: Float[Array, "layers gen_qubits"]
    dis_params: Float[Array, "layers dis_qubits"]

    _qdev: qml.Device = eqx.static_field(repr=False)

    _index_reg: tuple[str, ...] = eqx.static_field(repr=False, compare=False)
    _gen_ancillary: tuple[str, ...] = eqx.static_field(
        repr=False, compare=False
    )
    _dis_ancillary: tuple[str, ...] = eqx.static_field(
        repr=False, compare=False
    )
    _feature_reg: tuple[str, ...] = eqx.static_field(repr=False, compare=False)
    _mpqc: MPQC = eqx.static_field(repr=False)

    _qnode_train_fake: qml.QNode = eqx.static_field(repr=False, compare=False)
    _qnode_train_real: qml.QNode = eqx.static_field(repr=False, compare=False)
    _qnode_generate: qml.QNode = eqx.static_field(repr=False, compare=False)

    _disable_jax_vmap: bool = eqx.static_field(repr=False, compare=False)

    def __init__(
        self,
        features_dim: int,
        batch_size: int,
        gen_params: Float[Array, "layers gen_qubits"],
        dis_params: Float[Array, "layers dis_qubits"],
        trainable: Operation = qml.RY,
        entangler: EntanglerLayer = StaircaseEntangler(),
        device: qml.Device = IdealDeviceJax,
        diff_method: str = "best",
        disable_jax_vmap: bool = False,
    ):
        """Create and configure a batch GAN.

        Args:
          features_dim: Training examples and the generated data will be
            vectors of reals of this size.
          batch_size: A setting > 1 introduces index register qubits, allowing
            training to happen on more than one training example at a time.
          gen_params: The initial parameters for the generator.
          dis_params: The initial parameters for the discriminator
          trainable: The trainable unitary
          entangler: The entagler layer
          device: The device to run the circuits
          diff_method: The method of differentiation of QNodes
          disable_jax_vmap: Whether to avoid using jax.vmp, which sometimes has compatibility issues with Qiskit
        """
        assert is_p2(features_dim), "Feature dimension must be a power of 2"
        assert is_p2(batch_size), "Batch size must be a power of 2"

        self.gen_params = gen_params
        self.dis_params = dis_params

        def format_wires(name: str, num: int):
            return tuple(name + str(i) for i in range(num))

        n_features = int(jnp.log2(features_dim))
        _gen_ancillary = gen_params.shape[1] - n_features
        _dis_ancillary = dis_params.shape[1] - n_features

        self._index_reg = format_wires("i", int(jnp.log2(batch_size)))
        self._gen_ancillary = format_wires("ag", _gen_ancillary)
        self._feature_reg = format_wires("f", int(jnp.log2(features_dim)))
        self._dis_ancillary = format_wires("ad", _dis_ancillary)

        wires = (
            self._gen_ancillary
            + self._dis_ancillary
            + self._index_reg
            + self._feature_reg
        )

        latent_shape = (
            len(self._gen_ancillary)
            + len(self._feature_reg)
            + len(self._index_reg),
        )
        super().__init__(latent_shape, gen_params, dis_params)

        self._qdev = device(wires)
        self._qnode_train_fake = qml.QNode(
            # Could produce incorrect results sliently if `interface="jax"` is
            # not included
            self._circuit_train_fake,
            self._qdev,
            interface="jax",
            diff_method=diff_method,
        )
        self._qnode_train_real = qml.QNode(
            # Could produce incorrect results sliently if `interface="jax"` is
            # not included
            self._circuit_train_real,
            self._qdev,
            interface="jax",
            diff_method=diff_method,
        )
        self._qnode_generate = qml.QNode(
            self._circuit_generate, self._qdev, interface="jax"
        )

        self._mpqc = MPQC(trainable, entangler)

        self._disable_jax_vmap = disable_jax_vmap

    # See the docstrings for GAN for these overriden methods:
    def random_latent(
        self, key: PRNGKeyArray, batch: int
    ) -> Float[Array, "batch latent"]:
        size = (
            len(self._gen_ancillary)
            + len(self._feature_reg)
            + len(self._index_reg)
        )
        return jr.uniform(
            key, (batch,) + self.latent_shape, minval=0, maxval=jnp.pi / 2
        )

    def train_fake(
        self, latent: Float[Array, "batch latent"]
    ) -> Float[Array, ""]:
        # To support the GAN interface, we must accept a (batch, latent)-shape
        # latent space array, but we only need one of the latent vectors.
        return self._measure(
            self._qnode_train_fake(self.gen_params, self.dis_params, latent[0])
        )

    def train_real(
        self, features: Float[Array, "batch feature"]
    ) -> Float[Array, ""]:
        return self._measure(self._qnode_train_real(self.dis_params, features))

    def generate(
        self, latent: Float[Array, "batch latent"]
    ) -> Float[Array, "batch feature"]:
        n = 2 ** len(self._feature_reg)

        def f(lvec):
            out = self._qnode_generate(self.gen_params, lvec)[:n]
            return out / jnp.sum(out)

        if not self._disable_jax_vmap:
            return jax.vmap(f)(latent)
        else:
            return jnp.array([f(lvec) for lvec in latent])

    def _measure(self, probs: Float[Array, " probs"]):
        """Postselect for ancillary bits all being 0 and return a probability.

        All the ancillary bits are first in the device setup, so we can throw
        away all probs other than the first 2^n, where n is the number of
        qubits NOT being postselected for.
        """
        n = 2 ** (len(self._index_reg) + len(self._feature_reg))
        probs = probs[0:n]
        # Add up all probabilities for the final bit of the discriminator
        # output being 1.
        return jnp.sum(probs[0::2]) / jnp.sum(probs)

    def _circuit_train_fake(self, gen_params, dis_params, latent):
        self._circuit_gen(gen_params, latent)
        self._circuit_dis(dis_params)
        return qml.probs(self._qdev.wires)

    def _circuit_train_real(self, dis_params, features):
        embedding_wires = self._index_reg + self._feature_reg
        features_normalized = (
            features
            / jnp.sum(features, axis=1, keepdims=True)
            / features.shape[0]
        )
        # Because the index register comes first, this put the first training
        # example into the amplitudes where i=0, the second where i=1, and so
        # on.
        features_flatten = features_normalized.reshape(
            2 ** len(embedding_wires)
        )
        qml.AmplitudeEmbedding(jnp.sqrt(features_flatten), embedding_wires)
        self._circuit_dis(dis_params)
        return qml.probs(self._qdev.wires)

    def _circuit_generate(self, gen_params, latent):
        self._circuit_gen(gen_params, latent)
        return qml.probs()

    def _circuit_gen(self, gen_params, latent):
        wires = self._gen_ancillary + self._index_reg + self._feature_reg
        qml.AngleEmbedding(latent, wires, rotation="Y")
        self._mpqc(gen_params, self._gen_ancillary + self._feature_reg)

    def _circuit_dis(self, dis_params):
        self._mpqc(dis_params, self._dis_ancillary + self._feature_reg)

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
    return jnp.isclose(jnp.log2(x), 2 ** jnp.log2(x))
