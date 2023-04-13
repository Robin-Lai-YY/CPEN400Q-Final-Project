"""Differentiable multi-layer parameterized quantum circuits for quantum
machine learning.
"""
import jax.numpy as jnp
import jax.random as jr
import pennylane as qml
from pennylane.operation import Operation
from jaxtyping import Array, Float
from typing import Sequence
from abc import abstractmethod, ABCMeta


class EntanglerLayer(metaclass=ABCMeta):
    """Base class for MPQC entangler layers (allows us to thread random keys
    through easily).
    """

    @abstractmethod
    def __call__(self, _layer: int, _wires: Sequence[int]):
        """Apply the entangler layer (call in a QNode).

        Args:
          layer: The layer number (possibly used for deterministic PRNG use).
          wires: Sequence of wires to apply the layer to.
        """
        raise NotImplementedError


class StaircaseEntangler(EntanglerLayer):
    """MPQC entangler layer using a layout identical to the one from
    'Experimental Quantum GANs': each qubit is entangled to the next in the
    sequence with the controlled gate.

    Attributes:
      entangler: Controlled gate to apply to entangle qubits.
    """

    def __init__(self, entangler: Operation = qml.CZ):
        self.entangler = entangler

    def __call__(self, _layer: int, wires: Sequence[int]):
        for control, target in zip(wires, wires[1:]):
            self.entangler((control, target))


class RandomEntangler(EntanglerLayer):
    """MPQC entangler layer using a pseudorandom layout.  Every qubit will be
    connected to an entangler control and target, but the order will be
    different across layers.

    Attributes:
      key: PRNG key controlling the layout.
      entangler: Controlled gate to apply to entangle qubits.
    """

    def __init__(self, key: jr.PRNGKeyArray, entangler: Operation = qml.CZ):
        self.key = key
        self.entangler = entangler

    def __call__(self, layer: int, wires: Sequence[int]):
        k = jr.fold_in(self.key, layer)
        wires_permuted = jr.permutation(k, jnp.arange(len(wires)))
        for control, target in zip(wires_permuted, wires_permuted[1:]):
            self.entangler((control, target))


class MPQC:
    """A multi-layer parameterzed quantum circuit.

    Attributes:
      trainable: A single-qubit parameterized operation, applied to all qubits
        in each layer.
      entangler: An EntanglerLayer, to be applied after each trainable layer.
    """

    def __init__(
        self,
        trainable: Operation = qml.RY,
        entangler: EntanglerLayer = StaircaseEntangler(),
    ):
        self.trainable = trainable
        self.entangler = entangler

    def __call__(
        self, weights: Float[Array, "layer qubit"], wires: Sequence[int]
    ):
        """Apply the MPQC in a QNode.

        Args:
          weights: An array of trainable parameters of shape (number of layers,
            number of qubit).  Determines the total number of layers.
          wires: Sequence of wires to apply the MPQC to.
        """
        for i, layer in enumerate(weights):
            for r, w in zip(layer, wires):
                self.trainable(r, w)
            self.entangler(i, wires)
