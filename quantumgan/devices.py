"""Devices used in the paper.
"""
import penneylane as qml
from qml.devices.default_qubit_jax import DefaultQubitJax
from pennylane_qiskit import AerDevice
from qiskit_aer.noise import NoiseModel


class IdealDeviceJax(DefaultQubitJax):
    """An ideal quantum device for JAX."""
    def __init__(self, wires):
        super(IdealDeviceJax, self).__init__(wires, shots=None)

class NoisyDevice(AerDevice):
    """A noisy device simulating the superconducting quantum processor in the paper."""
    def __init__(self, wires):
        noise_model = noise.NoiseModel()
        # Add Noise Model here
        super(NoisyDevice, self).__init__(wires, shots=3000, noise_model=noise_model, backend="aer_simulator")
