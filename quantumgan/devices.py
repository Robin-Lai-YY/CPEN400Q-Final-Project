"""Devices that are worth trying out.
"""
import qiskit_aer.noise as noise
from pennylane.devices.default_qubit_jax import DefaultQubitJax
from pennylane_qiskit import AerDevice, IBMQDevice


class IdealDeviceJax(DefaultQubitJax):
    """An ideal quantum device for JAX."""

    def __init__(self, wires):
        """Create an ideal quantum device for JAX.

        Args:
          wires: the wires in the device.
        """
        super(IdealDeviceJax, self).__init__(wires, shots=None)


class IbmDevice(IBMQDevice):
    """A device for the remote IBMQ API."""

    def __init__(self, wires, backend="ibmq_belem", ibmqx_token=None):
        """Create an ideal quantum device for JAX.

        Args:
          wires: the wires in the device.
          backend: name of the IBMQ compute resource
          ibmqx_token: your account's IBMQ quantum API token
        """
        super(IbmDevice, self).__init__(
            wires, shots=3000, backend=backend, ibmqx_token=ibmqx_token
        )


class NoisyDevice(AerDevice):
    """A noisy device simulating the superconducting quantum
    processor used by the paper.
    """

    def __init__(self, wires):
        """Create a noisy device simulating the superconducting
        quantum processor used by the paper.

        Args:
          wires: the wires in the device.
        """
        super(NoisyDevice, self).__init__(
            wires,
            shots=3000,
            noise_model=NoisyDevice.getNoiseModel(),
            backend="aer_simulator",
        )

    @staticmethod
    def getNoiseModel():
        """Get the noise model of the superconducting
        quantum processor.

        Returns:
          A NoiseModel that attempts to model the superconducting
        quantum processor.
        """
        # Some data from the paper.
        T1_relaxation = 35.4 * 1000.0  # average, in nanoseconds
        T2_dephasing = 4.2 * 1000.0  # average, in nanoseconds
        f_00 = 0.964  # average
        f_11 = 0.905  # average
        # X/2 gate fidelity: 0.9994 (average)
        # CZ gate fidelity: 0.985 (average)

        # The paper does not provide the gate time, so we picked
        # one that is hopefully reasonable for a superconducting
        # quantum processor.
        T_single_gate = 130  # in nanoseconds

        relaxation_error = noise.thermal_relaxation_error(
            T1_relaxation, T2_dephasing, T_single_gate
        )
        readout_error = noise.ReadoutError(
            [[f_00, 1 - f_00], [1 - f_11, f_11]]
        )

        m = noise.NoiseModel()
        m.add_basis_gates("ry")
        m.add_all_qubit_quantum_error(relaxation_error, ["ry"])
        m.add_all_qubit_readout_error(readout_error)
        return m
