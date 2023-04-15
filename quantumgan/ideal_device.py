from pennylane.devices.default_qubit_jax import DefaultQubitJax


class IdealDeviceJax(DefaultQubitJax):
    """An ideal quantum device for JAX."""

    def __init__(self, wires):
        """Create an ideal quantum device for JAX.

        Args:
          wires: the wires in the device.
        """
        super(IdealDeviceJax, self).__init__(wires, shots=None)
