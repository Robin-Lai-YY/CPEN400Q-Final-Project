# CPEN 400Q final project: Experimental Quantum GANs

![Batch GAN circuit](https://user-images.githubusercontent.com/33556084/232192493-dd3f1fc5-7bc6-494a-beae-6ade8cb9bd27.png)

This repository contains software for reproducing some of the result from
[Experimental Quantum Generative Adversarial Networks for Image Generation
](https://arxiv.org/abs/2010.06201) by He-Liang Huang et. al.

Links:
- [Companion report](report/report.pdf)
- [Presentation](report/presentation.pdf)
- [Demo notebook](demo/talk-demo.ipynb)
  This also makes a good quickstart for learning the library API.

## Quickstart

If you use [Poetry](https://python-poetry.org/) already, getting started is
easy:

```
$ poetry install
$ poetry run jupyterlab  # Run the demo notebooks
$ poetry run python scripts/figures.py  # Reproduce the report figures
$ poetry run python scripts/figures_index.py  # Reproduce the report figures
```

If you have a lot of CPU cores and time, you can also generate the bar plots 
(the `--noise` flag is optional; it depends on a device that does not support JAX so it is very slow):
```
$ poetry run python scripts/plots.py --noise # Reproduce the report bar plots
```

If your python environment looks like [this](https://xkcd.com/1987/), a container
image can be used to run Jupyter:

```
$ docker build -t quantumgan . && \
    docker run -it --rm -p 8888:8888 quantumgan poetry run jupyter lab --allow-root --ip 0.0.0.0 --no-browser
```

## Software

The (hopefully) reusable components are in the `quantumgan` modules defined by
this package.  A brief overview:
- `mpqc`: Multi-layer parameterized quantum circuits
- `gan`: The `GAN` interface.  The `batch` and `classical` modules implement
  this interface so the same code can be used to train them.
- `train`: Training routines for any `GAN` implementation.
- `datasets`: Artificial test datasets, tools for evaluating GAN output quality.
- `batch`: The "batch GAN" strategy from "Experimental Quantum GANs".
- `classical`: Classical GANs for the performance comparisons

Public APIs have docstrings, while private methods start with `_` and do not.

The scripts used for generating the data in the report are in `scripts/`.  These
are not part of the public API, but might be interesting regardless, as examples
of how the library can be used.

### Development notes

Before committing, remember to follow these steps:
- `poetry check`: ensure `pyproject.toml` and the lock file are in a good state.
- `poetry run black quantumgan/`: format the code.
- `poetry run flake8 quantumgan/`: run the linter.
- `poetry run pytype quantumgan/`: run the static type checker.

# Software Limitations
We encountered the following software limitation:
- We found that `jax.vmap` is incompatible with the noise model in Qiskit Aer. As
a workaround, we provide an option (as a parameter in the 
[`__init__`](quantumgan/batch.py) function) to use a regular loop to 
replace `jax.vmap` when running the circuit on a noisy device.

# Contributions

- Yuyou Lai
  - Implemented classical GAN in python to understand how the networks work.
  - Implemented quantum GAN (the patch method) in pennylane with teammates (reference: https://pennylane.ai/qml/demos/tutorial_quantum_gans.html)
  - Modified some code in figures.py to generate  plots to visualize the loss over time with different index qubit.
  - Helped debugging
  - Wrote parts of Theory section of the Doc
- Juntong Luo
  - Added index qubits to the batch GAN.
  - Modified the batch GAN to run on noisy devices and IBM Quantum's 
    real hardware.
  - Implemented the classical GAN with CNN generator, and worked on figuring 
    out the architecture of the classical GANs in the paper.
  - Worked on scripts for generating some of the plots in the Results section 
    and wrote parts of the Software and the Reproducibility section.
  - Helped debugging the batch GAN and the Fréchet distance.
- Sam Schweigel
  - Wrote the initial JAX implementation of the [batch GAN](quantumgan/batch.py).
  - Abstracted out the [GAN interface](quantumgan/gan.py) and made the [generic
  training code](quantumgan/train.py).
  - Wrote the code to evaluate the GAN's performance using the [Fréchet
    distance](quantumgan/datasets.py).
  - Wrote about some results, batch GAN theory, and created the quantkiz and
    [other diagrams](scripts/figures.py) in the report.
  - Created some code in the [hyperparameter search](scripts/plots.py) script to
    do it in parallel and persistently cache the results.
- Bolong Tan 
  - Wrote the Intoduction of the report.
  - Double check the whole report follows the requirement.
  - Helped debugging.
