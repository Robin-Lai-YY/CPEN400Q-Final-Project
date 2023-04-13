# CPEN 400Q final project: Experimental Quantum GANs

## Quickstart

```
$ poetry install
$ poetry run jupyterlab  # Run the demo notebooks
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

### Development notes

Before committing, remember to follow these steps:
- `poetry check`: ensure `pyproject.toml` and the lock file are in a good state.
- `poetry run black quantumgan/`: format the code.
- `poetry run flake8 quantumgan/`: run the linter.
- `poetry run pytype quantumgan/`: run the static type checker.
