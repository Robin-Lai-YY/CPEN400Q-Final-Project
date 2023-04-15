FROM python:3.10.11-slim-bullseye

WORKDIR /src
RUN pip install poetry
COPY \
    pyproject.toml \
    poetry.lock \
    ./
RUN poetry install
COPY README.md .flake8 __init__.py ./
COPY quantumgan/ quantumgan/
COPY demo/ demo/
COPY scripts/ scripts/
RUN poetry install
