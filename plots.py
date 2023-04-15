"""Script to (repeatably) generate the plots used in the report.
"""
import matplotlib.pyplot as plt
import jax.random as jr
import optax
import multiprocessing
from multiprocessing import Pool
from tqdm.auto import tqdm
import shelve
from json import dumps, loads
from itertools import product

from quantumgan.gan import GAN
from quantumgan.batch import BatchGAN
from quantumgan.classical import BarMLPGAN
from quantumgan.datasets import generate_grayscale_bar, frechet_distance
from quantumgan.train import train_gan, TrainResult


def create_batch_gan(
    params_key: jr.PRNGKeyArray,
    batch_size,
    gen_layers,
    gen_ancillary,
    dis_layers,
    dis_ancillary,
):
    features_dim = 4

    init_gen_params, init_dis_params = BatchGAN.init_params(
        params_key,
        features_dim,
        gen_layers=gen_layers,
        gen_ancillary=gen_ancillary,
        dis_layers=dis_layers,
        dis_ancillary=dis_ancillary,
    )

    gan = BatchGAN(
        features_dim,
        batch_size,
        init_gen_params,
        init_dis_params,
    )

    return gan


def create_mlp_gan(
    params_key: jr.PRNGKeyArray,
    gen_hidden,
    dis_hidden,
):
    gan = BarMLPGAN(params_key, gen_hidden, dis_hidden)
    return gan


def evaluate_gan(key: jr.PRNGKeyArray, train_result: TrainResult):
    eval_samples = 1000
    key, latent_key = jr.split(key)
    gan0 = train_result.checkpoints[0][1]
    latent = gan0.random_latent(latent_key, eval_samples)
    key, gen_key, sample_key = jr.split(key, 3)
    samples = generate_grayscale_bar(sample_key, eval_samples)

    scores = []
    for i, gan in train_result.checkpoints:
        generated = gan.generate(latent)
        scores.append((i, frechet_distance(samples, generated).item()))

    return scores


def train_and_evaluate(config):
    key = jr.PRNGKey(0)
    seed, ty, train_config, kwargs = config
    params_key, data_key, train_key, eval_key = jr.split(
        jr.fold_in(key, seed), 4
    )

    match ty:
        case "batch":
            gan = create_batch_gan(params_key, **kwargs)
        case "mlp":
            gan = create_mlp_gan(params_key, **kwargs)

    gen_optimizer = optax.sgd(train_config["gen_lr"])
    dis_optimizer = optax.sgd(train_config["dis_lr"])

    train_data = generate_grayscale_bar(
        data_key, train_config["iters"]
    ).reshape(-1, train_config["batch_size"], 4)
    train_result = train_gan(
        train_key,
        gan,
        gen_optimizer,
        dis_optimizer,
        train_data,
        checkpoint_freq=500,
    )

    return config, evaluate_gan(eval_key, train_result)


def configuration_space():
    for gen_layers, dis_layers in product(range(3, 6), range(3, 6)):
        yield (
            "batch",
            {
                "iters": 3500,
                "batch_size": 1,
                "gen_lr": 0.05,
                "dis_lr": 0.001,
            },
            {
                "batch_size": 1,
                "gen_layers": gen_layers,
                "gen_ancillary": 1,
                "dis_layers": dis_layers,
                "dis_ancillary": 1,
            },
        )
    yield (
        "mlp",
        {
            "iters": 3500,
            "batch_size": 1,
            "gen_lr": 0.05,
            "dis_lr": 0.001,
        },
        {
            "gen_hidden": 5,
            "dis_hidden": (5, 2),
        },
    )


if __name__ == "__main__":
    training_runs = 20
    jobs = []

    with shelve.open("results.db") as db:
        for config in configuration_space():
            for seed in range(training_runs):
                c = (seed,) + config
                if dumps(c) not in db:
                    jobs.append(c)

    if len(jobs) > 0:
        multiprocessing.set_start_method("spawn")
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = tqdm(
                pool.imap_unordered(train_and_evaluate, jobs, chunksize=1),
                total=len(jobs),
            )

            with shelve.open("results.db") as db:
                for r in results:
                    if r is None:
                        continue
                    config, data = r
                    db[dumps(config)] = data
