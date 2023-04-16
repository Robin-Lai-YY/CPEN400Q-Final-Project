"""Script to (repeatably) generate the plots used in the report.
"""
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import jax.random as jr
import pandas as pd
import optax
import multiprocessing
from multiprocessing import Pool
from tqdm.auto import tqdm
import shelve
from json import dumps, loads
from itertools import product
from collections import defaultdict

from quantumgan.gan import GAN
from quantumgan.batch import BatchGAN
from quantumgan.classical import BarMLPGAN
from quantumgan.datasets import generate_grayscale_bar, frechet_distance
from quantumgan.train import train_gan, TrainResult
from quantumgan.ideal_device import IdealDeviceJax
from quantumgan.devices import NoisyDevice


def create_batch_gan(
    params_key: jr.PRNGKeyArray,
    batch_size,
    gen_layers,
    gen_ancillary,
    dis_layers,
    dis_ancillary,
    device=IdealDeviceJax,
    disable_jax_vmap=False,
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
        device=device,
        disable_jax_vmap=disable_jax_vmap,
    )

    return gan


def create_mlp_gan(
    params_key: jr.PRNGKeyArray,
    latent_shape,
    gen_hidden,
    dis_hidden,
):
    gan = BarMLPGAN(params_key, latent_shape, gen_hidden, dis_hidden)
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
            jit = True
        case "mlp":
            gan = create_mlp_gan(params_key, **kwargs)
            jit = True
        case "batch_noisy":
            gan = create_batch_gan(
                params_key, device=NoisyDevice, disable_jax_vmap=True, **kwargs
            )
            jit = False  # Avoid conversion between JAX arrays and python numpy arrays

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
        checkpoint_freq=50,
        show_progress=True,
        jit=jit,
    )

    return config, evaluate_gan(eval_key, train_result)


mlpgan_conf1 = (
    "mlp",
    {
        "iters": 350,
        "batch_size": 1,
        "gen_lr": 0.05,
        "dis_lr": 0.001,
    },
    {
        "latent_shape": 1,
        "gen_hidden": 1,
        "dis_hidden": [20, 10],
    },
)

mlpgan_conf2 = (
    "mlp",
    {
        "iters": 350,
        "batch_size": 1,
        "gen_lr": 0.05,
        "dis_lr": 0.001,
    },
    {
        "latent_shape": 2,
        "gen_hidden": 2,
        "dis_hidden": [20, 10],
    },
)

mlpgan_conf3 = (
    "mlp",
    {
        "iters": 350,
        "batch_size": 1,
        "gen_lr": 0.05,
        "dis_lr": 0.001,
    },
    {
        "latent_shape": 2,
        "gen_hidden": 8,
        "dis_hidden": [20, 10],
    },
)

cnngan_conf1 = (
    "cnn",
    {
        "iters": 350,
        "batch_size": 1,
        "gen_lr": 0.05,
        "dis_lr": 0.001,
    },
    {
        "gen_hidden": 3,
        "dis_hidden": [20, 10],
    },
)

batchgan_conf1 = (
    "batch",
    {
        "iters": 350,
        "batch_size": 1,
        "gen_lr": 0.05,
        "dis_lr": 0.001,
    },
    {
        "batch_size": 1,
        "gen_layers": 3,
        "gen_ancillary": 1,
        "dis_layers": 4,
        "dis_ancillary": 1,
    },
)

batchgan_noisy_conf1 = (
    "batch_noisy",
    {
        "iters": 350,
        "batch_size": 1,
        "gen_lr": 0.05,
        "dis_lr": 0.001,
    },
    {
        "batch_size": 1,
        "gen_layers": 3,
        "gen_ancillary": 1,
        "dis_layers": 4,
        "dis_ancillary": 1,
    },
)


def configuration_space(noise_graph):
    yield batchgan_conf1
    yield mlpgan_conf1
    if noise_graph:
        yield batchgan_noisy_conf1


def plot_fd(ax, filt, color, mediancolor, width):
    scores = defaultdict(list)
    with shelve.open("results.db") as db:
        for key in db:
            seed, ty, train_conf, gan_conf = loads(key)
            if filt((ty, train_conf, gan_conf)):
                for i, fd in db[key]:
                    scores[i].append(fd)
    df = pd.DataFrame(scores)
    ax.set_ylim(1e-2, 2)
    sns.boxplot(
        df,
        ax=ax,
        whis=0.5,
        color=color,
        width=width,
        medianprops=dict(color=mediancolor),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
    )


def create_plots(noise_graph, latex_backend):
    if latex_backend:
        mpl.use("pgf")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "TeX Gyre Pagella",
            }
        )

    fig, ax = plt.subplots(1, 3, sharey="row", figsize=(10, 4))
    ax[0].set_title("Batch GAN (Ideal) FD score")
    ax[0].set_xlabel("Training iteration")
    ax[0].set_ylabel("FD score")
    ax[0].set_yscale("log")
    plot_fd(
        ax[0],
        lambda c: c == batchgan_conf1,
        color="darkgreen",
        width=0.5,
        mediancolor="lightskyblue",
    )
    ax[1].set_title("Batch GAN (Noisy) FD score")
    ax[1].set_xlabel("Training iteration")
    ax[1].set_yscale("log")
    if noise_graph:
        plot_fd(
            ax[1],
            lambda c: c == batchgan_noisy_conf1,
            color="yellowgreen",
            width=0.5,
            mediancolor="lightskyblue",
        )
    ax[2].set_title("MLP GAN FD score")
    ax[2].set_xlabel("Training iteration")
    ax[2].set_yscale("log")
    plot_fd(
        ax[2],
        lambda c: c == mlpgan_conf1,
        color="darkred",
        width=0.5,
        mediancolor="yellow",
    )
    fig.savefig("plots/fd_scores.pdf")


def parse():
    parser = argparse.ArgumentParser(
        description="Script for generating the box plots"
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="include plots for the noisy device",
    )
    parser.add_argument(
        "--latex",
        default=False,
        type=bool,
        help="use LaTeX backend for generating plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    if not args.noise:
        print("Skipping plots for the noisy device")
    training_runs = 10
    jobs = []

    with shelve.open("results.db") as db:
        for config in configuration_space(args.noise):
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

    create_plots(args.noise, args.latex)
