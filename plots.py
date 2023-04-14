"""Script to (repeatably) generate the plots used in the report.
"""
import matplotlib.pyplot as plt
import jax.random as jr
import optax
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import shelve

from quantumgan.gan import GAN
from quantumgan.batch import BatchGAN
from quantumgan.datasets import generate_grayscale_bar, frechet_distance
from quantumgan.train import train_gan, TrainResult


def train_batch_gan(
    key: jr.PRNGKeyArray,
):
    key, params_key = jr.split(key)
    features_dim = 4
    batch_size = 1
    train_iters = 350

    init_gen_params, init_dis_params = BatchGAN.init_params(
        params_key,
        features_dim,
        gen_layers=3,
        gen_ancillary=1,
        dis_layers=4,
        dis_ancillary=1,
    )

    gan = BatchGAN(
        features_dim,
        batch_size,
        init_gen_params,
        init_dis_params,
    )

    gen_optimizer = optax.sgd(0.05)
    dis_optimizer = optax.sgd(0.001)

    key, data_key = jr.split(key)
    train_data = generate_grayscale_bar(data_key, train_iters)
    train_data = train_data.reshape(-1, batch_size, features_dim)

    return train_gan(
        key,
        gan,
        gen_optimizer,
        dis_optimizer,
        train_data,
        checkpoint_freq=50,
    )

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

key = jr.PRNGKey(0)
def train_and_evaluate(i: int):
    train_key, eval_key = jr.split(jr.fold_in(key, i), 2)
    train_result = train_batch_gan(train_key)
    return key, evaluate_gan(eval_key, train_result)

scores = []
total = 100

with shelve.open('results') as db:
    with Pool(processes=1) as pool:
        for i in range(total):
            scores = tqdm(pool.imap_unordered(train_and_evaluate, range(total)), total=total)

            for j, score in scores:
                db[j] == score
