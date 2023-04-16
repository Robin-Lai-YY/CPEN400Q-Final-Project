import matplotlib
import matplotlib.pyplot as plt
import jax.random as jr
import pennylane as qml
import optax

from quantumgan.train import train_gan
import quantumgan.datasets as datasets
from quantumgan.datasets import generate_grayscale_bar
from quantumgan.batch import BatchGAN
from quantumgan.mpqc import RandomEntangler

# matplotlib.use("pgf")
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "TeX Gyre Pagella",
#     }
# )

key = jr.PRNGKey(0)

def train_example(key, n_index_qubits):
    key, params_key = jr.split(key)
    features_dim = 4
    batch_size = 2 ** n_index_qubits

    gen_params, dis_params = BatchGAN.init_params(
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
        gen_params,
        dis_params,
    )

    key, train_key = jr.split(key)

    gen_optimizer = optax.sgd(0.05)
    dis_optimizer = optax.sgd(0.001)

    key, data_key = jr.split(key)
    train_data = datasets.generate_grayscale_bar(data_key, batch_size*2000).reshape(
        2000, batch_size, 4
    )

    return train_gan(
        train_key,
        gan,
        gen_optimizer,
        dis_optimizer,
        train_data,
        show_progress=True,
    )


rows, cols = 5, 3
fig, ax = plt.subplots(rows, cols, sharey="row", figsize=(10, 10))
for row in range(rows):
    key, current_key = jr.split(key)
    for col in range(cols):
        train_result = train_example(current_key, col)
        ax[row, col].set_title(f"Set {row}, indices: {col}")
        ax[row, col].set_xlabel("Iteration")
        ax[row, col].plot(
            range(2000), train_result.g_loss_history, train_result.d_loss_history
        )
    ax[row, 0].set_ylabel("Loss")
    ax[row, cols-1].legend(["Generator", "Discriminator"])

fig.tight_layout()
fig.savefig("report/plots/loss_compare_index.pdf")
