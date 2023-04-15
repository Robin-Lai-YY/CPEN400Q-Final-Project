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

matplotlib.use("pgf")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "TeX Gyre Pagella",
    }
)

key = jr.PRNGKey(0)
data = generate_grayscale_bar(key, 8).reshape(4, 2, 2, 2)

fig, ax = plt.subplots(2, 4)
for row in range(2):
    for col in range(4):
        ax[row, col].matshow(data[row, col])
fig.savefig("report/plots/bar_data.pdf")

gen_params, dis_params = BatchGAN.init_params(key, 4, 3, 1, 4, 1)
gan = BatchGAN(4, 1, gen_params, dis_params)

qml.draw_mpl(gan._qnode_train_fake)(
    gen_params, dis_params, gan.random_latent(key, 1)
)
plt.savefig("report/plots/batch_gan_circuit.pdf")
plt.close()


def train_example(key):
    key, params_key = jr.split(key)
    features_dim = 4
    batch_size = 1

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
        1,
        gen_params,
        dis_params,
    )

    key, train_key = jr.split(key)

    gen_optimizer = optax.sgd(0.05)
    dis_optimizer = optax.sgd(0.001)

    key, data_key = jr.split(key)
    train_data = datasets.generate_grayscale_bar(data_key, 2000).reshape(
        -1, batch_size, 4
    )

    return train_gan(
        train_key,
        gan,
        gen_optimizer,
        dis_optimizer,
        train_data,
        show_progress=True,
    )


train_result1 = train_example(jr.PRNGKey(1))

latent = gan.random_latent(jr.PRNGKey(0), 8)
# Final checkpoint
data = train_result1.checkpoints[-1][1].generate(latent).reshape(2, 4, 2, 2)

fig, ax = plt.subplots(2, 4)
for row in range(2):
    for col in range(4):
        ax[row, col].matshow(data[row, col])
fig.savefig("report/plots/generated_bars.pdf")

train_result2 = train_example(jr.PRNGKey(6))

fig, ax = plt.subplots(1, 2, sharey="row", figsize=(8, 4))
ax[0].set_title("Loss with initial parameters set 1")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")
ax[0].plot(
    range(2000), train_result1.g_loss_history, train_result1.d_loss_history
)
ax[1].set_title("Loss with initial parameters set 2")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Loss")
ax[1].plot(
    range(2000), train_result2.g_loss_history, train_result2.d_loss_history
)
ax[1].legend(["Generator", "Discriminator"])
fig.savefig("report/plots/loss_compare.pdf")


key = jr.PRNGKey(0)
key, params_key = jr.split(key)

features_dim = 4
batch_size = 2
gen_params, dis_params = BatchGAN.init_params(
    params_key,
    features_dim,
    gen_layers=3,
    gen_ancillary=1,
    dis_layers=3,
    dis_ancillary=1,
)
gan = BatchGAN(
    features_dim,
    batch_size,
    gen_params,
    dis_params,
    trainable=qml.RZ,
    entangler=RandomEntangler(key, entangler=qml.CNOT),
)

qml.draw_mpl(gan._qnode_train_fake)(
    gan.gen_params, gan.dis_params, gan.random_latent(key, 1)[0]
)
plt.savefig("report/plots/index_cnot_circuit.pdf")
