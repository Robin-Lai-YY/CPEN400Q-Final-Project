import matplotlib
import matplotlib.pyplot as plt
import jax.random as jr
import pennylane as qml
import optax

from quantumgan.train import train_gan
import quantumgan.datasets as datasets
from quantumgan.datasets import generate_grayscale_bar
from quantumgan.batch import BatchGAN

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

key = jr.PRNGKey(1)
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
train_data = datasets.generate_grayscale_bar(data_key, 5000).reshape(
    -1, batch_size, 4
)

train_result = train_gan(
    train_key,
    gan,
    gen_optimizer,
    dis_optimizer,
    train_data,
    show_progress=True,
)

latent = gan.random_latent(jr.PRNGKey(0), 8)
# Final checkpoint
data = train_result.checkpoints[-1][1].generate(latent).reshape(2, 4, 2, 2)

fig, ax = plt.subplots(2, 4)
for row in range(2):
    for col in range(4):
        ax[row, col].matshow(data[row, col])
fig.savefig("report/plots/generated_bars.pdf")
