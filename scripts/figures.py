import matplotlib
import matplotlib.pyplot as plt
import jax.random as jr
import pennylane as qml

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
fig.savefig("plots/bar_data.pdf")

gen_params, dis_params = BatchGAN.init_params(key, 4, 3, 1, 4, 1)
gan = BatchGAN(4, 1, gen_params, dis_params)

qml.draw_mpl(gan._qnode_train_fake)(
    gen_params, dis_params, gan.random_latent(key, 1)
)
plt.savefig("plots/batch_gan_circuit.pdf")
plt.close()
