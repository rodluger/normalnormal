import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import george
from george.kernels import CosineKernel
from scipy.linalg import cho_factor


def norm_cov(mu, Sig, N=20):

    # Terms
    K = Sig.shape[0]
    j = np.ones((K, 1))
    m = np.mean(Sig)
    mvec = (Sig @ j) / K
    z = m / mu ** 2
    s = m * j - mvec

    # Coefficients
    fac = 1.0
    alpha = 0.0
    beta = 0.0
    for n in range(0, N + 1):
        alpha += fac
        beta += 2 * n * fac
        fac *= z * (2 * n + 3)

    # We're done
    return (
        (alpha / mu ** 2) * Sig
        + (alpha / (mu ** 2 * m)) * (s @ s.T - mvec @ mvec.T)
        + (beta / (mu ** 2 * m)) * (s @ s.T)
    )


# GP Mean
mu = 0.75

# GP Amplitude
std = 0.1

# Asymmetry term
offset = 0.5

# Dimension
K = 1000

# Expansion order in our series solution
N = 20

# Number of samples in numerical estimate
M = 100000

# Get the covariance matrix
t = np.linspace(0, 1, K)
period = 0.75
kernel = std ** 2 * (offset + CosineKernel(np.log(period)))
gp = george.GP(kernel)
gp.compute(t)
Sigma = gp.get_matrix(t) + 1e-12 * np.eye(K)

# Compute the normalized covariance using the series expansion
Sigma_norm = norm_cov(mu, Sigma, N=N)

# Compute it numerically by sampling
np.random.seed(0)
L = np.tril(cho_factor(Sigma, lower=True)[0])
u = np.random.randn(K, M)
x = mu + L @ u
xnorm = x / np.mean(x, axis=0).reshape(1, -1)
Sigma_norm_num = np.cov(xnorm)

# Figure setup
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# fig.subplots_adjust(wspace=0.05)
vmin = np.hstack((Sigma_norm, Sigma_norm_num)).min()
vmax = np.hstack((Sigma_norm, Sigma_norm_num)).max()

# Numerical solution
im = ax[0].imshow(Sigma_norm_num, cmap="viridis", vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=10)

# Difference
diff = (Sigma_norm - Sigma_norm_num) / Sigma_norm.max()
vmax = np.abs(diff).max()
vmin = -vmax
im = ax[1].imshow(diff, cmap="bwr", vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=10)

# Appearance
ax[0].set_title(r"$\tilde{\mathbf{\Sigma}}_\mathrm{num}$", fontsize=25)
ax[1].set_title(
    r"$\,\,\,\,\,\,\,\,\,"
    r"\left(\tilde{\mathbf{\Sigma}}"
    r"- \tilde{\mathbf{\Sigma}}_\mathrm{num}\right)"
    r"/ \, \tilde{\mathbf{\Sigma}}_\mathrm{max}$",
    fontsize=25,
)
for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])

# We're done
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight")
