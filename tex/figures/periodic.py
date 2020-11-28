import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import CosineKernel


def norm_cov(mu, Sig, N=20):

    # Terms
    K = Sig.shape[0]
    j = np.ones((K, 1))
    m = np.mean(Sig)
    mvec = (Sig @ j) / K
    x = m / mu ** 2
    s = m * j - mvec

    # Coefficients
    fac = 1.0
    alpha = 0.0
    beta = 0.0
    for n in range(0, N + 1):
        alpha += fac
        beta += 2 * n * fac
        fac *= x * (2 * n + 3)

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

# Get the covariance matrix
t = np.linspace(0, 1, K)
period = 0.75
kernel = std ** 2 * (offset + CosineKernel(np.log(period)))
gp = george.GP(kernel)
gp.compute(t)
Sigma = gp.get_matrix(t)

# Compute the normalized covariance using the series expansion
Sigma_norm = norm_cov(mu, Sigma, N=N)

# Figure setup
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
vmin = Sigma_norm.min()
vmax = Sigma_norm.max()

ax[0].imshow(Sigma, cmap="viridis", vmin=vmin, vmax=vmax)
ax[1].imshow(Sigma_norm, cmap="viridis", vmin=vmin, vmax=vmax)

ax[0].set_title(r"$\mathbf{\Sigma}$", fontsize=25)
ax[1].set_title(r"$\mathbf{\tilde{\Sigma}}$", fontsize=25)

for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])

# We're done
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight")
