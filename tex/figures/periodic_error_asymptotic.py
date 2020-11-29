import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import CosineKernel
from scipy.linalg import cho_factor
from tqdm import tqdm


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

    # Record the value of the expansion parameter
    print("z:", z)

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
K = 10

# Expansion order in our series solution
N = 20

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


ntrials = 30
log10M = np.linspace(2, 6, 24)
M = np.array(10 ** log10M, dtype=int)

error_max = np.zeros((ntrials, len(M)))
error_med = np.zeros((ntrials, len(M)))

for n in tqdm(range(ntrials)):
    for k in range(len(M)):
        np.random.seed(n)
        u = np.random.randn(K, M[k])
        x = mu + L @ u
        xnorm = x / np.mean(x, axis=0).reshape(1, -1)
        Sigma_norm_num = np.cov(xnorm)

        error_max[n, k] = np.max(
            np.abs((Sigma_norm - Sigma_norm_num) / Sigma_norm.max())
        )
        error_med[n, k] = np.median(
            np.abs((Sigma_norm - Sigma_norm_num) / Sigma_norm.max())
        )

fig, ax = plt.subplots(1)
for n in tqdm(range(ntrials)):
    ax.plot(M, error_max[n], "C0-", alpha=0.2, lw=1)
    ax.plot(M, error_med[n], "C1-", alpha=0.2, lw=1)
ax.plot(M, np.mean(error_max, axis=0), "C0-", alpha=1, lw=2, label="max")
ax.plot(M, np.mean(error_med, axis=0), "C1-", alpha=1, lw=2, label="med")

ax.legend(loc="lower left")
ax.set_ylim(1e-4, 1e0)
ax.set_xlim(1e2, 1e6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of samples")
ax.set_ylabel("error")

# We're done
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight")
