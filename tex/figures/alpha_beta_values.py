import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def alpha_beta(z, nmax=50):

    fac_prev = 1.0
    fac = 1.0
    alpha = 0.0
    beta = 0.0
    for n in range(0, nmax + 1):
        if fac > fac_prev:
            break
        alpha += fac
        beta += 2 * n * fac
        fac_prev = fac
        fac *= z * (2 * n + 3)

    return alpha, beta, fac, 2 * n * fac


zmax = 0.1

z = np.logspace(-5, np.log10(zmax), 1000)
alpha = np.zeros_like(z)
beta = np.zeros_like(z)
error1 = np.zeros_like(z)
error2 = np.zeros_like(z)
for k in range(len(z)):
    alpha[k], beta[k], error1[k], error2[k] = alpha_beta(z[k])

fig, ax = plt.subplots(1)
axt = ax.twinx()
ax.plot(z, alpha, color="C0", label=r"$\alpha$")
axt.plot(z, beta, color="C1")

ax.fill_between(z, alpha - error1, alpha + error1, color="C0", alpha=0.1)
axt.fill_between(z, beta - error2, beta + error2, color="C1", alpha=0.1)

ax.set_xscale("log")

ax.set_ylim(0.95, 2)
axt.set_ylim(-0.1, 2)

ax.set_xlabel(r"$z$", fontsize=26)
ax.set_ylabel(r"$\alpha$", fontsize=26)
axt.set_ylabel(r"$\beta$", fontsize=26)

ax.plot(z, alpha * np.nan, color="C1", label=r"$\beta$")
ax.legend(loc="upper left")


ax.axvline(0.023, color="k", ls="--", lw=1, alpha=0.75)


# We're done
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight")
