import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma


def alpha(z, n):
    return factorial(2 * n + 1) / (2 ** n) / factorial(n) * z ** n


def beta(z, n):
    return alpha(z, n) * 2 * n


def log10_alpha(z, n):
    return (
        loggamma(2 * n + 2) - loggamma(n + 1) - n * np.log(2) + n * np.log(z)
    ) / np.log(10)


def log10_beta(z, n):
    return log10_alpha(z, n) + np.log10(2 * n)


# Settings
nmax = 150
zs = [0.005, 0.01, 0.02]
amax = 1.10
bmax = 0.25

# Compute
n = np.arange(0, nmax, 1)
la = [log10_alpha(z, n) for z in zs]
a = [np.cumsum(10 ** la[k]) for k in range(len(zs))]
lb = [log10_beta(z, n) for z in zs]
b = [np.cumsum(10 ** lb[k]) for k in range(len(zs))]

# Setup
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(wspace=0.1)
ax_dummy = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=3)
ax_coeff = ax_dummy.twinx()
ax_alpha = plt.subplot2grid((2, 4), (0, 3))
ax_beta = plt.subplot2grid((2, 4), (1, 3))

# Plot
for k, z in enumerate(zs):
    ax_coeff.plot(n, la[k], "C{}-".format(k))
    ax_coeff.plot(n, lb[k], "C{}--".format(k))
    ax_alpha.plot(n, np.minimum(amax + 1, a[k]), "C{}-".format(k))
    ax_beta.plot(n, np.minimum(bmax + 1, b[k]), "C{}-".format(k))

# Tweak appearance
ax_dummy.set_xscale("log")
ax_dummy.set_yscale("log")
log_ymin, log_ymax = ax_coeff.get_ylim()
ax_coeff.set_ylim(log_ymin, log_ymax)
ax_dummy.set_ylim(10 ** log_ymin, 10 ** log_ymax)
ax_dummy.set_xlim(1, nmax * 1.1)
ax_dummy.set_yticks([1e-40, 1e-20, 1e0, 1e20, 1e40, 1e60])
ax_coeff.set_yticks([])
ax_dummy.set_xlabel(r"coefficient number")
ax_dummy.set_ylabel(r"coefficient value")
ax_alpha.set_ylabel(r"$\alpha$ partial sum")
ax_alpha.yaxis.tick_right()
ax_alpha.yaxis.set_label_position("right")
ax_beta.set_ylabel(r"$\beta$ partial sum")
ax_beta.yaxis.tick_right()
ax_beta.yaxis.set_label_position("right")
ax_alpha.set_xscale("log")
ax_alpha.set_ylim(1.0, amax)
ax_beta.set_xscale("log")
ax_beta.set_ylim(0.0, bmax)
ax_beta.set_xlabel(r"coefficient number")
for axis in [ax_dummy, ax_alpha, ax_beta]:
    axis.set_xlim(1, nmax)

for axis in [ax_alpha, ax_beta]:
    axis.tick_params(axis="both", which="major", labelsize=10)

# Mark the minima
for k in range(len(zs)):
    n_ = n[np.argmin(la[k])]
    la_ = la[k][np.argmin(la[k])]
    ax_coeff.plot([n_, n_], [log_ymin, la_], "C{}-".format(k), lw=1)
    a_ = a[k][np.argmin(la[k])]
    ax_alpha.plot([n_, nmax], [a_, a_], "C{}--".format(k), lw=1)

    n_ = n[np.argmin(lb[k][1:])]
    lb_ = lb[k][1:][np.argmin(lb[k][1:])]
    ax_coeff.plot([n_, n_], [log_ymin, lb_], "C{}--".format(k), lw=1)
    b_ = b[k][1:][np.argmin(lb[k][1:])]
    ax_beta.plot([n_, nmax], [b_, b_], "C{}--".format(k), lw=1)

# Labels & legends
ax_dummy.plot(0, 0, "k-", label=r"$\alpha_n$")
ax_dummy.plot(0, 0, "k--", label=r"$\beta_n$")
ax_dummy.legend(fontsize=16, loc="upper left")
for k, z in enumerate(zs):
    ax_coeff.plot(
        0, 0, "C{}-".format(k), label=r"$z = {}$".format(z),
    )
ax_coeff.legend(fontsize=16, loc="lower left")

# We're done
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight")
