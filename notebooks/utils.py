import numpy as np
from scipy.special import factorial


def g(n):
    if n % 2 == 0:
        return factorial(n) / (2 ** (n // 2) * factorial(n // 2))
    else:
        return 0


def d(i, j):
    if i == j:
        return 1
    else:
        return 0


def dbar(i, j):
    if i == j:
        return 0
    else:
        return 1


def uprod(*idx):
    if len(idx) % 2 != 0:
        return 0
    elif len(idx) == 2:
        return g(2)
    elif len(idx) == 4:
        i, j, k, l = idx
        return g(4) * d(i, j) * d(k, l) * d(i, k) + g(2) ** 2 * (
            d(i, j) * d(k, l) * dbar(i, k)
            + d(i, k) * d(j, l) * dbar(i, j)
            + d(i, l) * d(j, k) * dbar(i, j)
        )
    else:
        raise NotImplementedError("Case not implemented.")


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
