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
    K = Sig.shape[0]
    J = np.ones((K, K))
    barSig = np.mean(Sig)

    # Powers of S
    S = (Sig @ J) / K ** 2
    Spow = [np.eye(K)]
    for n in range((N + 2) // 2):
        Spow.append(Spow[-1] @ S)

    # Series solution
    norm_cov = np.zeros_like(Sig)
    for n in range(0, N + 1, 2):
        norm_cov += (
            (-1) ** n
            * (n + 1) ** 2
            * g(n)
            * (
                barSig ** ((n + 2) // 2) * J
                - K / mu * (Spow[(n + 2) // 2] + Spow[(n + 2) // 2].T)
                + n / ((n + 1) * mu ** 2) * Spow[n // 2] @ Sig
                + barSig ** (n // 2) / ((n + 1) * mu ** 2) * Sig
            )
        )

    return norm_cov
