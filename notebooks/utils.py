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

    # Powers of S / barSig
    G = (Sig @ J) / K ** 2 / barSig
    Gpow = [np.eye(K)]
    for n in range((N + 2) // 2):
        Gpow.append(Gpow[-1] @ G)

    norm_cov = np.zeros_like(Sig)
    for n in range(0, N + 1, 2):
        fac = (-1) ** n * (n + 1) * g(n) * barSig ** (n // 2) / (mu ** n)
        norm_cov += fac * (
            Sig
            + n * (Gpow[n // 2]) @ Sig
            + (n + 1) * barSig * J
            - (n + 1) * barSig * K * (Gpow[(n + 2) // 2] + Gpow[(n + 2) // 2].T)
        )

    return norm_cov / mu ** 2
