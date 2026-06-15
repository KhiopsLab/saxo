import numpy as np
from scipy import special as sp


def logcomb(N, K):
    return sp.gammaln(N + 1) - sp.gammaln(N - K + 1) - sp.gammaln(K + 1)


def logfactorial(N):
    return sp.gammaln(N + 1)


def logstirling2(N, K):
    # https://lipn.univ-paris13.fr/alea2012/SLIDES/Lundi/Guy%20Louchard%20-%20stirling2b.pdf
    # see Sachkov, Probabilistic Methods in Combinatorial Analysis p164
    # o(1) approx if K < N / log(N)
    return N * np.log(K) - logfactorial(K) + (N / K - K) * np.exp(-N / K)


def logbell(N, K):
    return sp.logsumexp([logstirling2(N, i) for i in range(1, K + 1)])
