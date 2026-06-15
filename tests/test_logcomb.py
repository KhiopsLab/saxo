import math
import pytest

from scipy.special import stirling2

from saxo.logcomb import logcomb, logfactorial, logstirling2, logbell


@pytest.mark.parametrize("N", range(100, 1000, 100))
@pytest.mark.parametrize("K", range(1, 10, 1))
def test_logcomb(N, K):
    assert math.isclose(math.log(math.comb(N, K)), logcomb(N, K))
    assert math.isclose(math.log(math.factorial(N)), logfactorial(N))
    assert math.isclose(
        math.log(stirling2(N, K, exact=True)), logstirling2(N, K), abs_tol=1e-3
    )
    assert math.isclose(
        math.log(sum((stirling2(N, i, exact=True) for i in range(1, K + 1)))),
        logbell(N, K),
        abs_tol=1e-3,
    )
