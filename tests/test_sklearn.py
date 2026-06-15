import numpy as np
import pandas as pd
from aeon.datasets import load_gunpoint
from sklearn import config_context
from sklearn.utils.estimator_checks import parametrize_with_checks

from saxo.sklearn import SAXO, from_aeon_dataset, khiops_sort, khiops_unsort


def test_sklearn_saxo_set_output():
    X, _ = load_gunpoint()
    saxo = SAXO(max_intervals=10, max_symbols=10, n_jobs=-1).fit(X)
    assert isinstance(saxo.transform(X), np.ndarray)
    with config_context(transform_output="pandas"):
        assert isinstance(saxo.transform(X), pd.DataFrame)


def test_khiops_sort_unsort():
    X, _ = load_gunpoint()
    _, df = from_aeon_dataset(X)
    df_sorted = khiops_sort(df, "id")
    assert not df.equals(df_sorted)
    assert df.equals(khiops_unsort(df_sorted, "id"))


@parametrize_with_checks([SAXO(), SAXO(refined=False)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
