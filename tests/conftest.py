import numpy as np
import pandas as pd
import pytest
from aeon.datasets import load_gunpoint

from saxo.sklearn import from_aeon_dataset, khiops_sort


@pytest.fixture(scope="session")
def cached_data():
    X, y = load_gunpoint()
    domain, df_secondary = from_aeon_dataset(X)
    df_secondary = khiops_sort(df_secondary, "id")
    df_root = pd.DataFrame({"id": np.arange(X.shape[0]), "label": y.reshape(-1)})
    df_root = khiops_sort(df_root, "id")
    return domain, df_root, df_secondary
