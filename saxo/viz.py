from matplotlib import pyplot as plt
from sklearn.base import check_is_fitted

from saxo.sklearn import SAXO


def plot_saxo(saxo: SAXO, axes: list[plt.Axes], X=None):
    check_is_fitted(saxo)
    saxo_results = saxo.results_
    timestamp_bounds = saxo_results.timestamp_bounds
    value_variables = saxo_results.value_variables
    for k in range(len(value_variables)):
        value_bounds = saxo_results.value_bounds[value_variables[k]]
        for i in range(1, len(timestamp_bounds) - 1):
            axes[k].axvline(timestamp_bounds[i], color="black")
        for i in range(len(value_bounds)):
            for j in range(1, len(value_bounds[i]) - 1):
                axes[k].plot(
                    (timestamp_bounds[i], timestamp_bounds[i + 1]),
                    (value_bounds[i][j], value_bounds[i][j]),
                    color="black",
                    linestyle="--",
                )
        if X is not None:
            axes[k].plot(X[:, k, :].T, alpha=0.1, color="black")
            axes[k].set_xlim((0, X.shape[-1] - 1))
