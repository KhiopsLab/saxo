import csv
import os
import tempfile
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from khiops import core as kh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from saxo import train_saxo
from saxo.report import read_saxo_results_file
from saxo.utils import make_saxo_labels


def to_khiops_txt(df: pd.DataFrame, path: str) -> None:
    df.to_csv(
        path,
        sep="\t",
        header=True,
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        encoding="utf-8",
        index=False,
    )


def khiops_sort(df: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    return df.sort_values(by=by, key=lambda col: col.astype(str), kind="mergesort")


def khiops_unsort(df: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    return df.sort_values(by=by, key=lambda col: col.astype(int), kind="mergesort")


def from_aeon_dataset(X: np.ndarray) -> Tuple[kh.DictionaryDomain, pd.DataFrame]:

    domain = kh.DictionaryDomain()
    dictionary = kh.Dictionary()
    dictionary.name = "series"
    domain.add_dictionary(dictionary)
    id = kh.Variable()
    id.name = "id"
    id.type = "Categorical"
    dictionary.add_variable(id)
    ts = kh.Variable()
    ts.name = "timestamp"
    ts.type = "Numerical"
    dictionary.add_variable(ts)
    for i in range(X.shape[1]):
        value = kh.Variable()
        value.name = f"value_{i}"
        value.type = "Numerical"
        dictionary.add_variable(value)

    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(X.shape[0]), X.shape[2]),
            "timestamp": np.tile(np.arange(X.shape[2]), X.shape[0]),
        }
        | {f"value_{i}": X[:, i, :].reshape(-1) for i in range(X.shape[1])}
    )

    return domain, data


class SAXO(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        timestamp_discretization="SAXO",
        min_optimization_time=None,
        refined=True,
        max_symbols=None,
        max_intervals=None,
        n_jobs=None,
    ):
        self.timestamp_discretization = timestamp_discretization
        self.min_optimization_time = min_optimization_time
        self.refined = refined
        self.max_symbols = max_symbols
        self.max_intervals = max_intervals
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.one_d_array = False
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = []
        return tags

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=False, allow_nd=True, ensure_all_finite=False)
        assert X.ndim == 3
        with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
            domain, df = from_aeon_dataset(X)
            df = khiops_sort(df, "id")
            output_table_path = f"{work_dir}/saxo-train-data.txt"
            to_khiops_txt(df, output_table_path)

            max_symbols = 0 if self.max_symbols is None else self.max_symbols
            max_intervals = 0 if self.max_intervals is None else self.max_intervals
            max_cores = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
            min_optimization_time = (
                0 if self.min_optimization_time is None else self.min_optimization_time
            )
            timestamp_discretization_method = (
                "EqualWidth"
                if self.timestamp_discretization is None
                else self.timestamp_discretization
            )

            value_names = []
            for variable in domain.get_dictionary("series").variables:
                if variable.name.startswith("value"):
                    value_names.append(variable.name)

            (
                saxo_dictionary_file_path,
                biclustering_reports_paths,
                timestamp_discretization_report_path,
            ) = train_saxo(
                domain,
                "series",
                output_table_path,
                "id",
                "timestamp",
                value_names,
                work_dir,
                timestamp_discretization_method=timestamp_discretization_method,
                max_symbols=max_symbols,
                max_intervals=max_intervals,
                max_cores=max_cores,
                min_optimization_time=min_optimization_time,
                refined=self.refined,
                build_distance_variables=True,
            )

            self.results_ = read_saxo_results_file(
                biclustering_reports_paths,
                "id",
                "timestamp",
                timestamp_discretization_report_path,
            )
            self.model_ = kh.read_dictionary_file(saxo_dictionary_file_path)

            for feature_name in self.get_feature_names_out():
                dictionary = self.model_.get_dictionary("SAXO_series")
                dictionary.get_variable(feature_name).used = True
                dictionary.get_variable(f"{feature_name}_distance").used = False

        return self

    def get_feature_names_out(self, input_features=None):
        bounds = self.results_.timestamp_bounds
        return make_saxo_labels(bounds)

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False, allow_nd=True, ensure_all_finite=False)
        assert X.ndim == 3

        with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
            _, df_secondary = from_aeon_dataset(X)
            df_secondary = khiops_sort(df_secondary, "id")
            secondary_table_path = f"{work_dir}/saxo-test-data.txt"
            to_khiops_txt(df_secondary, secondary_table_path)

            df_root = pd.DataFrame({"id": np.arange(X.shape[0])})
            df_root = khiops_sort(df_root, "id")
            root_table_path = f"{work_dir}/saxo-root-data.txt"
            to_khiops_txt(df_root, root_table_path)

            output_table_path = f"{work_dir}/saxo-test-representation.txt"
            kh.deploy_model(
                self.model_,
                "SAXO_series",
                root_table_path,
                output_table_path,
                additional_data_tables={"Table_series": secondary_table_path},
            )
            df_transformed = pd.read_csv(output_table_path, sep="\t")
            df_transformed = khiops_unsort(df_transformed, "id")
            df_transformed.drop("id", axis=1, inplace=True)

        return df_transformed.values

    def score_samples(self, X):

        try:
            for feature_name in self.get_feature_names_out():
                dictionary = self.model_.get_dictionary("SAXO_series")
                dictionary.get_variable(feature_name).used = False
                dictionary.get_variable(f"{feature_name}_distance").used = True

            # higher the distance the lower the score (sklearn convention)
            scores = -self.transform(X)

        finally:
            for feature_name in self.get_feature_names_out():
                dictionary = self.model_.get_dictionary("SAXO_series")
                dictionary.get_variable(feature_name).used = True
                dictionary.get_variable(f"{feature_name}_distance").used = False

        return scores

    @property
    def compression_gain_(self):
        if self.refined:
            return 1 - self.results_.cost / self.results_.null_cost
        else:
            results = self.results_.timestamp_discretization_report
            return 1 - results.cost / results.null_cost
