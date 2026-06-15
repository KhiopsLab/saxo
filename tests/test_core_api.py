from copy import deepcopy
import os
import tempfile

import khiops.core as kh
import pytest
from khiops.core.dictionary import Variable
from khiops.core.helpers import _build_multi_table_dictionary_domain

from saxo import train_saxo
from saxo.sklearn import to_khiops_txt


def test_train_saxo(cached_data):
    domain, root, data = cached_data
    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        root_path = os.path.join(work_dir, "root.csv")
        to_khiops_txt(root, root_path)

        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0"],
            output_path,
            max_intervals=10,
            max_symbols=26,
            triclustering_sample_percentage=10,
            biclustering_sample_percentage=100,
            max_cores=10,
        )
        assert os.path.exists(output_path)
        model_report_path = os.path.join(output_path, "model.json")
        saxo_dictionary_domain = kh.read_dictionary_file(
            os.path.join(output_path, "saxo.kdic")
        )
        label_variable = Variable()
        label_variable.name = "label"
        label_variable.type = "Categorical"
        saxo_dictionary_domain.get_dictionary("SAXO_series").add_variable(
            label_variable
        )
        kh.train_predictor(
            saxo_dictionary_domain,
            "SAXO_series",
            root_path,
            "label",
            model_report_path,
            additional_data_tables={"Table_series": data_path},
            max_constructed_variables=0,
            max_trees=0,
        )
        assert os.path.exists(model_report_path)
        assert (
            kh.read_analysis_results_file(model_report_path)
            .test_evaluation_report.get_snb_performance()
            .auc
            > 0.7
        )


def test_train_saxo_on_yoga_with_mt_dict(cached_data):
    domain, _, data = cached_data

    domain = deepcopy(domain)
    root_dictionary_name = "SAXO_series"
    domain.get_dictionary("series").key = ["id"]
    domain = _build_multi_table_dictionary_domain(
        domain,
        root_dictionary_name,
        "Table_series",
        update_secondary_table_name=False,
    )
    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0"],
            output_path,
            max_intervals=10,
            max_symbols=26,
            triclustering_sample_percentage=10,
            biclustering_sample_percentage=100,
            max_cores=10,
        )
        assert os.path.exists(output_path)


def test_train_saxo_on_yoga_st_with_root(cached_data):
    domain, _, data = cached_data
    domain = deepcopy(domain)
    domain.get_dictionary("series").key = ["id"]
    domain.get_dictionary("series").root = True
    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0"],
            output_path,
            max_intervals=10,
            max_symbols=26,
            triclustering_sample_percentage=10,
            biclustering_sample_percentage=100,
            max_cores=10,
        )
        assert os.path.exists(output_path)


def test_train_saxo_on_yoga_with_mv(cached_data):
    domain, _, data = cached_data

    domain = deepcopy(domain)
    dictionary = domain.get_dictionary("series")
    random_variable = kh.Variable()
    random_variable.name = "value_1"
    random_variable.used = True
    random_variable.type = "Numerical"
    random_variable.object_type = ""
    random_variable.structure_type = ""
    random_variable.rule = str(kh.Rule("Random"))
    dictionary.add_variable(random_variable)

    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0", "value_1"],
            output_path,
            max_intervals=10,
            max_symbols=26,
            triclustering_sample_percentage=10,
            biclustering_sample_percentage=100,
            max_cores=10,
        )
        assert os.path.exists(output_path)


def test_train_saxo_on_yoga_with_ts_fails(cached_data):
    domain, _, data = cached_data

    domain = deepcopy(domain)
    dictionary = domain.get_dictionary("series")
    formatted_timestamp = kh.Variable()
    formatted_timestamp.name = "formatted_timestamp"
    formatted_timestamp.used = True
    formatted_timestamp.type = "Timestamp"
    formatted_timestamp.object_type = ""
    formatted_timestamp.structure_type = ""
    formatted_timestamp.rule = str(
        kh.Rule(
            "BuildTimestamp",
            kh.Rule("BuildDate", 1, 1, 1),
            kh.Rule("BuildTime", 0, 0, dictionary.get_variable("timestamp")),
        )
    )
    dictionary.add_variable(formatted_timestamp)

    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        with pytest.raises(kh.KhiopsRuntimeError):
            train_saxo(
                domain,
                "series",
                data_path,
                "id",
                "formatted_timestamp",
                ["value_0"],
                output_path,
                max_intervals=10,
                max_symbols=26,
                triclustering_sample_percentage=10,
                biclustering_sample_percentage=100,
                max_cores=10,
            )


def test_train_white_noise_should_fail(cached_data):
    domain, _, data = cached_data

    domain = deepcopy(domain)
    dictionary = domain.get_dictionary("series")
    random_variable = kh.Variable()
    random_variable.name = "value_1"
    random_variable.used = True
    random_variable.type = "Numerical"
    random_variable.object_type = ""
    random_variable.structure_type = ""
    random_variable.rule = str(kh.Rule("Random"))
    dictionary.add_variable(random_variable)

    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        with pytest.raises(kh.KhiopsRuntimeError):
            train_saxo(
                domain,
                "series",
                data_path,
                "id",
                "timestamp",
                ["value_1"],
                output_path,
                max_intervals=10,
                max_symbols=26,
                triclustering_sample_percentage=10,
                biclustering_sample_percentage=100,
                max_cores=10,
            )


def test_train_saxo_equal_width(cached_data):
    domain, _, data = cached_data
    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0"],
            output_path,
            max_intervals=5,
            max_symbols=26,
            timestamp_discretization_method="EqualWidth",
            biclustering_sample_percentage=100,
            max_cores=10,
        )
        assert os.path.exists(output_path)


def test_train_saxo_custom_discretization(cached_data):
    domain, _, data = cached_data
    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0"],
            output_path,
            max_symbols=26,
            timestamp_discretization_method=[0, 10, 20, 30],
            biclustering_sample_percentage=100,
            max_cores=10,
        )
        assert os.path.exists(output_path)


def test_train_saxo_with_distance(cached_data):
    domain, _, data = cached_data
    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        train_saxo(
            domain,
            "series",
            data_path,
            "id",
            "timestamp",
            ["value_0"],
            output_path,
            max_intervals=5,
            max_symbols=26,
            timestamp_discretization_method="EqualWidth",
            biclustering_sample_percentage=100,
            max_cores=10,
            build_distance_variables=True,
        )
        assert os.path.exists(output_path)
