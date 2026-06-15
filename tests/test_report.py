import os
import tempfile

from saxo import train_saxo
from saxo.report import read_saxo_results_file
from saxo.sklearn import to_khiops_txt


def test_read_saxo_report(cached_data):
    domain, _, data = cached_data

    with tempfile.TemporaryDirectory(prefix="saxo_") as work_dir:
        data_path = os.path.join(work_dir, "data.csv")
        to_khiops_txt(data, data_path)

        output_path = os.path.join(work_dir, "output")

        _, biclustering_reports_paths, timestamp_discretization_report_path = (
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
        )

        saxo_results = read_saxo_results_file(
            biclustering_reports_paths,
            "id",
            timestamp_discretization_result_path=timestamp_discretization_report_path,
        )
        assert saxo_results.cost < saxo_results.null_cost
