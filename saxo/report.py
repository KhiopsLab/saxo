import math
from collections import defaultdict
from functools import cached_property

import khiops.core as kh

from saxo.logcomb import logbell, logcomb, logfactorial
from saxo.utils import get_coclustering_bounds, get_datagrid_bounds


def read_saxo_results_file(
    biclustering_results_paths: list[str],
    id_variable: str,
    timestamp_variable: str | None = None,
    timestamp_discretization_result_path: str | None = None,
):
    biclustering_reports = []
    for biclustering_report_path in biclustering_results_paths:
        biclustering_result = kh.read_coclustering_results_file(
            biclustering_report_path
        )
        biclustering_reports.append(biclustering_result.coclustering_report)

    if timestamp_discretization_result_path is not None:
        timestamp_discretization_result = kh.read_coclustering_results_file(
            timestamp_discretization_result_path
        )
        if timestamp_discretization_result.coclustering_report is not None:
            timestamp_discretization_report = (
                timestamp_discretization_result.coclustering_report
            )
        else:
            timestamp_discretization_result = kh.read_analysis_results_file(
                timestamp_discretization_result_path
            )
            timestamp_discretization_report = (
                timestamp_discretization_result.preparation_report
            )
    else:
        timestamp_discretization_result = None

    return SAXOResults(
        biclustering_reports,
        id_variable,
        timestamp_variable,
        timestamp_discretization_report,
    )


class SAXOResults:
    def __init__(
        self,
        biclustering_reports: list[kh.CoclusteringReport],
        id_variable: str,
        timestamp_variable: str | None = None,
        timestamp_discretization_report: kh.CoclusteringReport
        | kh.PreparationReport
        | None = None,
    ):
        self.biclustering_reports = biclustering_reports
        self.id_variable = id_variable
        self.timestamp_variable = timestamp_variable
        self.timestamp_discretization_report = timestamp_discretization_report

    @cached_property
    def cost(self) -> float:
        cost = 0
        frequencies = []  # m^t
        unique_ids = []  # N^t
        global_unique_ids = set()  # N
        id_part_numbers = []  # k_C^t
        for biclustering_report in self.biclustering_reports:
            cost += biclustering_report.cost
            frequency = 0
            for cell in biclustering_report.cells:
                frequency += cell.frequency
            frequencies.append(frequency)

            unique_id = set()
            id_dimension = biclustering_report.get_dimension(self.id_variable)
            id_part_numbers.append(id_dimension.part_number)
            for part in id_dimension.parts:
                for value in part.values:
                    unique_id.add(value.value)
                    global_unique_ids.add(value.value)

            unique_ids.append(unique_id)

        # remove overestimated saxo cost by sum of biclustering costs
        # see derivations/saxo_cost.md
        cost += math.log(sum(frequencies))
        cost += logcomb(sum(frequencies) + len(frequencies) - 1, len(frequencies) - 1)
        cost += 2 * logfactorial(sum(frequencies))
        for frequency in frequencies:
            cost -= logfactorial(frequency)

        # should not matter ...
        cost += len(frequencies) * math.log(len(global_unique_ids))
        for unique_id in unique_ids:
            cost -= math.log(len(unique_id))
        for unique_id, id_part_number in zip(unique_ids, id_part_numbers):
            cost += logbell(len(global_unique_ids), id_part_number)
            cost -= logbell(len(unique_id), id_part_number)

        if hasattr(cost, "item"):
            cost = cost.item()

        return cost

    @cached_property
    def null_cost(self) -> float:
        null_cost = 0
        frequencies = 0  # m
        id_frequencies = defaultdict(int)  # p_s and N
        for biclustering_report in self.biclustering_reports:
            for cell in biclustering_report.cells:
                frequencies += cell.frequency

            for part in biclustering_report.get_dimension(self.id_variable).parts:
                for value in part.values:
                    id_frequencies[value.value] += value.frequency

        null_cost += 2 * math.log(frequencies)
        null_cost += math.log(len(id_frequencies))
        null_cost += logcomb(
            frequencies + len(id_frequencies) - 1, len(id_frequencies) - 1
        )
        null_cost += 3 * logfactorial(frequencies)
        for frequency in id_frequencies.values():
            null_cost -= logfactorial(frequency)

        if hasattr(null_cost, "item"):
            null_cost = null_cost.item()

        return null_cost

    @cached_property
    def timestamp_bounds(self):
        if isinstance(self.timestamp_discretization_report, kh.CoclusteringReport):
            timestamp_bounds = get_coclustering_bounds(
                self.timestamp_discretization_report.get_dimension(
                    self.timestamp_variable
                )
            )
        elif isinstance(self.timestamp_discretization_report, kh.PreparationReport):
            timestamp_bounds = get_datagrid_bounds(
                self.timestamp_discretization_report.get_variable_statistics(
                    self.timestamp_variable
                ).data_grid.dimensions[0]
            )
        else:
            timestamp_bounds = []
            for biclustering_report in self.biclustering_reports:
                interval = biclustering_report.selection_value
                interval = interval.replace("]", "").replace("[", "")
                lower, upper = tuple([float(b) for b in interval.split(",")])
                if len(timestamp_bounds) == 0:
                    timestamp_bounds.append(lower)
                timestamp_bounds.append(upper)
        return timestamp_bounds

    @cached_property
    def value_variables(self):
        value_variables = []
        for dimension in self.biclustering_reports[0].dimensions:
            if dimension.name not in [self.id_variable, self.timestamp_variable]:
                value_variables.append(dimension.name)
        return value_variables

    @cached_property
    def value_bounds(self):
        value_bounds = {}
        for value_variable in self.value_variables:
            value_bounds[value_variable] = []
            for biclustering_report in self.biclustering_reports:
                value_bounds[value_variable].append(
                    get_coclustering_bounds(
                        biclustering_report.get_dimension(value_variable)
                    )
                )
        return value_bounds
