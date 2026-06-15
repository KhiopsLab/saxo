import os
import string
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import warnings

from khiops import core as kh
from khiops.core.helpers import _build_multi_table_dictionary_domain

from saxo.utils import get_coclustering_bounds, get_datagrid_bounds, make_saxo_labels

# SAXO symbols list
SAXO_SYMBOLS = (
    list(string.ascii_lowercase)
    + list(string.ascii_uppercase)
    + [str(i) for i in range(10)]
)


def train_saxo(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    id_variable,
    timestamp_variable,
    value_variables,
    output_saxo_dir,
    # data table format
    detect_format=True,
    header_line=None,
    field_separator=None,
    # coclustering parameters
    frequency_variable="",
    min_optimization_time=0,
    # saxo parameters
    max_symbols=0,
    max_intervals=0,
    timestamp_discretization_method="SAXO",
    triclustering_sample_percentage=100,
    biclustering_sample_percentage=100,
    build_distance_variables=False,
    refined=True,
    # misc
    max_cores=None,
    **kwargs,
):
    assert max_symbols <= len(SAXO_SYMBOLS), (
        f"max_symbols requested {max_symbols} too high (>{len(SAXO_SYMBOLS)})"
    )

    train_coclustering = partial(
        kh.train_coclustering,
        detect_format=detect_format,
        header_line=header_line,
        field_separator=field_separator,
        frequency_variable=frequency_variable,
        min_optimization_time=min_optimization_time,
    )

    if isinstance(dictionary_file_path_or_domain, kh.DictionaryDomain):
        domain = dictionary_file_path_or_domain
    else:
        domain = kh.read_dictionary_file(dictionary_file_path_or_domain)

    # If the input dictionary only contain one dictionary, make it secondary and create
    # a root dictionary with a link to the secondary table
    secondary_dictionary = domain.get_dictionary(dictionary_name)
    if secondary_dictionary.root:
        secondary_dictionary.root = False
    root_dictionaries = [d for d in domain.dictionaries if d.root]
    if len(root_dictionaries) == 0:
        root_dictionary_name = f"SAXO_{dictionary_name}"
        secondary_dictionary.key = [id_variable]
        domain = _build_multi_table_dictionary_domain(
            domain,
            root_dictionary_name,
            f"Table_{dictionary_name}",
            update_secondary_table_name=False,
        )
        root_dictionary = domain.get_dictionary(root_dictionary_name)
    else:
        root_dictionary = root_dictionaries[0]
        root_dictionary_name = root_dictionary.name

    for variable in root_dictionary.variables:
        if (
            variable.type == "Table"
            and variable.object_type == dictionary_name
            and variable.rule == ""
        ):
            secondary_table = variable
            break

    timestamp = domain.get_dictionary(dictionary_name).get_variable(timestamp_variable)

    if timestamp_discretization_method == "SAXO":
        triclustering_report_path = os.path.join(
            output_saxo_dir, "full_triclustering.khcj"
        )
        train_coclustering(
            domain,
            dictionary_name,
            data_table_path,
            [id_variable, timestamp.name] + value_variables,
            triclustering_report_path,
            sample_percentage=triclustering_sample_percentage,
            max_cores=1,
            **kwargs,
        )

        # Simplify triclustering to max_intervals (and max_symbols if not refined)
        simplified_report_path = os.path.join(output_saxo_dir, "triclustering.khcj")
        kh.simplify_coclustering(
            triclustering_report_path,
            simplified_report_path,
            max_part_numbers={timestamp.name: max_intervals}
            | ({id_variable: max_symbols} if not refined else {}),
            **kwargs,
        )

        # Remove full triclustering file
        if os.path.exists(triclustering_report_path):
            os.remove(triclustering_report_path)
        triclustering_report_path = simplified_report_path

        # Get interval bounds from triclustering
        triclustering_results = kh.read_coclustering_results_file(
            triclustering_report_path
        )
        timestamp_triclustering_dimension = (
            triclustering_results.coclustering_report.get_dimension(timestamp.name)
        )
        timestamp_bounds = get_coclustering_bounds(timestamp_triclustering_dimension)
        timestamp_discretization_report_path = triclustering_report_path

    elif timestamp_discretization_method in ["EqualWidth", "EqualFreq"]:
        timestamp_discretization_report_path = os.path.join(
            output_saxo_dir, "discretization.khj"
        )
        kh.train_predictor(
            domain,
            dictionary_name,
            data_table_path,
            "",
            timestamp_discretization_report_path,
            sample_percentage=triclustering_sample_percentage,
            use_complement_as_test=False,
            max_trees=0,
            do_data_preparation_only=True,
            discretization_method=timestamp_discretization_method,
            max_parts=max_intervals,
            max_cores=max_cores,
            detect_format=detect_format,
            header_line=header_line,
            field_separator=field_separator,
            **kwargs,
        )

        # Get interval bounds from unsupervised discretization
        timestamp_discretization_results = kh.read_analysis_results_file(
            timestamp_discretization_report_path
        )
        timestamp_discretization = (
            timestamp_discretization_results.preparation_report.get_variable_statistics(
                timestamp.name
            ).data_grid.dimensions[0]
        )
        timestamp_bounds = get_datagrid_bounds(timestamp_discretization)

    elif isinstance(timestamp_discretization_method, list):
        if max_intervals > 0:
            warnings.warn(
                f"max_intervals was set to {max_intervals} but has no effect with custom timestamp_discretization_method."
            )
        timestamp_bounds = timestamp_discretization_method
        timestamp_discretization_report_path = None
    else:
        raise ValueError(
            f"Timestamp discretization method : {timestamp_discretization_method} not supported."
        )

    saxo_labels = make_saxo_labels(timestamp_bounds)

    # Create recoding rule from interval bounds
    timestamp_discretization = kh.Variable()
    timestamp_discretization.name = f"{timestamp_variable}_interval_bounds"
    timestamp_discretization.label = f"{timestamp_variable} discretization"
    timestamp_discretization.used = False
    timestamp_discretization.type = "Structure"
    timestamp_discretization.object_type = ""
    timestamp_discretization.structure_type = "IntervalBounds"
    timestamp_discretization.rule = str(
        kh.Rule("IntervalBounds", *timestamp_bounds[1:-1])
    )

    discretized_timestamp = kh.Variable()
    discretized_timestamp.name = f"{timestamp_variable}_interval_index"
    discretized_timestamp.label = f"discretized {timestamp_variable}"
    discretized_timestamp.used = False
    discretized_timestamp.type = "Numerical"
    discretized_timestamp.object_type = ""
    discretized_timestamp.structure_type = ""
    discretized_timestamp.rule = str(
        kh.Rule("IntervalIndex", timestamp_discretization, timestamp)
    )

    interval_labels = kh.Variable()
    interval_labels.name = f"{timestamp_variable}_interval_labels"
    interval_labels.label = f"{timestamp_variable} interval labels"
    interval_labels.used = False
    interval_labels.type = "Structure"
    interval_labels.object_type = ""
    interval_labels.structure_type = "VectorC"
    interval_labels.rule = str(kh.Rule("VectorC", *saxo_labels))

    encoded_timestamp = kh.Variable()
    encoded_timestamp.name = f"{timestamp_variable}_encoded"
    encoded_timestamp.label = f"encoded {timestamp_variable}"
    encoded_timestamp.used = False
    encoded_timestamp.type = "Categorical"
    encoded_timestamp.object_type = ""
    encoded_timestamp.structure_type = ""
    encoded_timestamp.rule = str(
        kh.Rule("ValueAtC", interval_labels, discretized_timestamp)
    )

    # Add recoding rule to dictionary
    for v in [
        timestamp_discretization,
        discretized_timestamp,
        interval_labels,
        encoded_timestamp,
    ]:
        domain.get_dictionary(dictionary_name).add_variable(v)

    # Create table extraction rule
    extraction_tables = []
    for label in saxo_labels:
        extraction_table = kh.Variable()
        extraction_table.name = f"Table_{dictionary_name}_{label}"
        extraction_table.type = "Table"
        extraction_table.object_type = dictionary_name
        extraction_table.used = True
        extraction_table.rule = str(
            kh.Rule(
                "TableSelection",
                secondary_table,
                kh.Rule("EQc", encoded_timestamp, label),
            ),
        )

        extraction_tables.append(extraction_table)

    # Adding them to dictionary
    for extraction_table in extraction_tables:
        domain.get_dictionary(root_dictionary_name).add_variable(extraction_table)

    if refined:
        # Train biclustering on each interval
        def train_biclustring(label):
            biclustering_report_path = os.path.join(
                output_saxo_dir, f"full_biclustering_{label}.khcj"
            )
            train_coclustering(
                domain,
                dictionary_name,
                data_table_path,
                [id_variable] + value_variables,
                biclustering_report_path,
                sample_percentage=biclustering_sample_percentage,
                selection_variable=encoded_timestamp.name,
                selection_value=label,
                max_cores=1,
                **kwargs,
            )
            simplified_report_path = os.path.join(
                output_saxo_dir, f"biclustering_{label}.khcj"
            )
            kh.simplify_coclustering(
                biclustering_report_path,
                simplified_report_path,
                max_part_numbers={id_variable: max_symbols},
                **kwargs,
            )
            if os.path.exists(biclustering_report_path):
                os.remove(biclustering_report_path)
            biclustering_report_path = simplified_report_path
            return biclustering_report_path

        with ThreadPoolExecutor(max_workers=max_cores) as executor:
            biclustering_reports_paths = list(
                executor.map(train_biclustring, saxo_labels)
            )
    else:
        assert timestamp_discretization_method == "SAXO"
        biclustering_reports_paths = [timestamp_discretization_report_path] * len(
            saxo_labels
        )

    # Construct SAXO dictionary containing all biclustering
    saxo_domain = deepcopy(domain)

    symbols_map = kh.Variable()
    symbols_map.name = "saxo_symbols"
    symbols_map.label = "symbols list"
    symbols_map.used = False
    symbols_map.type = "Structure"
    symbols_map.structure_type = "VectorC"
    symbols_map.rule = str(kh.Rule("VectorC", *SAXO_SYMBOLS))
    saxo_domain.get_dictionary(root_dictionary_name).add_variable(symbols_map)

    saxo_dictionary_file_path = os.path.join(output_saxo_dir, "saxo.kdic")

    for label, extraction_table, biclustering_report_path in zip(
        saxo_labels, extraction_tables, biclustering_reports_paths
    ):
        if os.path.exists(biclustering_report_path):
            if os.path.exists(saxo_dictionary_file_path):
                os.remove(saxo_dictionary_file_path)
            kh.prepare_coclustering_deployment(
                saxo_domain,
                root_dictionary_name,
                biclustering_report_path,
                extraction_table.name,
                id_variable,
                saxo_dictionary_file_path,
                max_part_numbers={id_variable: max_symbols},
                variables_prefix=f"{label}_",
                build_distance_variables=build_distance_variables,
                **kwargs,
            )
            saxo_domain = kh.read_dictionary_file(saxo_dictionary_file_path)
            saxo_root_dictionary = saxo_domain.get_dictionary(root_dictionary_name)
            og_prediction = saxo_root_dictionary.get_variable(
                f"{label}_{id_variable}PredictedLabel"
            )
            og_prediction.used = False
            saxo_prediction = kh.Variable()
            saxo_prediction.name = label
            saxo_prediction.label = "Predicted saxo symbol"
            saxo_prediction.used = True
            saxo_prediction.type = "Categorical"
            saxo_prediction.rule = str(
                kh.Rule(
                    "ValueAtC",
                    symbols_map,
                    saxo_root_dictionary.get_variable(f"{label}_{id_variable}Index"),
                )
            )
            saxo_domain.get_dictionary(root_dictionary_name).add_variable(
                saxo_prediction
            )
            if build_distance_variables:
                for var in saxo_root_dictionary.variables:
                    if var.name.startswith(f"{label}_{id_variable}Distance"):
                        var.used = False
                saxo_distance = kh.Variable()
                saxo_distance.name = f"{label}_distance"
                saxo_distance.label = "Distance to saxo symbol"
                saxo_distance.used = True
                saxo_distance.type = "Numerical"
                saxo_distance.rule = str(
                    kh.Rule(
                        "ValueAt",
                        saxo_root_dictionary.get_variable(
                            f"{label}_{id_variable}PartDistances"
                        ),
                        saxo_root_dictionary.get_variable(
                            f"{label}_{id_variable}Index"
                        ),
                    )
                )
                saxo_domain.get_dictionary(root_dictionary_name).add_variable(
                    saxo_distance
                )
    saxo_domain.export_khiops_dictionary_file(saxo_dictionary_file_path)

    return (
        saxo_dictionary_file_path,
        biclustering_reports_paths,
        timestamp_discretization_report_path,
    )
