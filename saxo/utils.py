from itertools import pairwise

from khiops import core as kh


def get_coclustering_bounds(dimension: kh.CoclusteringDimension):
    leaves_bounds = []
    for cluster in dimension.clusters:
        if cluster.is_leaf:
            if len(leaves_bounds) == 0:
                leaves_bounds.append(cluster.leaf_part.lower_bound)
            leaves_bounds.append(cluster.leaf_part.upper_bound)
    return leaves_bounds


def get_datagrid_bounds(dimension: kh.DataGridDimension):
    bounds = []
    for part in dimension.partition:
        if len(bounds) == 0:
            bounds.append(part.lower_bound)
        bounds.append(part.upper_bound)
    return bounds


def make_saxo_labels(bounds: list[int]) -> list[str]:
    return (
        [f"[{bounds[0]},{bounds[1]}]"]
        + [f"]{lower},{upper}]" for lower, upper in pairwise(bounds[1:-1])]
        + [f"]{bounds[-2]},{bounds[-1]}]"]
    )
