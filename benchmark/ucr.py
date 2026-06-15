import os
import pickle

from aeon.datasets import load_classification
from sklearn import config_context
from sklearn.calibration import LabelEncoder

from saxo.sklearn import SAXO

datasets = [
    "StarLightCurves",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "ECGFiveDays",
    "MoteStrain",
    "CinCECGTorso",
    "MedicalImages",
    "WordSynonyms",
    "TwoPatterns",
    "CBF",
    "FaceAll",
    "Symbols",
    "50Words",
    "Wafer",
    "Yoga",
    "FacesUCR",
    "CricketX",
    "CricketY",
    "CricketZ",
]

results = []

output_path = "ucr-output"

if not os.path.exists(output_path):
    os.mkdir(output_path)

for dataset in datasets:
    result = {}

    X, y = load_classification(dataset)
    y = LabelEncoder().fit_transform(y)

    saxo = SAXO(max_symbols=26, n_jobs=-1).fit(X)
    result["saxo_cost"] = saxo.results_.cost
    result["saxo_null_cost"] = saxo.results_.null_cost
    result["modl_cost"] = saxo.results_.timestamp_discretization_report.cost
    result["modl_null_cost"] = saxo.results_.timestamp_discretization_report.null_cost

    with config_context(transform_output="pandas"):
        saxo.transform(X).to_csv(
            os.path.join(output_path, dataset + ".txt"), index=False, sep="\t"
        )

    print(result)

    with open(os.path.join(output_path, dataset + ".pickle"), "wb") as f:
        pickle.dump(result, f)

    results.append(result)
