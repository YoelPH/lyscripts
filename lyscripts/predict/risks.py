"""
Predict risks of involvements using the samples that were drawn during the inference
process.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import emcee
import h5py
import lymph
import numpy as np
import pandas as pd
import yaml
from rich.progress import track

from ..helpers import model_from_config, report


def predicted_risk(
    involvement: Dict[str, Dict[str, bool]],
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    samples: np.ndarray,
    t_stage: str,
    midline_ext: bool = False,
    given_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
    given_diagnosis_spsn: Optional[List[float]] = None,
    invert: bool = False,
    description: Optional[str] = None,
    **_kwargs,
) -> np.ndarray:
    """Compute the probability of arriving in a particular `involvement` in a given
    `t_stage` using a `model` with pretrained `samples`. This probability can be
    computed for a `given_diagnosis` that was obtained using a modality with
    specificity & sensitivity provided via `given_diagnosis_spsn`. If the model is an
    instance of `lymph.MidlineBilateral`, one can specify whether or not the primary
    tumor has a `midline_ext`.

    Both the `involvement` and the `given_diagnosis` should be dictionaries like this:

    ```python
    involvement = {
        "ipsi":  {"I": False, "II": True , "III": None , "IV": None},
        "contra: {"I": None , "II": False, "III": False, "IV": None},
    }
    ```

    The returned probability can be `invert`ed.

    Set `verbose` to `True` for a visualization of the progress.
    """
    model.modalities = {"risk": given_diagnosis_spsn}

    # wrap the iteration over samples in a rich progressbar if `verbose`
    enumerate_samples = enumerate(samples)
    if description is not None:
        enumerate_samples = track(
            enumerate_samples,
            description=description,
            total=len(samples),
            console=report,
            transient=True,
        )

    risks = np.zeros(shape=len(samples), dtype=float)

    if isinstance(model, lymph.Unilateral):
        given_diagnosis = {"risk": given_diagnosis["ipsi"]}

        for i,sample in enumerate_samples:
            risks[i] = model.risk(
                involvement=involvement["ipsi"],
                given_params=sample,
                given_diagnoses=given_diagnosis,
                t_stage=t_stage
            )
        return 1. - risks if invert else risks

    elif not isinstance(model, (lymph.Bilateral, lymph.MidlineBilateral)):
        raise TypeError("Model is not a known type.")

    given_diagnosis = {"risk": given_diagnosis}

    for i,sample in enumerate_samples:
        risks[i] = model.risk(
            involvement=involvement,
            given_params=sample,
            given_diagnoses=given_diagnosis,
            t_stage=t_stage,
            midline_extension=midline_ext,
        )
    return 1. - risks if invert else risks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True,
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to the data file if risk is to be compared to prevalence"
    )
    parser.add_argument(
        "--params", default="params.yaml",
        help="Path to parameter file (YAML)"
    )
    parser.add_argument(
        "--risks", default="models/risks.hdf5",
        help="Output path for predicted risks (HDF5 file)"
    )

    args = parser.parse_args()

    with report.status("Read in parameters..."):
        params_path = Path(args.params)
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    if args.data is not None:
        with report.status("Read in training data..."):
            data_path = Path(args.data)
            # Only read in two header rows when using the Unilateral model
            is_unilateral = params["model"]["class"] == "Unilateral"
            header = [0, 1] if is_unilateral else [0, 1, 2]
            DATA = pd.read_csv(data_path, header=header)
            report.success(f"Read in training data from {data_path}")

    with report.status("Loading samples..."):
        model_path = Path(args.model)
        reader = emcee.backends.HDFBackend(model_path, read_only=True)
        SAMPLES = reader.get_chain(flat=True)
        report.success(f"Loaded samples with shape {SAMPLES.shape} from {model_path}")

    with report.status("Set up model..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
        )
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters"
        )

    risks_path = Path(args.risks)
    risks_path.parent.mkdir(exist_ok=True)
    num_risks = len(params["risks"])
    with h5py.File(risks_path, mode="w") as risks_storage:
        for i,scenario in enumerate(params["risks"]):
            risks = predicted_risk(
                model=MODEL,
                samples=SAMPLES,
                description=f"Compute risks for scenario {i+1}/{num_risks}...",
                **scenario
            )
            risks_dset = risks_storage.create_dataset(
                name=scenario["name"],
                data=risks,
            )
            for key,val in scenario.items():
                try:
                    risks_dset.attrs[key] = val
                except TypeError:
                    pass
        report.success(f"Computed risks of {num_risks} scenarios stored at {risks_path}")
