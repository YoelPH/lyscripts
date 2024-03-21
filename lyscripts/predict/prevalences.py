"""
Predict prevalences of diagnostic patterns using the samples that were inferred using
the model via MCMC sampling and compare them to the prevalence in the data.

This essentially amounts to computing the data likelihood under the model and comparing
it to the empirical likelihood of a given pattern of lymphatic progression.

Like :py:mod:`.predict.risks`, the computation of the prevalences can be done for
different scenarios. How to define these scenarios can be seen in the
`lynference`_ repository.

.. _lynference: https://github.com/rmnldwg/lynference
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from collections.abc import Generator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rich.progress import track

from lyscripts.decorators import log_state
from lyscripts.predict.utils import complete_pattern
from lyscripts.utils import (
    LymphModel,
    console,
    create_model,
    flatten,
    load_model_samples,
    load_patient_data,
    load_yaml_params,
)

logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the parser."""
    parser.add_argument(
        "model", type=Path,
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "data", type=Path,
        help="Path to the data file to compare prediction and data prevalence"
    )
    parser.add_argument(
        "output", type=Path,
        help="Output path for predicted prevalences (HDF5 file)"
    )
    parser.add_argument(
        "--thin", default=1, type=int,
        help="Take only every n-th sample"
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )

    parser.set_defaults(run_main=main)


def get_match_idx(
    match_idx,
    pattern: dict[str, bool | None],
    data: pd.DataFrame,
    lnls: list[str],
    invert: bool = False,
) -> pd.Series:
    """Get indices of rows in the ``data`` where the diagnose matches the ``pattern``.

    This uses the ``match_idx`` as a starting point and updates it according to the
    ``pattern``. If ``invert`` is set to ``True``, the function returns the inverted
    indices.

    >>> pattern = {"II": True, "III": None}
    >>> lnls = ["II", "III"]
    >>> data = pd.DataFrame.from_dict({
    ...     "II":  [True, False],
    ...     "III": [False, False],
    ... })
    >>> get_match_idx(True, pattern, data, lnls)
    0     True
    1    False
    Name: II, dtype: bool
    """
    for lnl in lnls:
        if lnl not in pattern or pattern[lnl] is None:
            continue
        if invert:
            match_idx |= data[lnl] != pattern[lnl]
        else:
            match_idx &= data[lnl] == pattern[lnl]

    return match_idx


def does_t_stage_match(data: pd.DataFrame, t_stage: str | None) -> pd.Series:
    """Return indices of the ``data`` where ``t_stage`` of the patients matches."""
    if t_stage is None:
        return pd.Series([True] * len(data))
    return data["tumor", "1", "t_stage"] == t_stage


def does_midline_ext_match(
    data: pd.DataFrame,
    midline_ext: bool | None = None
) -> pd.Index:
    """Return indices of ``data`` where ``midline_ext`` of the patients matches."""
    if midline_ext is None or data.columns.nlevels == 2:
        return True

    try:
        return data["info", "tumor", "midline_extension"] == midline_ext
    except KeyError as key_err:
        raise KeyError(
            "Data does not seem to have midline extension information"
        ) from key_err


def get_midline_ext_prob(data: pd.DataFrame, t_stage: str) -> float:
    """Get the prevalence of midline extension from ``data`` for ``t_stage``."""
    if data.columns.nlevels == 2:
        return None

    has_matching_t_stage = does_t_stage_match(data, t_stage)
    eligible_data = data[has_matching_t_stage]
    has_matching_midline_ext = does_midline_ext_match(eligible_data, midline_ext=True)
    matching_data = eligible_data[has_matching_midline_ext]
    return len(matching_data) / len(eligible_data)


def create_patient_row(
    pattern: dict[str, dict[str, bool]],
    t_stage: str,
    modality: str = "prev",
    midline_ext: bool | None = None,
) -> pd.DataFrame:
    """Create pandas ``DataFrame`` row from arguments.

    It is created from the specified involvement ``pattern``, along with their
    ``t_stage`` and ``midline_ext`` (if provided). If ``midline_ext`` is not provided,
    the function creates two patient rows. One of a patient _with_ and one of a patient
    _without_ a midline extention.
    """
    flat_pattern = flatten({modality: pattern})
    patient_row = pd.DataFrame(flat_pattern, index=[0])
    patient_row["tumor", "1", "t_stage"] = t_stage

    if midline_ext is not None:
        patient_row["tumor", "1", "extension"] = midline_ext

    return patient_row


@log_state()
def compute_observed_prevalence(
    pattern: dict[str, dict[str, bool]],
    data: pd.DataFrame,
    lnls: list[str],
    t_stage: str = "early",
    modality: str = "max_llh",
    midline_ext: bool | None = None,
    invert: bool = False,
    **_kwargs,
):
    """Extract prevalence of a ``pattern`` from the ``data``.

    This also considers the ``t_stage`` of the patients. If the ``data`` contains
    bilateral information, one can choose to factor in whether or not the patient's
    ``midline_ext`` should be considered as well.

    By giving a list of ``lnls``, one can restrict the matching algorithm to only those
    lymph node levels that are provided via this list.

    When ``invert`` is set to ``True``, the function returns 1 minus the prevalence.
    """
    pattern = complete_pattern(pattern, lnls)

    has_matching_t_stage = does_t_stage_match(data, t_stage)
    has_matching_midline_ext = does_midline_ext_match(data, midline_ext)

    eligible_data = data.loc[has_matching_t_stage & has_matching_midline_ext, modality]
    eligible_data = eligible_data.dropna(axis="index", how="all")

    # filter the data by the LNL pattern they report
    do_lnls_match = not invert
    if data.columns.nlevels == 2:
        do_lnls_match = get_match_idx(
            do_lnls_match,
            pattern["ipsi"],
            eligible_data,
            lnls=lnls,
            invert=invert,
        )
    else:
        for side in ["ipsi", "contra"]:
            do_lnls_match = get_match_idx(
                do_lnls_match,
                pattern[side],
                eligible_data[side],
                lnls=lnls,
                invert=invert
            )

    try:
        matching_data = eligible_data.loc[do_lnls_match]
    except KeyError:
        # return X, X if no actual pattern was selected
        len_matching_data = 0 if invert else len(eligible_data)
        return len_matching_data, len(eligible_data)

    return len(matching_data), len(eligible_data)


def compute_predicted_prevalence(
    loaded_model: LymphModel,
    given_params: np.ndarray,
    midline_ext: bool,
    midline_ext_prob: float = 0.3,
) -> float:
    """
    Given a ``loaded_model`` compute the prevalence (i.e. likelihood) of data.

    If ``midline_ext`` is ``True``, the prevalence is computed for the case where the
    tumor does extend over the mid-sagittal line, while if it is ``False``, it is
    predicted for the case of a lateralized tumor.

    If ``midline_ext`` is set to ``None``, the prevalence is marginalized over both
    cases, assuming the provided ``midline_ext_prob``.
    """
    if False: #isinstance(loaded_model, MidlineBilateral):
        loaded_model.check_and_assign(given_params)
        if midline_ext is None:
                # marginalize over patients with and without midline extension
            prevalence = (
                    midline_ext_prob * loaded_model.ext.likelihood(log=False) +
                    (1. - midline_ext_prob) * loaded_model.noext.likelihood(log=False)
                )
        elif midline_ext:
            prevalence = loaded_model.ext.likelihood(log=False)
        else:
            prevalence = loaded_model.noext.likelihood(log=False)
    else:
        prevalence = loaded_model.likelihood(
            given_param_args=given_params,
            log=False,
        )

    return prevalence


@log_state()
def generate_predicted_prevalences(
    pattern: dict[str, dict[str, bool]],
    model: LymphModel,
    samples: np.ndarray,
    t_stage: str = "early",
    midline_ext: bool | None = None,
    midline_ext_prob: float = 0.3,
    modality_spsn: list[float] | None = None,
    invert: bool = False,
    **_kwargs,
) -> Generator[float, None, None]:
    """Compute the prevalence of a given ``pattern`` of lymphatic progression using a
    ``model`` and trained ``samples``.

    Do this computation for the specified ``t_stage`` and whether or not the tumor has
    a ``midline_ext``. `modality_spsn` defines the values for specificity & sensitivity
    of the diagnostic modality for which the prevalence is to be computed. Default is
    a value of 1 for both.

    Use ``invert`` to compute 1 - p.
    """
    lnls = list(model.graph.lnls.keys())
    pattern = complete_pattern(pattern, lnls)

    if modality_spsn is None:
        model.modalities = {"prev": [1., 1.]}
    else:
        model.modalities = {"prev": modality_spsn}

    patient_row = create_patient_row(
        pattern=pattern,
        t_stage=t_stage,
        midline_ext=midline_ext,
    )
    model.load_patient_data(patient_row, mapping=lambda x: x)

    # compute prevalence as likelihood of diagnose `prev`, which was defined above
    for sample in samples:
        prevalence = compute_predicted_prevalence(
            loaded_model=model,
            given_params=sample,
            midline_ext=midline_ext,
            midline_ext_prob=midline_ext_prob,
        )
        yield (1. - prevalence) if invert else prevalence


def main(args: argparse.Namespace):
    """Function to run the risk prediction routine."""
    params = load_yaml_params(args.params)
    model = create_model(params)
    samples = load_model_samples(args.model)
    data = load_patient_data(args.data)

    args.output.parent.mkdir(exist_ok=True)
    num_prevalences = len(params["prevalences"])
    with h5py.File(args.output, mode="w") as prevalences_storage:
        for i,scenario in enumerate(params["prevalences"]):
            prevs_gen = generate_predicted_prevalences(
                model=model,
                samples=samples[::args.thin],
                midline_ext_prob=get_midline_ext_prob(data, scenario["t_stage"]),
                **scenario
            )
            prevs_progress = track(
                prevs_gen,
                total=len(samples[::args.thin]),
                description=f"Compute prevalences for scenario {i+1}/{num_prevalences}...",
                console=console,
                transient=True,
            )
            prevs_arr = np.array(list(p for p in prevs_progress))
            prevs_h5dset = prevalences_storage.create_dataset(
                name=scenario["name"],
                data=prevs_arr,
            )
            num_match, num_total = compute_observed_prevalence(
                data=data,
                lnls=len(model.get_params()),
                **scenario,
            )
            for key,val in scenario.items():
                try:
                    prevs_h5dset.attrs[key] = val
                except TypeError:
                    pass

            prevs_h5dset.attrs["num_match"] = num_match
            prevs_h5dset.attrs["num_total"] = num_total

        logger.info(
            f"Computed prevalences of {num_prevalences} scenarios stored at "
            f"{args.output}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
