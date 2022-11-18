"""
A `streamlit` app for computing, displaying and reproducing prevalence estimates.
"""
import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import lymph
import pandas as pd

from lyscripts.predict.prevalences import compute_observed_prevalence
from lyscripts.predict.utils import clean_pattern
from lyscripts.utils import (
    LymphModel,
    create_model_from_config,
    get_lnls,
    load_data_for_model,
    load_hdf5_samples,
    load_yaml_params,
)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this `streamlit` app.
    """
    parser.add_argument(
        "--message", type=str,
        help="Print our this little message."
    )

    parser.set_defaults(run_main=launch_streamlit)


def get_lnl_pattern_label(selected: Optional[bool] = None) -> str:
    """Return labels for the involvement options of an LNL."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Involved"
    elif not selected:
        return "Healthy"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def get_midline_ext_label(selected: Optional[bool] = None) -> str:
    """Return labels for the options of the midline extension."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Extension"
    elif not selected:
        return "Lateralized"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def launch_streamlit(*_args, discard_args_idx: int = 3, **_kwargs):
    """
    Regardless of the entry point into this script, this function will start
    `streamlit` and pass on the provided command line arguments.
    """
    try:
        from streamlit.web.cli import main as st_main
    except ImportError as mnf_err:
        raise ImportError(
            "Install lyscripts with the `apps` option to install the necessary "
            "requirements for running the streamlit apps."
        ) from mnf_err

    sys.argv = ["streamlit", "run", __file__, "--", *sys.argv[discard_args_idx:]]
    st_main()


def main(args: argparse.Namespace):
    """
    The main function that contains the `streamlit` code and main functionality.
    """
    import streamlit as st

    st.title("Prevalence")

    with st.sidebar:
        model, patient_data, samples = interactive_load(st)

    st.write("---")
    contra_col, ipsi_col = st.columns(2)
    container = {"ipsi": ipsi_col, "contra": contra_col}

    lnls = get_lnls(model)
    is_unilateral = isinstance(model, lymph.Unilateral)

    pattern = {}
    for side in ["ipsi", "contra"]:
        with container[side]:
            pattern[side] = interactive_pattern(st, is_unilateral, lnls, side)

    pattern = clean_pattern(pattern, lnls)
    st.write("---")

    res_tuple = interactive_additional_params(st, model, patient_data)
    t_stage, selected_modality, midline_ext, invert = res_tuple

    observed_prev = compute_observed_prevalence(
        pattern=pattern,
        lnls=lnls,
        data=patient_data,
        modality=selected_modality,
        t_stage=t_stage,
        midline_ext=midline_ext,
        invert=invert,
    )


def interactive_additional_params(
    streamlit: ModuleType,
    model: LymphModel,
    data: pd.DataFrame,
) -> Tuple[str, bool, bool]:
    """
    Allow the user to select T-category, midline extension and whether to invert the
    computed prevalence (meaning computing $1 - p$, when $p$ is the prevalence).

    The respective controls are presented next to each other in three dedicated columns.
    """
    control_cols = streamlit.columns([1,2,1,1])
    t_stage = control_cols[0].selectbox(
        label="T-category",
        options=model.diag_time_dists.keys(),
    )
    modalities_in_data = data.columns.get_level_values(level=0).difference(
        ["patient", "tumor", "positive_dissected", "total_dissected"]
    )
    selected_modality = control_cols[1].selectbox(
        label="Modality",
        options=modalities_in_data
    )
    midline_ext = control_cols[2].radio(
        label="Midline Extension",
        options=[False, None, True],
        index=0,
        format_func=get_midline_ext_label,
        horizontal=True,
    )
    control_cols[3].write("")
    control_cols[3].write("")
    invert = control_cols[3].checkbox(
        label="Invert?",
        help="When selecting this option, 1 - the prevalence will be computed",
    )

    return t_stage, selected_modality, midline_ext, invert


def interactive_load(streamlit):
    """
    Load the YAML file defining the parameters, the CSV file with the patient data
    and the HDF5 file with the drawn model samples interactively.
    """
    params_file = streamlit.file_uploader(
        label="YAML params file",
        type=["yaml", "yml"],
        help="Parameter YAML file containing configurations w.r.t. the model etc.",
    )
    params = load_yaml_params(params_file)
    model = create_model_from_config(params)
    is_unilateral = isinstance(model, lymph.Unilateral)

    streamlit.write("---")

    data_file = streamlit.file_uploader(
        label="CSV file of patient data",
        type=["csv"],
        help="CSV spreadsheet containing lymphatic patterns of progression",
    )
    header_rows = [0,1] if is_unilateral else [0,1,2]
    patient_data = load_data_for_model(data_file, header_rows=header_rows)

    streamlit.write("---")

    samples_file = streamlit.file_uploader(
        label="HDF5 sample file",
        type=["hdf5", "hdf", "h5"],
        help="HDF5 file containing the samples."
    )
    samples = load_hdf5_samples(samples_file)

    return model, patient_data, samples


def interactive_pattern(
    streamlit,
    is_unilateral: bool,
    lnls: List[str],
    side: str
) -> Dict[str, bool]:
    """
    """
    streamlit.subheader(f"{side}lateral")
    side_pattern = {}

    if side == "contra" and is_unilateral:
        return side_pattern

    for lnl in lnls:
        side_pattern[lnl] = streamlit.radio(
            label=f"LNL {lnl}",
            options=[False, None, True],
            index=1,
            key=f"{side}_{lnl}",
            format_func=get_lnl_pattern_label,
            horizontal=True,
        )

    return side_pattern


if __name__ == "__main__":
    if "__streamlit__" in locals():
        parser = argparse.ArgumentParser(description=__doc__)
        _add_arguments(parser)

        args = parser.parse_args()
        main(args)

    else:
        launch_streamlit(discard_args_idx=1)
