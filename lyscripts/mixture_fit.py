"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and the mixture model.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import os
from collections import namedtuple

try:
    from multiprocess import Pool
except ModuleNotFoundError:
    from multiprocessing import Pool

from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from lymph import models
from lymixture import LymphMixture
from rich.progress import Progress, TimeElapsedColumn, track

from lyscripts.utils import (
    create_mixture,
    load_patient_data,
    load_yaml_params,
    to_numpy,
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
    """Add arguments to a ``subparsers`` instance and run its main function when chosen.

    This is called by the parent module that is called via the command line.
    """
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to training data files"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Path to the HDF5 file to store the results in"
    )
    parser.add_argument(
        "--history", type=Path, nargs="?",
        help="Path to store the burnin history in (as CSV file)."
    )

    parser.add_argument(
        "-w", "--walkers-per-dim", type=int, default=10,
        help="Number of walkers per dimension",
    )
    parser.add_argument(
        "-b", "--burnin", type=int, nargs="?",
        help="Number of burnin steps. If not provided, sampler runs until convergence."
    )
    parser.add_argument(
        "--check-interval", type=int, default=100,
        help="Check convergence every `check_interval` steps."
    )
    parser.add_argument(
        "--trust-fac", type=float, default=50.,
        help="Factor to trust the autocorrelation time for convergence."
    )
    parser.add_argument(
        "--rel-thresh", type=float, default=0.05,
        help="Relative threshold for convergence."
    )
    parser.add_argument(
        "-n", "--nsteps", type=int, default=100,
        help="Number of MCMC samples to draw, irrespective of thinning."
    )
    parser.add_argument(
        "-t", "--thin", type=int, default=10,
        help="Thinning factor for the MCMC chain."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )
    parser.add_argument(
        "-c", "--cores", type=int, nargs="?",
        help=(
            "Number of parallel workers (CPU cores/threads) to use. If not provided, "
            "it will use all cores. If set to zero, multiprocessing will not be used."
        )
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Seed value to reproduce the same sampling round."
    )

    parser.set_defaults(run_main=main)


MIXTURE = None

def log_prob_fn(theta: np.array) -> float:
    """log probability function using global variables because of pickling."""
    return MIXTURE.likelihood(use_complete = True, given_resps = MIXTURE.get_resps(norm = True))


def check_convergence(params_history, likelihood_history, steps_back_list, absolute_tolerance = 0.01):
    current_params = params_history[-1]
    current_likelihood = likelihood_history[-1]
    for steps_back in steps_back_list:
        previous_params = params_history[-steps_back - 1]
        if np.allclose(to_numpy(current_params), to_numpy(previous_params)):
            logger.info(f"Converged after {len(params_history)} steps. due to parameter similarity")
            return True  # Return True if any of the steps is close
        elif np.isclose(current_likelihood, likelihood_history[-steps_back - 1],rtol = 0, atol = absolute_tolerance):
            logger.info(f"Converged after {len(params_history)} steps. due to likelihood similarity")
            return True
    return False



def run_burnin(
    sampler: emcee.EnsembleSampler,
    burnin: int | None = None,
    check_interval: int = 100,
    trust_fac: float = 50.0,
    rel_thresh: float = 0.05,
) -> BurninHistory:
    """Run the burnin phase of the MCMC sampling.

    This will run the sampler for ``burnin`` steps or (if ``burnin`` is `None`) until
    convergence is reached. The convergence criterion is based on the autocorrelation
    time of the chain, which is computed every `check_interval` steps. The chain is
    considered to have converged if the autocorrelation time is smaller than
    `trust_fac` times the number of iterations and the relative change in the
    autocorrelation time is smaller than `rel_thresh`.

    The samples of the burnin phase will be stored, such that one can resume a
    cancelled run. Also, metrics collected during the burnin phase will be returned
    in a :py:obj:`.BurninHistory` namedtuple. This may be used for plotting and
    diagnostics.
    """
    state = get_starting_state(sampler)
    history = BurninHistory([], [], [], [])
    num_accepted = 0

    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            description="[blue]INFO     [/blue]Burn-in phase ",
            total=burnin,
        )
        while sampler.iteration < (burnin or np.inf):
            for state in sampler.sample(state, iterations=check_interval):
                progress.update(task, advance=1)

            new_acor_time = sampler.get_autocorr_time(tol=0).mean()
            old_acor_time = history.acor_times[-1] if len(history.acor_times) > 0 else np.inf

            new_accept_frac = (
                (np.sum(sampler.backend.accepted) - num_accepted)
                / (sampler.nwalkers * check_interval)
            )
            num_accepted = np.sum(sampler.backend.accepted)

            history.steps.append(sampler.iteration)
            history.acor_times.append(new_acor_time)
            history.accept_fracs.append(new_accept_frac)
            history.max_log_probs.append(np.max(state.log_prob))

            is_converged = burnin is None
            is_converged &= new_acor_time * trust_fac < sampler.iteration
            is_converged &= np.abs(new_acor_time - old_acor_time) / new_acor_time < rel_thresh

            if is_converged:
                break

    if is_converged:
        logger.info(f"Converged after {sampler.iteration} steps.")
    logger.info(f"Acceptance fraction: {sampler.acceptance_fraction.mean():.2%}")
    return history


def run_sampling(
    sampler: emcee.EnsembleSampler,
    nsteps: int,
    thin: int,
) -> None:
    """Run the MCMC sampling phase to produce `nsteps` samples.

    This sampling will definitely produce `nsteps` samples, irrespective of the `thin`
    parameter, which controls how many steps in between two stored samples are skipped.
    The samples will be stored in the backend of the `sampler`.

    Note that this will reset the `sampler`'s backend, assuming the stored samples are
    from the burnin phase.
    """
    state = get_starting_state(sampler)
    sampler.backend.reset(sampler.nwalkers, sampler.ndim)

    for _sample in track(
        sequence=sampler.sample(state, iterations=nsteps * thin, thin=thin, store=True),
        description="[blue]INFO     [/blue]Sampling phase",
        total=nsteps * thin,
    ):
        continue


def run_EM():
    """Run the EM algorithm to determine the optimal parameters.
    """
    is_converged = False
    iteration = 0
    params_history = []
    likelihood_history = []
    params_history.append(params.copy())
    likelihood_history.append(MIXTURE.likelihood(use_complete=False))
    # Number of steps to look back for convergence
    look_back_steps = 3

    while not is_converged:
        print(iteration)
        print(likelihood_history[-1])
        latent = LymphMixture.em.expectation(MIXTURE, params)
        params = LymphMixture.em.maximization(MIXTURE, latent)
        
        # Append current params and likelihood to history
        params_history.append(params.copy())
        likelihood_history.append(MIXTURE.likelihood(use_complete=False))
        
        # Check if converged
        if iteration >= 3:  # Ensure enough history is available
            is_converged = check_convergence(params_history, likelihood_history,list(range(1,look_back_steps+1)))

    return params_history, likelihood_history

def main(args: argparse.Namespace) -> None:
    """Main function to run the EM algorithm for a mixture model"""
    # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
    os.environ["OMP_NUM_THREADS"] = "1"

    params = load_yaml_params(args.params)
    inference_data = load_patient_data(args.input)

    # ugly, but necessary for pickling
    global MIXTURE
    MIXTURE = create_mixture(params)

    mapping = params["model"].get("mapping", None)
    if isinstance(MIXTURE, models.Unilateral):
        side = params["model"].get("side", "ipsi")
        MIXTURE.load_patient_data(inference_data, side=side, mapping=mapping)
    else:
        raise "Only Unilateral has been implemented so far"

    # emcee does not support numpy's new random number generator yet.
    rng = np.random.default_rng(params["em"].get("seed", 42))
    starting_values = {k: rng.uniform() for k in MIXTURE.get_params()}
    MIXTURE.set_params(**starting_values)
    MIXTURE.normalize_mixture_coefs()
    params_history, likelihood_history = run_EM()    
    
    if args.history is not None:
        logger.info(f"Saving history to {args.history}.")
        burnin_history_df = pd.DataFrame(burnin_history._asdict()).set_index("steps")
        burnin_history_df.to_csv(args.history, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
