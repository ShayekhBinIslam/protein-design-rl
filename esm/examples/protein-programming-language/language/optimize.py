# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from rich.live import Live
from rich.table import Table

from language.data import MetropolisHastingsState
from language.folding_callbacks import FoldingCallback
from language.logging_callbacks import LoggingCallback
from language.program import ProgramNode
import ray
import omegaconf

# For importing proxy model
import hydra
from lightning import LightningDataModule, LightningModule, Trainer
import pandas as pd


# Import model for proxy MCMC
from src.data.atlas_datamodule_v0 import AtlasDataset
from src.models.reg_llm_v0 import RegressionLLMv0

# TODO: Make this less hacky
# path = "/efs/users/pablo_lemos_ce97f96/proxy_atlas/config.yaml"
# cfg = omegaconf.OmegaConf.load(path)
# model: LightningModule = hydra.utils.instantiate(cfg.model)
# trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
# datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

def get_proxy_ptm(sequence):
    data = pd.DataFrame({'sequence': [sequence]})
    datamodule.data_test = AtlasDataset(cfg=cfg, df=data)
    log = trainer.predict(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path='last')
    return log[0]['seq_scalar'].item()



def get_energy_esmfold(
        candidate: ProgramNode,
        folding_callback: FoldingCallback
) -> (float, list):

    sequence, _ = candidate.get_sequence_and_set_residue_index_ranges()
    folding_output = ray.get(folding_callback.fold.remote(sequence))

    energy_term_fns = candidate.get_energy_term_functions()
    candidate_energy_term_fn_values = [
        (name, weight, energy_fn(folding_output)) for name, weight, energy_fn in energy_term_fns
    ]

    candidate_energy: float = sum(
        [weight * value for _, weight, value in candidate_energy_term_fn_values]
    )

    return candidate_energy, candidate_energy_term_fn_values, folding_output


def get_energy_proxy(
        candidate: ProgramNode,
        folding_callback: FoldingCallback
) -> (float, list):

    sequence, _ = candidate.get_sequence_and_set_residue_index_ranges()
    df = pd.DataFrame(data={'sequence': sequence})

    datamodule.data_test = AtlasDataset(cfg=proxy_cfg, df=df)
    
    log = trainer.test(model=model, datamodule=datamodule, ckpt_path='last')
    candidate_energy = log[0]['seq_scalar']
    
    # Create a dummy list of energy terms
    candidate_energy_term_fn_values = [
    (
        f"{name_prefix}:{type(term).__name__}",
        weight,
        0.0,
    )
    for weight, term in zip(candidate.energy_function_weights, candidate.energy_function_terms)
    ]
    
    folding_output = None

    print('Energy', candidate_energy)

    return candidate_energy, candidate_energy_term_fn_values, folding_output


def metropolis_hastings_step(
    state: MetropolisHastingsState,
    folding_callback: FoldingCallback,
    logging_callback: LoggingCallback,
    verbose: bool = False,
) -> MetropolisHastingsState:
    # TODO(alex): The temperature here with the annealing rate is nothing like
    # what is in the paper. The paper uses (T_i = (T_min / T_max)^{i / M}. This
    # cannot be implemented with the current code. This also does not make
    # sense as it should be something like
    #   T_i = T_max * (T_min / T_max)^{i / M}
    # Otherwise the T_max means nothing...
    # The old thing is
    temperature = state.temperature * state.annealing_rate
    # We now ignore the annealing rate and implement this system
    # temperature = state.temperature

    candidate: ProgramNode = deepcopy(state.program)
    candidate.mutate()

    energy, energy_term_fn_values, folding_output = get_energy_esmfold(candidate=candidate, folding_callback=folding_callback)
    #energy, energy_term_fn_values, folding_output = get_energy_proxy(candidate=candidate, folding_callback=None)

    accept_candidate = False
    if state.energy is None:
        accept_candidate = True
    else:
        energy_differential: float = -energy + state.energy
        accept_probability: float = np.clip(
            # NOTE(scandido): We approximate the ratio of transition probabilities from
            # current to candidate vs. candidate to current to be equal, which is
            # approximately correct.
            np.exp(energy_differential / temperature),
            a_min=None,
            a_max=1.0,
        )
        accept_candidate: bool = np.random.uniform() < accept_probability

    if logging_callback is not None:
        logging_callback.log(
            candidate, state, folding_output, energy_term_fn_values, energy, accept_candidate
        )

    """
    if accept_candidate:
        sequence, _ = candidate.get_sequence_and_set_residue_index_ranges()
        if verbose:
            nice_energies = ", ".join(
                [f"{name}: {value:0.2f}" for name, _, value in energy_term_fn_values]
            )
            print(
                f"Step: {state.num_steps} Accepted {energy:.2f} with {nice_energies} with sequence {sequence}."
            )
    """

    return MetropolisHastingsState(
        program=candidate if accept_candidate else state.program,
        temperature=temperature,
        annealing_rate=state.annealing_rate,
        num_steps=state.num_steps + 1,
        energy=energy if accept_candidate else state.energy,
        best_energy=min(energy, state.energy) if state.energy else energy,
        energy_term_fn_values=energy_term_fn_values
        if accept_candidate
        else state.energy_term_fn_values,
        last_accepted_step=state.num_steps if accept_candidate else state.last_accepted_step,
        num_accepted=state.num_accepted + 1 if accept_candidate else state.num_accepted,
    )


def run_simulated_annealing(
    program: ProgramNode,
    initial_temperature: float,
    annealing_rate: float,
    total_num_steps: int,
    folding_callback: FoldingCallback,
    logging_callback: LoggingCallback,
    display_progress: bool = True,
    progress_verbose_print: bool = False,
) -> (ProgramNode, MetropolisHastingsState):
    # TODO(scandido): Track accept rate.

    state = MetropolisHastingsState(
        program=program,
        temperature=initial_temperature,
        annealing_rate=annealing_rate,
        num_steps=0,
        energy=None,
        best_energy=None,
        energy_term_fn_values=None,
        last_accepted_step=-1,
        num_accepted=0,
    )

    def _generate_table(state):
        table = Table()
        table.add_column("Energy name")
        table.add_column("Weight")
        table.add_column("Value")
        if state.energy_term_fn_values is None:
            return table
        for name, weight, value in state.energy_term_fn_values:
            table.add_row(name, f"{weight:.2f}", f"{value:.2f}")
        table.add_row("Energy", "", f"{state.energy:.2f}")
        table.add_row("Iterations", "", f"{state.num_steps} / {total_num_steps}")
        return table

    with Live() as live:
        for _ in range(1, total_num_steps + 1):
            state = metropolis_hastings_step(
                state,
                folding_callback,
                logging_callback,
                verbose=progress_verbose_print,
            )
            if display_progress:
                live.update(_generate_table(state))

    return state.program, state
