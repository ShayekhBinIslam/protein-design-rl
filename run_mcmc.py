import os
import sys
from pathlib import Path
from copy import deepcopy
sys.path.append('dockq-proxy/src/proxy_al')

import hydra
import numpy as np
from Bio import SeqIO
from language import EsmFoldv1
from language.data import MetropolisHastingsState
from language.folding_callbacks import FoldingCallback
from language.logging_callbacks import LoggingCallback
from language.program import ProgramNode
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.live import Live
from rich.table import Table

import src
from utils.sampling import PTMProxy
from helper import SequenceLogger


plotter = None


def get_energy_proxy(
    candidate: ProgramNode,
    proxy,
) -> (float, list):
    sequence, _ = candidate.get_sequence_and_set_residue_index_ranges()
    candidate_proxy_ptm = proxy.evaluate(sequence).item()
    # candidate_proxy_ptm = proxy.evaluate(sequence)
    # candidate_proxy_ptm = candidate_proxy_ptm[0][0]
    candidate_energy = 1.0 - candidate_proxy_ptm

    # Create a dummy list of energy terms
    name_prefix = "mcmc"
    candidate_energy_term_fn_values = [
        (
            f"{name_prefix}:{type(term).__name__}",
            weight,
            0.0,
        )
        for weight, term in zip(
            candidate.energy_function_weights, candidate.energy_function_terms
        )
    ]

    folding_output = None
    # print('candidate_proxy_ptm', candidate_proxy_ptm)
    return (
        candidate_proxy_ptm,
        candidate_energy,
        candidate_energy_term_fn_values,
        folding_output,
    )


def metropolis_hastings_step(
    state: MetropolisHastingsState,
    proxy,
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

    candidate_proxy_ptm, energy, energy_term_fn_values, folding_output = (
        get_energy_proxy(candidate, proxy)
    )
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

    new_state = MetropolisHastingsState(
        program=candidate if accept_candidate else state.program,
        temperature=temperature,
        annealing_rate=state.annealing_rate,
        num_steps=state.num_steps + 1,
        energy=energy if accept_candidate else state.energy,
        best_energy=min(energy, state.energy) if state.energy else energy,
        energy_term_fn_values=(
            energy_term_fn_values if accept_candidate else state.energy_term_fn_values
        ),
        last_accepted_step=(
            state.num_steps if accept_candidate else state.last_accepted_step
        ),
        num_accepted=state.num_accepted + 1 if accept_candidate else state.num_accepted,
    )

    if logging_callback is not None and state.energy is not None:
        logging_callback.add(
            {
                "sequence": [candidate.get_sequence_and_set_residue_index_ranges()[0]],
                "round": [int(new_state.num_steps)],
                "proxy_ptm": [candidate_proxy_ptm],
            }
        )

        if int(new_state.num_steps) % 100 == 0:  
			# Avoid GCP access issues
            logging_callback.save()

    return new_state


def run_simulated_annealing(
    cfg,
    program: ProgramNode,
    initial_temperature: float,
    annealing_rate: float,
    total_num_steps: int,
    folding_callback: FoldingCallback,
    logging_callback: LoggingCallback,
    display_progress: bool = True,
    progress_verbose_print: bool = False,
    proxy=None,
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
        table.add_row("Proxy PTM", "", f"{1.0 - state.energy:.2f}")
        table.add_row("Iterations", "", f"{state.num_steps} / {total_num_steps}")
        plotter.update({"Proxy PTM": 1.0 - state.energy})
        if state.num_steps % 10 == 0:
            plotter.send()
        return table

    candidate_proxy_ptm, _, _, _ = get_energy_proxy(program, proxy)
    if logging_callback is not None:
        print("Logging")
        logging_callback.add(
            {
                "sequence": [program.get_sequence_and_set_residue_index_ranges()[0]],
                "round": [1],
                "proxy_ptm": [candidate_proxy_ptm],
            }
        )

    with Live() as live:
        for _ in range(1, total_num_steps + 1):
            state = metropolis_hastings_step(
                state,
                proxy,
                logging_callback,
                verbose=progress_verbose_print,
            )
            if display_progress:
                live.update(_generate_table(state))

    return state.program, state


proxy_base = Path(os.path.dirname(src.__file__)).parent
config_path = proxy_base/'configs/proxy_al'
print(f"config_path: {config_path}")

@hydra.main(version_base=None, config_path=str(config_path), config_name="unconditional_base_config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    print(cfg_dict)
    global plotter
    plotter = PlotLosses(outputs=[MatplotlibPlot(figpath=cfg.experiment.plot_file)])

    proxy = PTMProxy(checkpoint_dir=cfg.checkpoint_dir, checkpoint_name=cfg.checkpoint_name)

    seq_logger = SequenceLogger(
        ["sequence", "round", "proxy_ptm"],
        cfg.experiment.save_filename,
        cfg.experiment.logdir,
    )

    annealing_rate = (cfg.chain.T_min / cfg.chain.T_max) ** ( 1 / cfg.chain.total_num_steps )
    
    optimized_programs = [
        run_simulated_annealing(
            cfg=cfg,
            program=hydra.utils.instantiate(cfg.program),
            initial_temperature=cfg.chain.T_max,
            annealing_rate=annealing_rate,
            total_num_steps=cfg.chain.total_num_steps,
            proxy=proxy,
            folding_callback=None,
            logging_callback=seq_logger,
            display_progress=cfg.chain.display_progress,
            progress_verbose_print=cfg.chain.progress_verbose_print,
        )
        for i in range(cfg.chain.num_chains)
    ]

    for i, (optimized_program, optimized_state) in enumerate(optimized_programs):
        sequence = optimized_program.get_sequence_and_set_residue_index_ranges()[0]
        print(f"Chain {i} final sequence = {sequence}")


if __name__ == "__main__":
    main()
