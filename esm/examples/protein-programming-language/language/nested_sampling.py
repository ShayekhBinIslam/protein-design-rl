# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from rich.live import Live
from rich.table import Table

from language.data import MetropolisHastingsState
from language.sequence import FixedLengthSequenceSegment
from language.folding_callbacks import FoldingCallback
from language.logging_callbacks import LoggingCallback
from language.program import ProgramNode
import ray

def get_energy(
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


def nested_sampling_step(
    state: MetropolisHastingsState,
    live_points: list,
    num_repeats: int,
    folding_callback: FoldingCallback,
    logging_callback: LoggingCallback,
    verbose: bool = False,
) -> (MetropolisHastingsState, list):

    # Find the lowest energy state
    lowest_state: MetropolisHastingsState = max(live_points, key=lambda x: x.energy)

    # Randomly select a state to mutate
    state_to_mutate: MetropolisHastingsState = np.random.choice(live_points)

    max_energy = lowest_state.energy
    candidate_energy = np.infty
    num_accepted_steps = 0
    #while not accept_candidate:
    while num_accepted_steps < num_repeats:
        candidate: ProgramNode = deepcopy(state_to_mutate.program)
        candidate.mutate()

        candidate_energy, candidate_energy_term_fn_values, folding_output = get_energy(candidate, folding_callback)
        
        # if candidate_energy < state.best_energy:
        #     state.best_energy = candidate_energy

        #accept_candidate = candidate_energy < max_energy
        if candidate_energy < max_energy:
            num_accepted_steps += 1
            
        accept_candidate = num_accepted_steps >= num_repeats

        state = MetropolisHastingsState(program = candidate,
                                        temperature = 0,
                                        annealing_rate = 0,
                                        num_steps = state.num_steps + 1,
                                        energy = candidate_energy,
                                        energy_term_fn_values = candidate_energy_term_fn_values,
                                        best_energy = candidate_energy if candidate_energy < state.best_energy else state.best_energy,
                                        last_accepted_step = 0,
                                        num_accepted = state.num_accepted + 1 if accept_candidate else state.num_accepted
                                        )

        # print(f"Num steps {state.num_steps}, num accepted {state.num_accepted}, accepted {accept_candidate}, candidate energy {candidate_energy}, max energy {max_energy}")

        if logging_callback is not None:
            logging_callback.log(
                candidate, state, folding_output, candidate_energy_term_fn_values, candidate_energy, accept_candidate
            )

    if verbose:
        print(f"Completed {state.num_steps} steps, accepted {state.num_accepted}, highest energy: {lowest_state.energy:.4f}, highest_energy{state.best_energy:.4f}.")

    #state.num_accepted += 1
    
    # Replace the lowest energy state with the candidate
    #state.dead_states.append(lowest_state)
    #state.live_states.remove(lowest_state)
    #state.live_states.append(single_state)

    live_points.remove(lowest_state)
    live_points.append(state)

    return state, live_points
    
    

def make_protomer_node(protomer):
    return ProgramNode(sequence_segment=protomer)


def run_nested_sampling(
    program: ProgramNode,
    num_live_points: int,
    total_num_steps: int,
    num_repeats: int,
    folding_callback: FoldingCallback,
    logging_callback: LoggingCallback,
    progress_verbose_print: bool = False,
) -> (ProgramNode, MetropolisHastingsState):
    # TODO(scandido): Track accept rate.

    if progress_verbose_print:
        print("Starting nested sampling...")
        print(f"Number of live points: {num_live_points}")
        print(f"Total number of steps: {total_num_steps}")

    live_points = []
    best_energy = np.infty
    for i in range(num_live_points):
        new_program = deepcopy(program)
        new_program.reset()

        energy, energy_term_fn_values, folding_output = get_energy(new_program, folding_callback)
        
        if energy < best_energy:
            best_energy = energy

        state = MetropolisHastingsState(program = new_program,
                                        temperature = 0,
                                        annealing_rate = 0,
                                        num_steps = i + 1,
                                        energy = energy,
                                        energy_term_fn_values = energy_term_fn_values,
                                        best_energy = best_energy,
                                        last_accepted_step = 0,
                                        num_accepted = i + 1
                                        )
        live_points.append(state)

        if logging_callback is not None:
            logging_callback.log(
                new_program, state, folding_output, energy_term_fn_values, energy, True
            )

    if progress_verbose_print:
        print("Generated initial live points.")

    while state.num_steps < total_num_steps:
        state, live_points = nested_sampling_step(
            state,
            live_points,
            num_repeats,
            folding_callback,
            logging_callback,
            verbose=progress_verbose_print,
        )

    best_state = max(live_points, key=lambda x: x.energy)
    return state, best_state.program
