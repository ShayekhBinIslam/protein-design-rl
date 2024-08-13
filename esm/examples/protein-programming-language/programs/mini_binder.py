from typing import List, Any

import language
from language import ConstantSequenceSegment, ProgramNode


def mini_binder_program(
    target_sequence: str,
    energy_function_terms: List[Any],
    energy_function_weights: List[float],
    binder_sequence: Any,
) -> ProgramNode:
    target_program = ProgramNode(sequence_segment=ConstantSequenceSegment(target_sequence))
    binder_program = ProgramNode(sequence_segment=binder_sequence)
    sep_program = ProgramNode(sequence_segment=ConstantSequenceSegment(":"))
    energy_function_terms = [getattr(language.energy, term)() for term in energy_function_terms]
    return ProgramNode(
        energy_function_terms=energy_function_terms,
        energy_function_weights=energy_function_weights,
        children=[binder_program, sep_program, target_program],
        # NOTE(alex) children_are_different_chains is broken,
        # get around this by using sep_program.
        children_are_different_chains=False,
    )
