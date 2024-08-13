from typing import List, Any

import language
from language import FixedLengthSequenceSegment, ProgramNode

# Ref: free_hallucination.py, mini_binder.py

def fixed_length_unconditional_program(
    sequence_length: int,
    energy_function_terms: List[Any],
    energy_function_weights: List[float],
) -> ProgramNode:
    sequence = FixedLengthSequenceSegment(sequence_length)
    energy_function_terms = [getattr(language.energy, term)() for term in energy_function_terms]
    return ProgramNode(
        energy_function_terms=energy_function_terms,
        energy_function_weights=energy_function_weights,
        sequence_segment=sequence,
    )