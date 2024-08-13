from typing import List, Any, Optional

import language
from language import (
    ConstantSequenceSegment,
    FixedLengthSequenceSegment,
    ProgramNode,
)

from language.sequence import SequenceSegmentFactory


def vhh_nodes() -> List[ProgramNode]:
    """Constructs a vhh optimization program. Optimizing only the variable region."""
    fix1 = "QVQLQESGGGLVQAGGSLRLSCAAS"
    var1 = "GRTFSEYA"
    fix2 = "MGWFRQAPGKEREFVATI"
    var2 = "SWSGGSTY"
    fix3 = "YTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYC"
    var3 = "AAAGLGTVVSEWDYDYDY"
    fix4 = "WGQGTQVTVSSGS"

    nodes = [
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix1)),
        ProgramNode(sequence_segment=FixedLengthSequenceSegment(len(var1))),
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix2)),
        ProgramNode(sequence_segment=FixedLengthSequenceSegment(len(var2))),
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix3)),
        ProgramNode(sequence_segment=FixedLengthSequenceSegment(len(var3))),
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix4)),
    ]
    return nodes


def vhh_nodes_variable(variable_programs: List[SequenceSegmentFactory]):
    assert len(variable_programs) == 3
    fix1 = "QVQLQESGGGLVQAGGSLRLSCAAS"
    fix2 = "MGWFRQAPGKEREFVATI"
    fix3 = "YTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYC"
    fix4 = "WGQGTQVTVSSGS"

    nodes = [
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix1)),
        variable_programs[0],
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix2)),
        variable_programs[1],
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix3)),
        variable_programs[2],
        ProgramNode(sequence_segment=ConstantSequenceSegment(fix4)),
    ]
    return nodes


def vhh_binder_program(
    target_sequence: str,
    energy_function_terms: List[Any],
    energy_function_weights: List[float],
    var1_program: SequenceSegmentFactory,
    var2_program: SequenceSegmentFactory,
    var3_program: SequenceSegmentFactory,
    target_hotspots: Optional[List[int]] = None,
) -> ProgramNode:
    
    variable_programs = [var1_program, var2_program, var3_program]
    target_program = ProgramNode(sequence_segment=ConstantSequenceSegment(target_sequence))
    vhh_program = ProgramNode(children=vhh_nodes_variable(variable_programs))
    sep_program = ProgramNode(sequence_segment=ConstantSequenceSegment(":"))
    processed_energy_function_terms = []
    # fixme (maks) maybe use kwargs for cleaner implementation
    for term in energy_function_terms:
        if term == "MinimizeInterfaceHotspotDistance" or term == "MaximizeHotspotProbability":
            if target_hotspots is None:
                raise NotImplementedError(
                    "MinimizeInterfaceHotspotDistance needs target hotspot list"
                )
            eft = getattr(language.energy, term)(target_hotspots)
        else:
            eft = getattr(language.energy, term)()
        processed_energy_function_terms.append(eft)

    return ProgramNode(
        energy_function_terms=processed_energy_function_terms,
        energy_function_weights=energy_function_weights,
        children=[vhh_program, sep_program, target_program],
        # NOTE(alex) children_are_different_chains is broken,
        # get around this by using sep_program.
        children_are_different_chains=False,
    )
