from typing import List
from datetime import datetime

import ray
from biotite.structure import AtomArray

from language import (
    ConstantSequenceSegment,
    EsmFoldv1,
    FixedLengthSequenceSegment,
    MaximizePLDDT,
    MaximizePTM,
    MaximizeIPTM,
    MaximizeInterfaceSize,
    MinimizeInterfaceHotspotDistance,
    MinimizeSurfaceHydrophobics,
    MinimizeBinderSurfaceHydrophobics,
    MinimizeBinderRMSD,
    MaximizeBinderPTM,
    MaximizeBinderPLDDT,
    ProgramNode,
    pdb_file_to_atomarray,
    run_simulated_annealing,
    sequence_from_atomarray,
)
from language.logging_callbacks import S3Logger


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


def vhh_program() -> ProgramNode:
    nodes = vhh_nodes()
    return ProgramNode(
        energy_function_terms=[MaximizePTM(), MaximizePLDDT(), MinimizeSurfaceHydrophobics()],
        energy_function_weights=[1.0, 1.0, 1.0],
        children=nodes,
    )


def vhh_binder_program(pdb_file) -> ProgramNode:
    target_atoms: AtomArray = pdb_file_to_atomarray(pdb_file)
    target_sequence = sequence_from_atomarray(target_atoms)
    target_program = ProgramNode(sequence_segment=ConstantSequenceSegment(target_sequence))
    vhh_program = ProgramNode(children=vhh_nodes())
    sep_program = ProgramNode(sequence_segment=ConstantSequenceSegment(":"))
    return ProgramNode(
        # TODO(alex) @ Maks suggests not minimizing hydrophobic score
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
            MaximizeIPTM(),
        ],
        energy_function_weights=[1.0, 1.0, 0.0, 10.0],
        # TODO(chenghao) right now it has to be target:sequence to ensure correct hotspot calculation
        children=[target_program, sep_program, vhh_program],
        children_are_different_chains=False,
    )


def minibinder_program(pdb_file) -> ProgramNode:
    target_atoms: AtomArray = pdb_file_to_atomarray(pdb_file)
    target_sequence = sequence_from_atomarray(target_atoms)

    # TODO(alex) @ Maks says these should be 50-60 AAs
    sequence_length = 50
    sequence = FixedLengthSequenceSegment(sequence_length)
    target = ConstantSequenceSegment(target_sequence)

    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
            MaximizeIPTM(),
        ],
        energy_function_weights=[1.0, 1.0, 1.0, 1.0],
        children=[
            # TODO(chenghao) right now it has to be target:sequence to ensure correct hotspot calculation
            ProgramNode(sequence_segment=target),
            ProgramNode(ConstantSequenceSegment(":")),
            ProgramNode(sequence_segment=sequence),
        ],
        # TODO(alex) children_are_different_chains is broken
        children_are_different_chains=False,
    )


@ray.remote(num_gpus=1)
def remote_run_simulated_annealing(
    program,
    program_name,
    start_time,
    initial_temperature=1.0,
    annealing_rate=0.97,
    total_num_steps=10_000,
    folding_callback=None,
    display_progress=False,
    progress_verbose_print=False,
):
    program = program()
    sequence, residue_indices = program.get_sequence_and_set_residue_index_ranges()
    print(f"Starting chain from: {sequence}")
    folding_callback = EsmFoldv1()
    folding_callback.load(device="cuda:0")
    logging_callback = S3Logger(program_name, start_time)
    optimized_program = run_simulated_annealing(
        program=program,
        initial_temperature=initial_temperature,
        annealing_rate=annealing_rate,
        total_num_steps=total_num_steps,
        folding_callback=folding_callback,
        logging_callback=logging_callback,
        display_progress=display_progress,
        progress_verbose_print=progress_verbose_print,
    )
    return optimized_program


if __name__ == "__main__":
    pdb_file = "/mnt/shared_storage/target_pdbs/CD3_epsilon_1xiw.pdb"
    # program = vhh_binder_program(pdb_file)
    name = "CD3E"
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Set up the program.

    T_max = 1.0
    T_min = 0.0001
    total_num_steps = 1000
    annealing_rate = (T_min / T_max) ** (1 / total_num_steps)

    handle = [
        remote_run_simulated_annealing.remote(
            program=lambda: vhh_binder_program(pdb_file),
            program_name=name,
            start_time=start_time,
            initial_temperature=T_max,
            annealing_rate=annealing_rate,
            total_num_steps=total_num_steps,
            display_progress=True,
            progress_verbose_print=True,
        )
        for i in range(5)
    ]

    optimized_programs = ray.get(handle)
    for optimized_program in optimized_programs:
        print(
            "Final sequence = {}".format(
                optimized_program.get_sequence_and_set_residue_index_ranges()[0]
            )
        )
