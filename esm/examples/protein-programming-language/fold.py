import boto3
import ray
from biotite.structure import AtomArray

from language import (
    ConstantSequenceSegment,
    EsmFoldv1,
    FixedLengthSequenceSegment,
    MaximizePLDDT,
    MaximizePTM,
    MinimizeSurfaceHydrophobics,
    ProgramNode,
    pdb_file_to_atomarray,
    run_simulated_annealing,
    sequence_from_atomarray,
)


def minibinder_program() -> ProgramNode:
    target_atoms: AtomArray = pdb_file_to_atomarray(
        "/mnt/shared_storage/target_pdbs/CD3_epsilon_1xiw.pdb"
    )
    target_sequence = sequence_from_atomarray(target_atoms)[:10]

    # TODO(alex) @ Maks says these should be 50-60 AAs
    sequence_length = 50
    sequence = FixedLengthSequenceSegment(sequence_length)
    target = ConstantSequenceSegment(target_sequence)

    return ProgramNode(
        energy_function_terms=[MaximizePTM(), MaximizePLDDT(), MinimizeSurfaceHydrophobics()],
        energy_function_weights=[1.0, 1.0, 1.0],
        children=[ProgramNode(sequence_segment=sequence), ProgramNode(sequence_segment=target)],
        children_are_different_chains=True,
    )


@ray.remote
def fold(
    sequence,
):
    folding_callback = EsmFoldv1.remote()
    folding_output = ray.get(folding_callback.fold.remote(sequence))
    return folding_output


if __name__ == "__main__":
    sequence = "QVQLQESGGGLVQAGGSLRLSCAASVRPHDGMGWFRQAPGKEREFVATIKNQRTAYTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYCMARYPHTFLDVFPWGQGTQVTVSSGS:YKVSISGTTVILTCPQYPGSEILWQHNDKNIGGDEDDKNIGSDEDHLSLKEFSELEQSGYYVCYPRGSKPEDANFYLYLRAR"
    handle = fold.remote(sequence)
    folding_output = ray.get(handle)

    from language import MinimizeInterfaceHotspotDistance
    # energy = MinimizeInterfaceHotspotDistance(hotspots=[5,6,7,8,9,10])
    energy = MinimizeInterfaceHotspotDistance(hotspots=[22, 23, 24, 25], backbone_only=True)
    #energy = MinimizeInterfaceHotspotDistance(hotspots=[76,77,78,79])
    energy = energy.compute(None, folding_output)
    print(energy)
    exit()

    import pickle

    file = open("test_output.pkl", "wb")
    pickle.dump(folding_output, file)
    exit()
    program = minibinder_program()
    sequence, residue_indices = program.get_sequence_and_set_residue_index_ranges()
    print(sequence, residue_indices)
    print(handle)
    import pickle

    file = open("array.pkl", "w")
    pickle.dump(folding_output, file)
    s3_client = boto3.client("s3")
    s3_client.upload_file("array.pkl", "esmfold-outputs", "zzz_test")
