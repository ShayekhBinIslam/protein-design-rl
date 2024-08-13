# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union, Optional

import numpy as np
from biotite.structure import AtomArray

ALL_RESIDUE_TYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

RESIDUE_TYPES_WITHOUT_CYSTEINE = deepcopy(ALL_RESIDUE_TYPES)
RESIDUE_TYPES_WITHOUT_CYSTEINE.remove("C")

RESIDUE_TYPES_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
RESIDUE_TYPES_3to1 = {v: k for k, v in RESIDUE_TYPES_1to3.items()}


class SequenceSegmentFactory(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get(self) -> str:
        pass

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def num_mutation_candidates(self) -> int:
        pass


class ConstantSequenceSegment(SequenceSegmentFactory):
    def __init__(self, sequence: str) -> None:
        super().__init__()
        self.sequence = sequence

    def get(self) -> str:
        return self.sequence

    def mutate(self) -> None:
        pass

    def num_mutation_candidates(self) -> int:
        return 0

    def reset(self) -> None:
        pass


class FixedLengthSequenceSegment(SequenceSegmentFactory):
    def __init__(
        self,
        initial_sequence: Union[str, int],
        disallow_mutations_to_cysteine=True,
    ) -> None:
        super().__init__()
        self.mutation_residue_types = (
            RESIDUE_TYPES_WITHOUT_CYSTEINE if disallow_mutations_to_cysteine else ALL_RESIDUE_TYPES
        )

        self.sequence = (
            initial_sequence
            if type(initial_sequence) == str
            else random_sequence(length=initial_sequence, corpus=self.mutation_residue_types)
        )

    def get(self) -> str:
        return self.sequence

    def mutate(self) -> None:
        self.sequence = substitute_one_amino_acid(self.sequence, self.mutation_residue_types)

    def num_mutation_candidates(self) -> int:
        return len(self.sequence)

    def reset(self) -> None:
        length = len(self.sequence)
        self.sequence = random_sequence(
            length=length, corpus=self.mutation_residue_types
        )
        

def substitute_one_amino_acid(sequence: str, corpus: List[str]) -> str:
    sequence = list(sequence)
    index = np.random.choice(len(sequence))
    sequence[index] = np.random.choice(corpus)
    return "".join(sequence)


def random_sequence(length: int, corpus: List[str]) -> str:
    "Generate a random sequence using amino acids in corpus."

    return "".join([np.random.choice(corpus) for _ in range(length)])


def sequence_from_atomarray(atoms: AtomArray) -> str:
    return "".join([RESIDUE_TYPES_3to1[aa] for aa in atoms[atoms.atom_name == "CA"].res_name])


class VariableLengthSequenceSegment(SequenceSegmentFactory):
    def __init__(
        self,
        initial_sequence: Union[str, int],
        disallow_mutations_to_cysteine=True,
        mutation_operation_probabilities: List[float] = [
            3.0,  # Substitution weight.
            1.0,  # Deletion weight.
            1.0,  # Insertion weight.
        ],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        random_max_length: bool = False,
    ) -> None:
        super().__init__()
        self.mutation_residue_types = (
            RESIDUE_TYPES_WITHOUT_CYSTEINE if disallow_mutations_to_cysteine else ALL_RESIDUE_TYPES
        )

        self.sequence = (
            initial_sequence
            if type(initial_sequence) == str
            else random_sequence(length=initial_sequence, corpus=self.mutation_residue_types)
        )

        self.mutation_operation_probabilities = np.array(mutation_operation_probabilities)
        self.mutation_operation_probabilities /= self.mutation_operation_probabilities.sum()
        self.min_length = min_length
        self.random_max_length = random_max_length
        self.max_length = max_length
        if random_max_length:
            self.max_length = np.random.randint(low=len(self.sequence), high=max_length + 1)

    def get(self) -> str:
        return self.sequence

    def mutate(self) -> None:
        if self.min_length is not None and len(self.sequence) <= self.min_length:
            mutation_operation = np.random.choice(
                [
                    self._mutate_substitution,
                    self._mutate_insertion,
                ],
                p=[0.75, 0.25],
            )

        elif self.max_length is not None and len(self.sequence) <= self.max_length:
            mutation_operation = np.random.choice(
                [
                    self._mutate_substitution,
                    self._mutate_deletion,
                ],
                p=[0.75, 0.25],
            )

        else:
            mutation_operation = np.random.choice(
                [
                    self._mutate_substitution,
                    self._mutate_deletion,
                    self._mutate_insertion,
                ],
                p=self.mutation_operation_probabilities,
            )
        mutation_operation()

    def _mutate_substitution(self) -> str:
        self.sequence = substitute_one_amino_acid(self.sequence, self.mutation_residue_types)

    def _mutate_deletion(self) -> str:
        self.sequence = delete_one_amino_acid(self.sequence)

    def _mutate_insertion(self) -> str:
        self.sequence = insert_one_amino_acid(self.sequence, self.mutation_residue_types)

    def num_mutation_candidates(self) -> int:
        # NOTE(brianhie): This should be `3*len(self.sequence) + 1`,
        # since there are `len(self.sequence)` substitutions and
        # deletions, and `len(self.sequence) + 1` insertions.
        # However, as this is used to weight sequence segments for
        # mutations when combined into a multi-segment program, we
        # just weight by `len(self.sequence)` for now.
        return len(self.sequence)

    def reset(self) -> None:
        length = len(self.sequence)
        self.sequence = random_sequence(
            length=length, corpus=self.mutation_residue_types
        )


def delete_one_amino_acid(sequence: str) -> str:
    index = np.random.choice(len(sequence))
    return sequence[:index] + sequence[index + 1 :]


def insert_one_amino_acid(sequence: str, corpus: List[str]) -> str:
    n = len(sequence)
    index = np.random.randint(0, n) if n > 0 else 0
    insertion = np.random.choice(corpus)
    return sequence[:index] + insertion + sequence[index:]
