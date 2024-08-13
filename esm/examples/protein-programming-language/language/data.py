from typing import List
from dataclasses import dataclass

from biotite.structure import AtomArray

from language.program import ProgramNode

@dataclass
class MetropolisHastingsState:
    program: ProgramNode
    temperature: float
    annealing_rate: float
    num_steps: int
    energy: float
    best_energy: float
    energy_term_fn_values: list
    last_accepted_step: int
    num_accepted: int


@dataclass
class LogState:
    chain_uuid: str
    sequence: str
    program: ProgramNode
    prev_state: MetropolisHastingsState
    atoms: AtomArray
    ptm: float
    plddt: float
    energy: float
    energy_term_fn_values: list


@dataclass
class GeneratorState:
    pass


@dataclass
class FilePathPointers:
    pdb_path: str
    tensors_path: str
    json_path: str


@dataclass
class EnergyTerm:
    name: str
    weight: float
    value: float


@dataclass
class EnergyState:
    energy_terms: List[EnergyTerm]
    energy: float
    combination_type: str = "sum"
