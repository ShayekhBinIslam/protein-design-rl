# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable
import copy
import torch
import numpy as np
import re
from scipy.spatial import distance_matrix
from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose

from language.folding_callbacks import FoldingResult
from language.utilities import get_atomarray_in_residue_range, linear_discount


class EnergyTerm(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self, node, folding_result: FoldingResult) -> float:
        pass


class MaximizePTM(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - folding_result.ptm


class MaximizePLDDT(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - folding_result.plddt


class MaximizeIPTM(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - folding_result.iptm


class MaximizeBinderPLDDT(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - float(folding_result.binder_plddt)


class MaximizeBinderPTM(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return 1.0 - float(folding_result.binder_ptm)


class MinimizeBinderRMSD(EnergyTerm):
    def __init__(
        self, rmsd_cutoffs: List[float] = [2.5, 15.0], backbone_only: bool = False
    ) -> None:
        super().__init__()
        self.backbone_only: bool = backbone_only
        self.rmsd_cutoffs = rmsd_cutoffs

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        _, binder_atoms_in_complex = get_target_binder_atoms(folding_result, self.backbone_only)
        binder_atoms_separate = folding_result.binder_atoms.coord
        rmsd = crmsd(binder_atoms_separate, binder_atoms_in_complex)
        return linear_discount(rmsd, self.rmsd_cutoffs)


class MinimizeBinderSurfaceHydrophobics(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return hydrophobic_score(folding_result.binder_atoms)


class MaximizeHotspotProbability(EnergyTerm):
    def __init__(
        self,
        hotspots: List,
        distance_threshold: float = 8.0,
        binder_first=True,
    ) -> None:
        super().__init__()
        self.hotspots: List = hotspots  # one hotspot actually
        self.distance_threshold: float = distance_threshold
        if not binder_first:
            raise NotImplementedError("implement target first")
    def _in_contact(self, folding_result):
        hotspots_atoms, binder_atoms = get_target_binder_atoms(folding_result,
            backbone_only=True,
            hotspots=self.hotspots,
            binder_first=True,
        )
        dist_matrix = distance_matrix(hotspots_atoms.coord, binder_atoms.coord)
        if (dist_matrix < self.distance_threshold).any():
            return True
        else:
            return False
    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        if self._in_contact(folding_result):
            score = 0.
        else:
            binder_seq, target_seq = folding_result.sequence.split(":")
            binder_len,target_len = len(binder_seq), len(target_seq)
            # maximize probability of the the contact
            mask = torch.zeros([binder_len+target_len,binder_len+target_len],dtype=torch.bool)
            mask[0:binder_len,binder_len + self.hotspots[0]] = True
            max_contact = folding_result.contacts[mask].max().item()
            score = 1. - max_contact
        print("hotspot",self.hotspots,"max contact", max_contact)
        return score




def contact_calculation(target_coords, binder_coords, distance_threshold) -> np.ndarray:
    dist_matrix = distance_matrix(target_coords, binder_coords)
    return np.sum(dist_matrix <= distance_threshold)


def get_target_binder_atoms(
    folding_result: FoldingResult,
    backbone_only: bool = False,
    hotspots: List = None,
    binder_first: bool = True,
) -> Tuple[AtomArray, AtomArray]:
    first_seq, second_seq = folding_result.sequence.split(":")
    first_seq_atoms_start_index = 1
    second_seq_atoms_start_index = (
        len(first_seq) + 1 + folding_result.residue_index_offset + folding_result.linker_length
    )

    if binder_first:
        binder_seq, target_seq = first_seq, second_seq
        binder_atoms = get_atomarray_in_residue_range(
            folding_result.atoms, first_seq_atoms_start_index, len(binder_seq) + 1
        )
        target_start_index = second_seq_atoms_start_index
    else:
        target_seq, binder_seq = first_seq, second_seq
        binder_atoms = get_atomarray_in_residue_range(
            folding_result.atoms,
            second_seq_atoms_start_index,
            second_seq_atoms_start_index + len(binder_seq) + 1,
        )
        target_start_index = 0

    if hotspots is None:
        target_atoms = get_atomarray_in_residue_range(
            folding_result.atoms, target_start_index + 1, target_start_index + len(target_seq) + 1
        )
    else:
        hotspots = [h + target_start_index + 1 for h in hotspots]
        assert max(hotspots) <= target_start_index + len(
            target_seq
        ), "the hotspot locations are beyond the length of the target"
        target_atoms = folding_result.atoms[np.isin(folding_result.atoms.res_id, hotspots)]

    if backbone_only:
        target_atoms = get_backbone_atoms(target_atoms)
        binder_atoms = get_backbone_atoms(binder_atoms)
    return target_atoms, binder_atoms


class MaximizeInterfaceSize(EnergyTerm):
    def __init__(
        self,
        distance_threshold: float = 5.0,
        interface_score_cutoffs: List[float] = [2.0, 60.0],
        backbone_only: bool = True,
        binder_first: bool = True,
    ) -> None:
        super().__init__()

        self.backbone_only: bool = backbone_only
        self.distance_threshold: float = distance_threshold
        self.interface_score_cutoffs: List[float] = interface_score_cutoffs
        self.binder_first: bool = binder_first

    def compute(self, node, folding_result: FoldingResult) -> float:
        # children_are_different_chains are broken, so we're using folding_results
        del node
        target_atoms, binder_atoms = get_target_binder_atoms(
            folding_result, self.backbone_only, binder_first=self.binder_first
        )

        num_contacts = contact_calculation(
            target_atoms.coord, binder_atoms.coord, distance_threshold=self.distance_threshold
        )

        return 1.0 - linear_discount(num_contacts, self.interface_score_cutoffs)


class MinimizeInterfaceHotspotDistance(EnergyTerm):
    def __init__(
        self,
        hotspots: List,
        closest_k: int = 5,
        distance_threshold: float = 5.0,
        backbone_only: bool = True,
        eps_min: float = 0.001,
        binder_first: bool = True,
        scoring_exp: Callable[[float], float] = np.exp,
        relu_on_score: bool = False,
    ) -> None:
        # scoring_exp could be:
        # increasing:
        # np.exp, np.exp2, (lambda x: np.sqrt(np.exp2(x))),
        # decreasing (turn on relu_on_score):
        # lambda x: 1 - np.exp(-x); lambda x: 1 - np.exp2(-x); lambda x: 1 - np.sqrt(np.exp2(x))
        super().__init__()

        self.hotspots: List = hotspots  # note all hotspots start from 0
        self.closest_k: int = closest_k
        self.backbone_only: bool = backbone_only
        self.distance_threshold: float = distance_threshold
        self.eps_min: float = eps_min
        self.binder_first: bool = binder_first
        self.scoring_exp: Callable[[float], float] = scoring_exp
        self.relu_on_score: bool = relu_on_score

    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        hotspots_atoms, binder_atoms = get_target_binder_atoms(
            folding_result,
            self.backbone_only,
            hotspots=self.hotspots,
            binder_first=self.binder_first,
        )

        dist_matrix = distance_matrix(hotspots_atoms.coord, binder_atoms.coord)
        # globally instead of locally
        # sorted_distances = np.sort(dist_matrix, axis=1)
        # avg_k_distances = np.mean(sorted_distances[:, :k], axis=1)
        sorted_distances = np.sort(dist_matrix, axis=1)
        avg_k_distances = np.mean(np.sort(sorted_distances[:, 0])[: self.closest_k])
        score = float(self.scoring_exp(avg_k_distances - self.distance_threshold))
        if self.relu_on_score:
            if score <= self.eps_min:
                score = self.eps_min
            elif score > 1.0:
                score = 1.0  # relu to maxout at 1
        return score


class FilterLiabilities(EnergyTerm):
    def __init__(
        self,
        cysteine: bool = False,
        glycosylation: bool = True,
        deamidation: bool = False,
        isomerization: bool = False,
        aromatic: bool = False,
        aggregation: bool = False,
        polyspecificity: bool = False,
        protease_sensitive: bool = False,
        integrin_binding: bool = False,
        lysine_glycation: bool = False,
        metal_catalyzed_fragmentation: bool = False,
        polyspecificity_agg: bool = False,
        streptavidin_binding: bool = False,
    ):
        super().__init__()
        self.filters = []
        # todo only detect unpaired cysteines
        if cysteine:
            self.filters.append("C")
        if glycosylation:
            self.filters.append("N[^P][STC]")
        if deamidation:
            self.filters.append("N[GSTNAHD]|GN[FYTG]")
        if isomerization:
            self.filters.append("D[THGSD]")
        if aromatic:
            self.filters.append("HYF|HWH")  # '[FYW]{3}'
        if polyspecificity:
            self.filters.append("GG|GGG|RR|VG|VV|VVV|WW|YY|W[A-Z]W")
        if protease_sensitive:
            self.filters.append("D[PGSVYFQKLD]")
        if aggregation:
            self.filters.append("FHW")
        if integrin_binding:
            self.filters.append("RGD|RYD|LDV|KGD")
        if lysine_glycation:
            self.filters.append("KE|EK|ED")
        if metal_catalyzed_fragmentation:
            self.filters.append("HS|SH|KT|H[A-Z]S|S[A-Z]H")
        if polyspecificity_agg:
            self.filters.append("[FILVWY]{3}")
        if streptavidin_binding:
            self.filters.append("HPQ|EPDW|PW[A-Z]WL|GDWVFI|PWPWLG")

    def compute(self, node, folding_result: FoldingResult):
        sequence = folding_result.sequence
        total = 0
        for filter_pattern in self.filters:
            total += len(re.findall(filter_pattern, sequence))
        return total

    def filter(self, sequences):
        filtered = sequences
        for filter_pattern in self.filters:
            filtered = [seq for seq in filtered if not re.search(filter_pattern, seq)]
        return filtered


class MinimizeESMNLL(EnergyTerm):
    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return folding_result.esm_nll


class MinimizeNGramKL(EnergyTerm):
    def compute(self, node, folding_result: FoldingResult) -> float:
        del node
        return folding_result.ngram_kl


class SymmetryRing(EnergyTerm):
    def __init__(self, all_to_all_protomer_symmetry: bool = False) -> None:
        super().__init__()
        self.all_to_all_protomer_symmetry: bool = all_to_all_protomer_symmetry

    def compute(self, node, folding_result: FoldingResult) -> float:
        protomer_nodes = node.get_children()
        protomer_residue_ranges = [
            protomer_node.get_residue_index_range() for protomer_node in protomer_nodes
        ]

        centers_of_mass = []
        for start, end in protomer_residue_ranges:
            backbone_coordinates = get_backbone_atoms(
                folding_result.atoms[
                    np.logical_and(
                        folding_result.atoms.res_id >= start,
                        folding_result.atoms.res_id < end,
                    )
                ]
            ).coord
            centers_of_mass.append(get_center_of_mass(backbone_coordinates))
        centers_of_mass = np.vstack(centers_of_mass)

        return (
            float(np.std(pairwise_distances(centers_of_mass)))
            if self.all_to_all_protomer_symmetry
            else float(np.std(adjacent_distances(centers_of_mass)))
        )


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[(atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")]


def _is_Nx3(array: np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3


def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates - np.roll(coordinates, shift=1, axis=0)
    return np.linalg.norm(m, axis=-1)


class MinimizeSurfaceHydrophobics(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return hydrophobic_score(folding_result.atoms, start, end)


_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}


def hydrophobic_score(
    atom_array: AtomArray,
    start_residue_index: Optional[int] = None,
    end_residue_index: Optional[int] = None,
) -> float:
    """
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface
    exposed. Typically, lower is better.
    """

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    # TODO(scandido): Resolve the float/bool thing going on here.
    atom_array = copy.deepcopy(atom_array)
    hydrophobic_surf = np.logical_and(selection_mask * hydrophobic_mask, sasa(atom_array))
    # TODO(brianhie): Figure out how to handle divide-by-zero.
    return sum(hydrophobic_surf) / sum(selection_mask * hydrophobic_mask)


class MinimizeSurfaceExposure(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return surface_ratio(folding_result.atoms, list(range(start, end)))


class MaximizeSurfaceExposure(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return 1.0 - surface_ratio(folding_result.atoms, list(range(start, end)))


def surface_ratio(atom_array: AtomArray, residue_indices: List[int]) -> float:
    """Computes ratio of atoms in specified ratios which are on the protein surface."""

    residue_mask = np.array([res_id in residue_indices for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))
    return sum(surface) / sum(residue_mask)


class MinimizeSurfaceExposure(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return surface_ratio(folding_result.atoms, list(range(start, end)))


class MaximizeSurfaceExposure(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        return 1.0 - surface_ratio(folding_result.atoms, list(range(start, end)))


def surface_ratio(atom_array: AtomArray, residue_indices: List[int]) -> float:
    """Computes ratio of atoms in specified ratios which are on the protein surface."""

    residue_mask = np.array([res_id in residue_indices for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))
    return sum(surface) / sum(residue_mask)


class MaximizeGlobularity(EnergyTerm):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        backbone = get_backbone_atoms(
            folding_result.atoms[
                np.logical_and(
                    folding_result.atoms.res_id >= start,
                    folding_result.atoms.res_id < end,
                )
            ]
        ).coord

        return float(np.std(distances_to_centroid(backbone)))


def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Computes the distances from each of the coordinates to the
    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    center_of_mass = get_center_of_mass(coordinates)
    m = coordinates - center_of_mass
    return np.linalg.norm(m, axis=-1)


class MinimizeCRmsd(EnergyTerm):
    def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
        super().__init__()

        self.template: AtomArray = template
        self.backbone_only: bool = backbone_only
        if self.backbone_only:
            self.template = get_backbone_atoms(template)

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

        if self.backbone_only:
            atoms = get_backbone_atoms(atoms)

        return crmsd(self.template, atoms)


def crmsd(atom_array_a: AtomArray, atom_array_b: AtomArray) -> float:
    # TODO(scandido): Add this back.
    # atom_array_a = canonicalize_within_residue_atom_order(atom_array_a)
    # atom_array_b = canonicalize_within_residue_atom_order(atom_array_b)
    superimposed_atom_array_b_onto_a, _ = superimpose(atom_array_a, atom_array_b)
    return float(rmsd(atom_array_a, superimposed_atom_array_b_onto_a).mean())


class MinimizeDRmsd(EnergyTerm):
    def __init__(self, template: AtomArray, backbone_only: bool = False) -> None:
        super().__init__()

        self.template: AtomArray = template
        self.backbone_only: bool = backbone_only
        if self.backbone_only:
            self.template = get_backbone_atoms(template)

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        atoms = get_atomarray_in_residue_range(folding_result.atoms, start, end)

        if self.backbone_only:
            atoms = get_backbone_atoms(atoms)

        return drmsd(self.template, atoms)


def drmsd(atom_array_a: AtomArray, atom_array_b: AtomArray) -> float:
    # TODO(scandido): Add this back.
    # atom_array_a = canonicalize_within_residue_atom_order(atom_array_a)
    # atom_array_b = canonicalize_within_residue_atom_order(atom_array_b)

    dp = pairwise_distances(atom_array_a.coord)
    dq = pairwise_distances(atom_array_b.coord)

    return float(np.sqrt(((dp - dq) ** 2).mean()))


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


class MatchSecondaryStructure(EnergyTerm):
    def __init__(self, secondary_structure_element: str) -> None:
        super().__init__()
        self.secondary_structure_element = secondary_structure_element

    def compute(self, node, folding_result: FoldingResult) -> float:
        start, end = node.get_residue_index_range()

        subprotein = folding_result.atoms[
            np.logical_and(
                folding_result.atoms.res_id >= start,
                folding_result.atoms.res_id < end,
            )
        ]
        sse = annotate_sse(subprotein)

        return np.mean(sse != self.secondary_structure_element)
