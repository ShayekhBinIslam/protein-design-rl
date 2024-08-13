# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import StringIO
from typing import Union, Optional

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    return PDBFile.read(pdb_path).get_structure(model=1)


def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int) -> AtomArray:
    return atoms[np.logical_and(atoms.res_id >= start, atoms.res_id < end)]

def linear_discount(x, cutoffs, eps_min=0.001, reverse=False):
    if reverse:
        x = (cutoffs[1] - x) / (cutoffs[1] - cutoffs[0])
    else:
        x = (x - cutoffs[0]) / (cutoffs[1] - cutoffs[0])
    # min(max(eps_min, x), 1.0)
    if x < eps_min:
        x = eps_min
    elif x > 1.0:
        x = 1.0 # relu to maxout at 1
    return x

def get_iptm(output, seq, distogram_bins, linker_length=25, compute_asym_id=True):
    split_sequence = seq.split(":")
    subsequence_lengths = [len(subsequence) for subsequence in split_sequence]
    num_subsequences = len(subsequence_lengths)
    assert num_subsequences > 1, "Sequence must have more than 1 chains to calculate iptm"
    sequence_length = sum(subsequence_lengths)
    sequence_with_linker_length = sequence_length + linker_length * (num_subsequences - 1)
    ptm_logits = output["ptm_logits"][
        None, :sequence_with_linker_length, :sequence_with_linker_length
    ]
    residue_weights = torch.ones(sequence_with_linker_length, device=ptm_logits.device)
    asym_id = torch.ones(sequence_with_linker_length, device=ptm_logits.device)
    chain_id = 1
    for i in range(num_subsequences):
        linker_idx = sum(subsequence_lengths[0 : i + 1]) + linker_length * i
        residue_weights[linker_idx : linker_idx + linker_length] = 0
        asym_id[linker_idx : linker_idx + linker_length] = chain_id + 1
        asym_id[linker_idx + linker_length :] = chain_id + 2
        chain_id += 1

    if not compute_asym_id:
        asym_id = False
        interface = False
    else:
        interface = True

    return compute_tm(
        logits=ptm_logits,
        max_bins=32,
        no_bins=distogram_bins,
        residue_weights=residue_weights,
        asym_id=asym_id,
        interface=interface,
    )


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    interface: bool = True,
    asym_id: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    # todo: add new mask for interface size and/or chunk of scores
    num_res = int(logits.shape[-2])
    if residue_weights is None:
        residue_weights = logits.new_ones(num_res)

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    pair_mask = torch.ones((num_res, num_res), dtype=torch.bool, device=logits.device)
    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]
    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
    normed_residue_mask = pair_residue_weights / (
        eps + pair_residue_weights.sum(dim=-1, keepdim=True)
    )
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers
