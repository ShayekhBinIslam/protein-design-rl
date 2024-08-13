# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ray
import torch
from biotite.database.rcsb import fetch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Optional

import esm
import numpy as np
from biotite.structure import AtomArray
from openfold.np.residue_constants import atom_order
from torch.utils._pytree import tree_map

from language.utilities import pdb_file_to_atomarray, get_iptm
from language.sequence import sequence_from_atomarray

from yaml_resolvers import download_files_resolver
# _LM_DESIGN_DIR_PATH = Path(__file__).parent.parent / "lm-design"
# sys.path.append(_LM_DESIGN_DIR_PATH)


@dataclass
class FoldingResult:
    sequence: str
    atoms: AtomArray
    pdb_string: str
    ptm: float
    plddt: float
    iptm: float
    esm_nll: float
    ngram_kl: float
    binder_plddt: float
    binder_ptm: float
    binder_atoms: AtomArray
    raw_output: Dict[str, torch.Tensor]
    residue_index_offset: int
    linker_length: int
    contacts: torch.Tensor 


class FoldingCallback(ABC):
    "Interface for running ESMFold and other folding methods."

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, device: str) -> None:
        pass

    @abstractmethod
    def fold(self, sequence: str) -> FoldingResult:
        pass


class Dummy(FoldingCallback):
    def __init__(self) -> None:
        super().__init__()
        self.test_pdb = "/mnt/shared_storage/test_pdbs/QVQLQ:EMAYK_ESM_v1.pdb"
        # self.test_array: AtomArray = deepcopy(pdb_file_to_atomarray(self.test_pdb))
        self.test_array: AtomArray = pdb_file_to_atomarray(fetch("6mrs", format="pdb"))

    def load(self, device: str) -> None:
        pass

    def fold(self, sequence: str) -> FoldingResult:
        result = FoldingResult(
            sequence=sequence,
            atoms=self.test_array,
            pdb_string=sequence_from_atomarray(self.test_array),
            ptm=float(np.random.rand()),
            plddt=float(np.random.rand()),
            esm_nll=0.0,
            ngram_kl=0.0,
            iptm=float(np.random.rand()),
            raw_output={
                "positions": torch.zeros(1),
                "aatype": torch.zeros(1),
                "atom37_atom_exists": torch.zeros(1),
                "residue_index": torch.zeros(1),
                "plddt": torch.zeros(1),
                # "ptm_logits",
                "ptm": torch.zeros(1),
                # "aligned_confidence_probs",
                "predicted_aligned_error": torch.zeros(1),
                "max_predicted_aligned_error": torch.zeros(1),
                "mean_plddt": torch.zeros(1),
                "chain_index": torch.zeros(1),
            },
        )
        return result


@ray.remote(num_gpus=1, max_restarts=-1, max_task_retries=-1)
class EsmFoldv1(FoldingCallback):
    "Runs ESMFold v1.0."

    def __init__(self, use_llm: bool = False, predict_contacts:bool=False, model_checkpoint: Optional[str] = None) -> None:
        """
        Args:
            use_llm: Whether to use the LLM energy calculator
            model_checkpoint: Path to the full model checkpoint (including pickled class, not only state dict).
                # Using this load method can reduce required memory by loading model directly to desired device.
        """
        super().__init__()

        self.model = None
        self.use_llm = use_llm
        self.predict_contacts = predict_contacts

        self.load(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_checkpoint=model_checkpoint,
        )

    def load(self, device: torch.device, model_checkpoint: Optional[str] = None) -> None:
        if model_checkpoint is None:
            self.model = esm.pretrained.esmfold_v1().eval()
        else:
            model_checkpoint = download_files_resolver(model_checkpoint)
            self.model = torch.load(model_checkpoint, map_location=device)

        self.model = self.model.to(device)
        if self.use_llm:
            from language.llm_energy import ESMLLMEnergyCalculator

            self.esm_llm_energy_calculator = ESMLLMEnergyCalculator.remote()
        if self.predict_contacts:
            from language.contacts import LLMContactPredictor
            self.contact_predictor = LLMContactPredictor.remote()

    def fold(
        self,
        sequence: str,
        # residue_indices: List[int],
        calculate_iptm: bool = True,
        calculate_esm_nll: bool = False,
        calculate_ngram_kl: bool = False,
        linker_length: int = 25,
        num_recycles: int = 4,
        residue_index_offset: int = 512,
        fold_binder_sep: bool = False,
        predict_contacts: bool = False,
    ) -> FoldingResult:
        assert self.model is not None, "Must call load() before fold()."

        # TODO: Current `esm.esmfold.v1.misc.output_to_pdb()` adds 1 to the `residx`
        # mistakenly, just subtract 1 for now but fix in a later version.
        # residue_indices = np.array(residue_indices) - 1

        esm_nll = -1.0
        if calculate_esm_nll:
            esm_nll = ray.get(self.esm_llm_energy_calculator.get_nll_energy_term.remote(sequence))

        ngram_kl = -1.0
        if calculate_ngram_kl:
            ngram_kl = ray.get(
                self.esm_llm_energy_calculator.get_ngram_energy_term.remote(sequence)
            )
        # todo (maks) make alternative here
        if predict_contacts:
            contacts = ray.get(self.contact_predictor.get_contacts.remote(sequence, 100))
        else:
            contacts = None

        binder_plddt = 0.0
        binder_ptm = 0.0
        binder_atoms = AtomArray(length=0)
        if fold_binder_sep:
            binder_raw_output = self.model.infer(
                sequence.split(":")[-1], num_recycles=num_recycles
            )
            binder_ptm = float(binder_raw_output["ptm"])
            binder_plddt = float(binder_raw_output["mean_plddt"]) / 100.0
            binder_pdb_string = esm.esmfold.v1.misc.output_to_pdb(binder_raw_output)[0]
            binder_atoms: AtomArray = pdb_file_to_atomarray(StringIO(binder_pdb_string)).copy()

        raw_output = self.model.infer(
            sequence,  # residx=torch.Tensor(residue_indices).long().reshape(1, -1),
            residue_index_offset=residue_index_offset,
            chain_linker="G" * linker_length,
            num_recycles=num_recycles,
        )
        raw_output = tree_map(lambda x: x.to("cpu"), raw_output)

        pdb_string = esm.esmfold.v1.misc.output_to_pdb(raw_output)[0]
        atoms: AtomArray = pdb_file_to_atomarray(StringIO(pdb_string)).copy()

        if calculate_iptm:
            iptm = float(get_iptm(raw_output, sequence, self.model.distogram_bins, linker_length))
            plddt = float(raw_output["mean_plddt"]) / 100.0
            ptm = float(
                get_iptm(
                    raw_output,
                    sequence,
                    self.model.distogram_bins,
                    linker_length,
                    compute_asym_id=False,
                )
            )
        else:
            ptm = float(raw_output["ptm"])
            plddt = raw_output["plddt"]
            plddt = plddt[0, ...].numpy()
            plddt = plddt.transpose()
            plddt = plddt[atom_order["CA"], :]
            plddt = float(plddt.mean()) / 100.0
            iptm = 0.0

        return FoldingResult(
            sequence=sequence,
            atoms=atoms,
            ptm=ptm,
            plddt=plddt,
            iptm=iptm,
            esm_nll=esm_nll,
            ngram_kl=ngram_kl,
            pdb_string=pdb_string,
            raw_output=raw_output,
            residue_index_offset=residue_index_offset,
            linker_length=linker_length,
            binder_plddt=binder_plddt,
            binder_ptm=binder_ptm,
            binder_atoms=binder_atoms,
            contacts=contacts
        )
