import dataclasses
import wandb
import gzip
import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple
import time

import boto3
import numpy as np

from language.data import (
    EnergyState,
    EnergyTerm,
    FilePathPointers,
    GeneratorState,
    MetropolisHastingsState,
)
from language.folding_callbacks import FoldingResult
from language.program import ProgramNode


class LoggingCallback(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log(
        self,
        candidate: ProgramNode,
        state: MetropolisHastingsState,
        folding_output: FoldingResult,
        energy_term_fn_values: list,
        energy: float,
    ) -> None:
        pass


class S3Logger(LoggingCallback):
    def __init__(self, log_dir: str, chain_num: int, chain_uuid) -> None:
        super().__init__()
        self.chain_num = chain_num
        self.chain_uuid = chain_uuid
        self.log_dir = log_dir + f"/{chain_num}_{self.chain_uuid}/"
        os.makedirs(self.log_dir)
        self.esm_logger = ESMFoldOutputFileLogger(Path(self.log_dir), self.chain_uuid)
        self.time = time.time()

    def log(
        self,
        candidate: ProgramNode,
        state: MetropolisHastingsState,
        folding_output: FoldingResult,
        energy_term_fn_values: list,
        energy: float,
        accept_candidate: bool,
    ) -> None:
        candidate = deepcopy(candidate)
        sequence, residue_indices = candidate.get_sequence_and_set_residue_index_ranges()
        candidate_binder_seq, candidate_target_seq = sequence.split(":")
        energy_terms = [
            EnergyTerm(name, weight, value) for name, weight, value in energy_term_fn_values
        ]
        energy_state = EnergyState(energy_terms, combination_type="sum", energy=energy)
        prot_id = self.esm_logger.get_protein_id(candidate_binder_seq, candidate_target_seq)
        new_time = time.time()
        print(
            f"Chain: {self.chain_num} Step: {state.num_steps} Energy {energy:.2f} Time: {new_time - self.time:0.2f} ID: {prot_id}"
        )
        self.time = new_time
        fp: FilePathPointers = self.esm_logger.log_to_file(
            step=state.num_steps,
            pdb_string=folding_output.pdb_string,
            folding_outputs=folding_output.raw_output,
            binder_seq=candidate_binder_seq,
            target_seq=candidate_target_seq,
            energy_state=energy_state,
            generator_state=state,
        )

        s3_client = boto3.client("s3")
        for path in [fp.pdb_path, fp.tensors_path, fp.json_path]:
            path = str(path)
            # NOTE(alex) hack to log to nicer path on s3
            s3_path = path[path.find("mcmc") :]
            s3_client.upload_file(path, "esmfold-outputs", s3_path)


class WandbLogger(LoggingCallback):
    def log(
        self,
        candidate: ProgramNode,
        state: MetropolisHastingsState,
        folding_output: FoldingResult,
        energy_term_fn_values: list,
        energy: float,
    ) -> None:
        pass


_FOLDING_OUTPUT_KEYS_TO_LOG = (
    "positions",
    "aatype",
    "atom37_atom_exists",
    "residue_index",
    "plddt",
    # "ptm_logits",
    "ptm",
    # "aligned_confidence_probs",
    "predicted_aligned_error",
    "max_predicted_aligned_error",
    "mean_plddt",
    "chain_index",
)


class BaseFoldingOutputFileLogger(ABC):
    @abstractmethod
    def log_to_file(
        self,
        pdb_string: str,
        folding_outputs: Dict[str, object],
        binder_seq: str,
        target_seq: str,
        energy_state: EnergyState,
        generator_state: GeneratorState = None,
        binder_id: str = None,
        target_id: str = None,
    ) -> FilePathPointers:
        """
        Takes in the above inputs and will write a pdb file, a tensors file
        (containing some keys from the folding_outputs dict and compressed),
        and a json file detailing various settings, sequences, IDs, and so on
        depending on the implementation of the subclass.

        Returns:
          A named tuple containing the path of the pdb, tensors file, and json
          file on the filesystem.
        """
        pass


class ESMFoldOutputFileLogger(BaseFoldingOutputFileLogger):
    def __init__(self, log_dir: Path, run_id: str):
        self.log_dir = log_dir
        self.run_id = run_id
        self.table = None

    def log_to_file(
        self,
        step: int,
        pdb_string: str,
        folding_outputs: Dict[str, object],
        binder_seq: str,
        target_seq: str,
        energy_state: EnergyState,
        generator_state: GeneratorState = None,
    ) -> FilePathPointers:
        os.makedirs(f"{self.log_dir}/{step}")
        protein_id = self.get_protein_id(binder_seq, target_seq)

        pdb_file_path = self._log_pdb_file(step, protein_id, pdb_string)
        tensors_file_path = self._log_tensors_file(step, protein_id, folding_outputs)
        summary_json_file_path = self._log_summary_file(
            step, protein_id, binder_seq, target_seq, energy_state, generator_state
        )
        return FilePathPointers(pdb_file_path, tensors_file_path, summary_json_file_path)

    def _log_pdb_file(self, step: int, protein_id: str, pdb_string: str) -> Path:
        log_path = self.log_dir / str(step) / f"{protein_id}.pdb.gz"
        with gzip.open(log_path, "wt") as f:
            f.write(pdb_string)

        return log_path

    def _log_tensors_file(
        self, step: int, protein_id: str, folding_outputs: Dict[str, object]
    ) -> Path:
        to_log = {}
        for key in _FOLDING_OUTPUT_KEYS_TO_LOG:
            to_log[key] = folding_outputs[key].cpu().numpy()

        log_path = self.log_dir / str(step) / f"{protein_id}_folding_out_arrs.npz"

        np.savez_compressed(log_path, **to_log)
        import gzip
        import shutil

        with open(log_path, "rb") as f_in:
            with gzip.open(str(log_path) + ".gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return str(log_path) + ".gz"

    def _log_summary_file(
        self,
        step: int,
        protein_id: str,
        binder_seq: str,
        target_seq: str,
        energy_state: EnergyState,
        generator_state: GeneratorState = None,
    ) -> Path:
        log_dict = {
            "protein_id": protein_id,
            "run_id": str(self.run_id),
            "sequences": {"binder": binder_seq, "target": target_seq},
            "energy_state": dataclasses.asdict(energy_state),
        }

        if generator_state is not None:
            log_dict["generator_state"] = dataclasses.asdict(generator_state)
            # program is not picklable, delete it and only store the sequences
            del log_dict["generator_state"]["program"]
            (
                accepted_sequence,
                _,
            ) = generator_state.program.get_sequence_and_set_residue_index_ranges()
            accepted_binder_seq, accepted_target_seq = accepted_sequence.split(":")
            log_dict["generator_state"]["accepted_binder_seq"] = accepted_binder_seq
            log_dict["generator_state"]["accepted_target_seq"] = accepted_target_seq

        log_path = self.log_dir / str(step) / f"{protein_id}_summary.json"

        energy_dict = {"total_energy": energy_state.energy}
        energy_dict.update({k.name: k.value for k in energy_state.energy_terms})

        generator_dict = dataclasses.asdict(generator_state)
        # if "energy_term_fn_values" in generator_dict and generator_state.energy_term_fn_values is not None:
        #    generator_dict["energy_term_fn_values"] = {
        #        k.name: k.value for k in generator_state.energy_term_fn_values
        # }

        generator_dict.pop("accepted_binder_seq", None)
        generator_dict.pop("accepted_target_seq", None)
        generator_dict.pop("program", None)
        generator_dict.pop("energy_term_fn_values", None)

        wandb_dict = {"energy": energy_dict, "generator": generator_dict}

        string_dict = {
            "protein_id": protein_id,
            "run_id": str(self.run_id),
            "binder": binder_seq,
            "target": target_seq,
        }
        string_dict.update(energy_dict)

        if self.table is None:
            self.table = wandb.Table(columns=list(string_dict.keys()), data=[string_dict.values()])
        else:
            self.table.add_data(*list(string_dict.values()))

        wandb_dict.update({"Sequence_Table": self.table})
        wandb.log(wandb_dict)

        json_str = json.dumps(log_dict, indent=4)
        with open(log_path, "w") as f:
            f.write(json_str)

        return log_path

    def get_protein_id(self, binder_seq: str, target_seq: str) -> Tuple[str, str]:
        tuple_str = ":".join([str(self.run_id), binder_seq, target_seq])

        hash_object = hashlib.sha1(bytes(tuple_str, "utf-8"))
        return hash_object.hexdigest()
