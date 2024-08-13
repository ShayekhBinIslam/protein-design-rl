'''
Do batch setting
Use this batch to update agent (replay buffer)

Adapt flex see below:
/home/ray/default/FLEXS/examples/Tutorial.ipynb
/home/ray/default/FLEXS/flexs/landscapes/bert_gfp.py
'''

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from copy import deepcopy
from typing import List

import numpy as np
from rich.live import Live
from rich.table import Table

from language.data import MetropolisHastingsState
from language.folding_callbacks import FoldingCallback
from language.logging_callbacks import LoggingCallback
from language.program import ProgramNode
from language import EsmFoldv1

import flexs
from utils.rl.esm_landscape import ESMLandscape
from utils.sampling import PTMMCDropoutProxy, Oracle, PTMProxy
from omegaconf import DictConfig, OmegaConf, open_dict
from Bio import SeqIO
import uuid
import numpy as np
import hydra
import boto3
import tempfile

import hydra
import omegaconf
import os
import pandas as pd
import torch

import os
import src

from pathlib import Path
from tqdm import tqdm
import logging
# Turn off cuda device logging
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
proxy_base = Path(os.path.dirname(src.__file__)).parent
config_path = proxy_base/'configs/proxy_al'

TOP_HOTSPOTS = {
    "CD3e": [81, 77, 78, 0, 80, 76, 79, 73, 74, 75],
    "TrkA": [15, 14, 52, 51, 21, 68, 61, 13, 10, 12],
    "MDM2": [47, 32, 29, 28, 70, 46, 25, 37, 68, 26],
    "PD1": [98, 29, 96, 99, 53, 56, 97, 100, 30, 57],
    "PDL1": [103, 102, 42, 101, 96, 97, 100, 98, 57, 44],
    "SARSCoV2": [143, 169, 152, 50, 168, 153, 161, 43, 45, 6],
}


def read_fasta(fasta_path: str):
    fasta_sequences = SeqIO.parse(open(fasta_path), "fasta")
    fasta_dict = SeqIO.to_dict(fasta_sequences)
    return {k: str(v.seq) for k, v in fasta_dict.items()}

@hydra.main(version_base=None, config_path=str(config_path), config_name="dynappo")
def main(cfg: DictConfig) -> None:
    available_sequences: dict = read_fasta(cfg.fasta_file)
    try:
        target_sequence = available_sequences[cfg.target_name]
    except KeyError:
        print(
            f"KeyError: Unknown Sequence: {cfg.target_name}. Available Sequences: {available_sequences.keys()}"
        )
    
    full_target_hotspots = TOP_HOTSPOTS[cfg.target_name]
    # Uniformly randomly choose a hotspot
    target_hotspots = [
        int(a) for a in np.random.choice(np.array(full_target_hotspots), size=cfg.chain.num_chains)
    ]
    print(f"Target Hotspots: {target_hotspots}")
    with open_dict(cfg):
        cfg.program.target_hotspots = target_hotspots
    cfg_dict = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    print(cfg_dict)

    # proxy = PTMMCDropoutProxy()
    proxy = PTMProxy()

    # oracle = Oracle(cfg.esm_batch_size)
    # oracle.evaluate(target_sequence)

    program = hydra.utils.instantiate(cfg.program)(
        target_sequence, target_hotspots=[target_hotspots[0]]
    )

    esm_landscape = ESMLandscape(proxy)

    dynappo_explorer = flexs.baselines.explorers.DynaPPOESM(  
        # DynaPPO has its own default ensemble model, so don't use CNN
        landscape=esm_landscape,
        env_batch_size=1, #cfg.env_batch_size,
        num_model_rounds=cfg.num_model_rounds,
        rounds=cfg.rounds,
        starting_program=program,
        sequences_batch_size=cfg.sequences_batch_size,
        model_queries_per_batch=cfg.model_queries_per_batch,
        model = None,
        use_dummy_model = False,
        # alphabet= 'ILVAGMFYWEDQNHCRKSTP_', 
        alphabet=flexs.utils.sequence_utils.AAS,
        log_file=f"/efs/users/riashat_islam_341e494/proxy_seqs/dynappoesm_rounds100_flexslog.csv",
    )
    dynappo_sequences, metadata = dynappo_explorer.run(esm_landscape)
    print("Saving dynappo generated sequences.")
    dynappo_sequences.to_csv(f"/efs/users/riashat_islam_341e494/proxy_seqs/dynappoesm_rounds100.csv")
    # del proxy; import gc; gc.collect(); torch.cuda.empty_cache()
    # ptms, plddts = oracle.evaluate(dynappo_sequences['sequence'])
    # dynappo_sequences['ptm'] = ptms
    # dynappo_sequences['plddt'] = plddts
    # print(dynappo_sequences)
    # dynappo_sequences.to_csv(f"/efs/users/riashat_islam_341e494/proxy_seqs/dynappo_trial_run_scores.csv")

if __name__ == "__main__":
    # ray.init(local_mode=True)
    main()