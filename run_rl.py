import argparse
import sys
sys.path.append('dockq-proxy')
import warnings
warnings.filterwarnings('ignore')
import os
import wandb 
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
from Bio import SeqIO

import torch
import flexs
from flexs.baselines.explorers import DQN_ESM, DiscreteSAC_ESM
from flexs.baselines.explorers.environments.esm_env import ESMEnvironment
from src.proxy_al.utils.rl.esm_landscape import ESMLandscape
from src.proxy_al.utils.sampling import PTMProxy, Oracle

import src
from pathlib import Path
import logging
from helper import get_explorer, set_seeds, TOP_HOTSPOTS

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

# Turn off cuda device logging
# logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
proxy_base = Path(os.path.dirname(src.__file__)).parent
config_path = proxy_base/'configs/proxy_al'
print(config_path)

NUM_ROUNDS = 10
SEQUENCES_BATCH_SIZE = 100
MODEL_QUERIES_PER_BATCH = 20
SCORE = 'PTM'
ALPHABET = 'AAS'

def read_fasta(fasta_path: str):
    fasta_sequences = SeqIO.parse(open(fasta_path), "fasta")
    fasta_dict = SeqIO.to_dict(fasta_sequences)
    return {k: str(v.seq) for k, v in fasta_dict.items()}

@hydra.main(version_base=None, config_path=str(config_path), config_name="dynappo")
def main(cfg: DictConfig) -> None:

    set_seeds(cfg.seed)
    if ALPHABET == 'AAS':
        alphabet = flexs.utils.sequence_utils.AAS
    else:
        raise ValueError

    exp_name = f"{SCORE}_{NUM_ROUNDS}rounds_bs{SEQUENCES_BATCH_SIZE}_queries_per_batch{MODEL_QUERIES_PER_BATCH}"
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

    starting_program = hydra.utils.instantiate(cfg.program)(
        target_sequence, target_hotspots=[target_hotspots[0]]
    )

    if SCORE == 'PTM':
        proxy = PTMProxy()

    oracle_landscape = ESMLandscape(Oracle(batch_size=64))

    env = ESMEnvironment(
        alphabet=alphabet,
        starting_program=starting_program,
        landscape=oracle_landscape,
        model=proxy,
        fitness_model_is_gt=False,
        handle_reset=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="esm-rl",
        config={
            'rounds': NUM_ROUNDS,
            'sequences_batch_size': SEQUENCES_BATCH_SIZE,
            'model_queries_per_batch': MODEL_QUERIES_PER_BATCH,
            'score': SCORE,
            'alphabet': ALPHABET,
            'alphabet_len': len(alphabet),
            'algo': cfg.explorer,
            'max_seq_len': env.seq_max_len,
            'num_actions': env.num_actions,
        }
    )

    exp_design_args = {
        'env': env,
        'model': proxy,
        'rounds': NUM_ROUNDS,
        'sequences_batch_size': SEQUENCES_BATCH_SIZE,
        'model_queries_per_batch': MODEL_QUERIES_PER_BATCH,
        'starting_program': starting_program,
        'alphabet': env.alphabet,
        'device': device
    }

    print(f"Max sequence length: {env.seq_max_len}")
    explorer = get_explorer(cfg.explorer, exp_design_args)
    sequences_data, metadata = explorer.run(oracle_landscape, score=SCORE)

   
if __name__ == "__main__":
    main()
