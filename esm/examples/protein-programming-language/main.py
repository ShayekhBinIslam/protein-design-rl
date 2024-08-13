from Bio import SeqIO
import uuid
import numpy as np
import hydra
import boto3
import tempfile
from language.logging_callbacks import S3Logger
from language import run_simulated_annealing
from omegaconf import DictConfig, OmegaConf, open_dict

from language import EsmFoldv1
import ray
from ray.air.integrations.wandb import setup_wandb

import sys
from hotspots import TOP_HOTSPOTS


class Logger:
    """Log stdout to console and file."""

    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, "w")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


# Each task requires one GB of system RAM
@ray.remote(num_cpus=1, memory=1 * 1000 * 1000 * 1000, resources={"is_cpu_only": 1})
def remote_run_simulated_annealing(
    cfg: DictConfig,
    chain_num: int,
    log_dir: str,
    program,
    program_name,
    initial_temperature=1.0,
    annealing_rate=0.97,
    total_num_steps=10_000,
    folding_callback=None,
    display_progress=False,
    progress_verbose_print=False,
):
    chain_uuid = uuid.uuid4()
    wandb_config = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=True)
    setup_wandb(
        config=wandb_config,
        api_key_file=cfg.api_key_file,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        rank_zero_only=False,
        #rank_zero_only=True,
        trial_id=str(chain_uuid),
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
    )
    sequence, residue_indices = program.get_sequence_and_set_residue_index_ranges()
    print(f"Starting chain {chain_num} from: {sequence}")
    logging_callback = S3Logger(log_dir, chain_num, chain_uuid)
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


def read_fasta(fasta_path: str):
    fasta_sequences = SeqIO.parse(open(fasta_path), "fasta")
    fasta_dict = SeqIO.to_dict(fasta_sequences)
    return {k: str(v.seq) for k, v in fasta_dict.items()}


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    #OmegaConf.register_new_resolver("download", download_files_resolver)
    s3_client = boto3.client("s3")
    log_path = cfg.paths.output_dir + "/stdout.log"
    sys.stdout = Logger(log_path)

    # Read Fasta
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

    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=cfg, f=fp.name)
        path = str(cfg.paths.output_dir)
        print('PATH =', path)
        s3_path = path[path.find("mcmc") :]
        print('S3 PATH =', s3_path + "/config.yaml")
        s3_client.upload_file(fp.name, "esmfold-outputs", s3_path + "/config.yaml")

    # ray.init(runtime_env={'pip': ['nltk']})

    annealing_rate = (cfg.chain.T_min / cfg.chain.T_max) ** (1 / cfg.chain.total_num_steps)
    handle = [
        remote_run_simulated_annealing.remote(
            cfg=cfg,
            chain_num=i,
            log_dir=cfg.paths.output_dir,
            program=hydra.utils.instantiate(cfg.program)(
                target_sequence, target_hotspots=[target_hotspots[i]]
            ),
            program_name=cfg.name,
            initial_temperature=cfg.chain.T_max,
            annealing_rate=annealing_rate,
            total_num_steps=cfg.chain.total_num_steps,
            folding_callback=EsmFoldv1.remote(model_checkpoint=cfg.esm_fold.model_checkpoint),
            display_progress=cfg.chain.display_progress,
            progress_verbose_print=cfg.chain.progress_verbose_print,
        )
        for i in range(cfg.chain.num_chains)
    ]

    optimized_programs = ray.get(handle)
    for i, (optimized_program, optimized_state) in enumerate(optimized_programs):
        sequence = optimized_program.get_sequence_and_set_residue_index_ranges()[0]
        print(f"Chain {i} final sequence = {sequence}")

    s3_path = log_path[log_path.find("mcmc") :]
    s3_client.upload_file(log_path, "esmfold-outputs", s3_path)


if __name__ == "__main__":
    main()
