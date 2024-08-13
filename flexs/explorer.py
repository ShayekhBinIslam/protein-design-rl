"""Defines abstract base explorer class."""
import abc
import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import flexs
import wandb
import helper
import scipy

def get_mean_max(per_round_scores, max_so_far):
    per_round_scores = np.array(per_round_scores)
    mean_score = per_round_scores.mean()
    max_score = per_round_scores.max()
    new_max_so_far = False

    if max_so_far is None:
        max_so_far = max_score
    else:
        if max_score > max_so_far:
            max_so_far = max(max_so_far, max_score)
            new_max_so_far = True
    return mean_score, max_score, max_so_far, new_max_so_far


class Explorer(abc.ABC):
    """
    Abstract base explorer class.

    Run explorer through the `run` method. Implement subclasses
    by overriding `propose_sequences` (do not override `run`).
    """

    def __init__(
        self,
        model: flexs.Model,
        name: str,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        log_file: Optional[str] = None,
        state_space: Optional[int] = None
    ):
        """
        Create an Explorer.

        Args:
            model: Model of ground truth that the explorer will use to help guide
                sequence proposal.
            name: A human-readable name for the explorer (may include parameter values).
            rounds: Number of rounds to run for (a round consists of sequence proposal,
                ground truth fitness measurement of proposed sequences, and retraining
                the model).
            sequences_batch_size: Number of sequences to propose for measurement from
                ground truth per round.
            model_queries_per_batch: Number of allowed "in-silico" model evaluations
                per round.
            starting_sequence: Sequence from which to start exploration.
            log_file: .csv filepath to write output.

        """
        self.model = model
        try:
            self.name = name
        except:
            pass

        self.rounds = rounds
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.starting_sequence = starting_sequence
        self.state_space = state_space

        self.proxy_important_sequences = {
            'max_so_far_sequences': [],
            'max_this_round_sequences': []
        }

        self.oracle_important_sequences = {
            'max_so_far_sequences': [],
            'max_this_round_sequences': []
        }

        self.log_file = log_file
        if self.log_file is not None:
            dir_path, filename = os.path.split(self.log_file)
            os.makedirs(dir_path, exist_ok=True)

        if model_queries_per_batch < sequences_batch_size:
            warnings.warn(
                "`model_queries_per_batch` should be >= `sequences_batch_size`"
            )

    @abc.abstractmethod
    def propose_sequences(self, measured_sequences_data: pd.DataFrame, round_idx: Optional[int]):
        """
        Propose a list of sequences to be measured in the next round.

        This method will be overriden to contain the explorer logic for each explorer.

        Args:
            measured_sequences_data: A pandas dataframe of all sequences that have been
            measured by the ground truth so far. Has columns "sequence",
            "true_score", "model_score", and "round".

        Returns:
            A tuple containing the proposed sequences and their scores
                (according to the model).

        """
        pass

    def _log_cool(
        self,
        sequences_data: pd.DataFrame,
        metadata: Dict,
        current_round: int,
        verbose: bool,
        round_start_time: float,
    ) -> None:
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                # First write metadata
                json.dump(metadata, f)
                f.write("\n")

                # Then write pandas dataframe
                sequences_data.to_csv(f, index=False)

        if verbose:
            tqdm.write(
                f"round: {current_round}, top true pTM: {sequences_data['true_fitness'].max()}, top proxy pTM: {sequences_data['proxy_fitness'].max()}, top pLDDT: {sequences_data['pLDDT'].max()}, top reward: {sequences_data['rewards'].max()} "
                f"time: {time.time() - round_start_time:02f}s"
            )

    def run(self, landscape: flexs.Landscape, verbose: bool = True) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self.name,
            "model_name": self.model.name,
            "landscape_name": landscape.name,
            "rounds": self.rounds,
            "sequences_batch_size": self.sequences_batch_size,
            "model_queries_per_batch": self.model_queries_per_batch,
        }

        # Initial sequences and their scores
        world_size = torch.cuda.device_count()
        
        self.model.cost = 0
        proxy_fitness, _ = self.model.get_fitness([self.starting_sequence] * world_size)
        true_fitness, true_plddt = landscape.get_fitness([self.starting_sequence] * world_size)

        sequences_data = pd.DataFrame({
                "sequence": self.starting_sequence,
                "proxy_fitness": proxy_fitness[:1],
                "true_fitness": true_fitness[:1],
                "rewards": proxy_fitness[:1],
                "round": 0,
                "pLDDT": true_plddt[:1],
                "model_cost": self.model.cost,
                "measurement_cost": 1,
                "sequence_length": len(self.starting_sequence),
        })

        self._log_cool(sequences_data, metadata, 0, verbose, time.time())
        dataset = landscape.get_dataset()

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = trange if verbose else range
        max_fitness_so_far = None
        max_true_fitness_so_far = None
        max_rewards_so_far = None
        max_true_plddt_so_far = None
        self.model.cost = 0

        for r in range_iterator(1, self.rounds + 1):
            round_start_time = time.time()
            all_X, all_y = dataset.get_full_dataset()

            if len(all_X) > 0:
                self.model.train(all_X, all_y)
            else:
                tqdm.write("First round has no data to train on in `explorer.run()`")

            proposed_seqs, rewards, proxy_fitness = None, None, None
            start_time = time.time()

            # proposed_seqs is candidate_seqs filtered by proxy fitness scores
            proposed_seqs, rewards, proxy_fitness, edit_distances, misc = self.propose_sequences(sequences_data, round_idx=r)
            tqdm.write(f"self.propose_sequences took {time.time() - start_time}s | {len(proposed_seqs)} unique sequences proposed")
            start_time = time.time()
            true_fitness, true_plddt = landscape.get_fitness(proposed_seqs)
            tqdm.write(f"landscape.get_fitness took {time.time() - start_time}s")
            assert len(proposed_seqs) <= self.sequences_batch_size, "Must propose <= `self.sequences_batch_size` sequences per round"

            dataset.add((proposed_seqs, true_fitness))
            seq_lengths = [len(seq_) for seq_ in proposed_seqs]

            this_batch = pd.DataFrame({
                "sequence": proposed_seqs,
                "proxy_fitness": proxy_fitness,
                "true_fitness": true_fitness,
                "rewards": rewards,
                "round": r,
                "pLDDT": true_plddt,
                "model_cost": self.model.cost,
                "measurement_cost": len(sequences_data) + len(proposed_seqs),
                "sequence_length": seq_lengths,
            })

            correlation_score = scipy.stats.pearsonr(proxy_fitness, true_fitness).statistic

            # Proxy pTM, Oracle pTM, rewards, and true pLDDT
            mean_proxy_fitness, max_proxy_fitness, max_fitness_so_far, new_proxy_max_so_far = get_mean_max(proxy_fitness, max_fitness_so_far)
            mean_true_fitness, max_true_fitness, max_true_fitness_so_far, new_oracle_max_so_far = get_mean_max(true_fitness, max_true_fitness_so_far)
            mean_true_plddt, max_true_plddt, max_true_plddt_so_far, _ = get_mean_max(true_plddt, max_true_plddt_so_far)
            mean_rewards, max_rewards, max_rewards_so_far, _ = get_mean_max(rewards, max_rewards_so_far)

            seq_lens = np.array([len(seq_) for seq_ in proposed_seqs])
            sequences_data = pd.concat([sequences_data, this_batch])
            self._log_cool(sequences_data, metadata, r, verbose, round_start_time)
            all_measured_seqs = set(sequences_data["sequence"].values)
            normalized_diversity_score = helper.mean_hamming_distance(proposed_seqs, normalized=True)

            _, this_batch = helper.diversity_and_fitness_dist(this_batch.copy(), group_by_key='proxy_bin', return_wandb_plots=False)
            _, this_batch = helper.diversity_and_fitness_dist(this_batch.copy(), group_by_key='oracle_bin', return_wandb_plots=False)
            self.get_important_sequences(this_batch, new_proxy_max_so_far, new_oracle_max_so_far)

            wandb_log_dict = {
                f'Proxy PTM/mean': mean_proxy_fitness,
                f'Proxy PTM/max': max_proxy_fitness,
                f'Proxy PTM/max_so_far': max_fitness_so_far,
                f'Proxy PTM/correlation': correlation_score,
                f'Oracle PTM/mean': mean_true_fitness,
                f'Oracle PTM/max': max_true_fitness,
                f'Oracle PTM/max_so_far': max_true_fitness_so_far,
                f'Oracle pLDDT/mean': mean_true_plddt,
                f'Oracle pLDDT/max': max_true_plddt,
                f'Oracle pLDDT/max_so_far': max_true_plddt_so_far,
                f'Sequence specs/edit-distance (mean)': edit_distances.mean(),
                f'Sequence specs/edit-distance (max)': edit_distances.max(),
                f'Sequence specs/edit-distance (min)': edit_distances.min(),
                f'Sequence specs/diversity score (normalized)': normalized_diversity_score,
                f'Sequence specs/diversity score': normalized_diversity_score * seq_lens.mean(),
                f'Exploration/# all proposed_seqs': len(all_measured_seqs),
                'Reward/mean': mean_rewards,
                'Reward/max': max_rewards,
                'Reward/max_so_far': max_rewards_so_far,
                "data/rounds": r,
                "data/model.cost": self.model.cost,
                "data/landscape.cost": landscape.cost,
            }

            if misc is not None:
                if isinstance(misc, float):
                    wandb_log_dict['Exploration/epsilon'] = misc
                elif isinstance(misc, dict):
                    wandb_log_dict.update(misc)
                else:
                    raise TypeError(f"misc must be a float or dict, not {type(misc)}")

            wandb.Table.MAX_ROWS = wandb.Table.MAX_ARTIFACT_ROWS
            wandb.log(wandb_log_dict)

        ckpt_fname = self.get_model_dict(wandb.run.dir)
        if ckpt_fname is not None: 
            wandb.log_model(path=ckpt_fname, name="explorer_model")

        important_seq_columns = this_batch.columns.tolist()
        important_sequences_df = {
            'proxy_max_per_round': pd.DataFrame(columns=important_seq_columns, data=self.proxy_important_sequences['max_this_round_sequences']),
            'proxy_max_so_far': pd.DataFrame(columns=important_seq_columns, data=self.proxy_important_sequences['max_so_far_sequences']),
            'oracle_max_per_round': pd.DataFrame(columns=important_seq_columns, data=self.oracle_important_sequences['max_this_round_sequences']),
            'oracle_max_so_far': pd.DataFrame(columns=important_seq_columns, data=self.oracle_important_sequences['max_so_far_sequences']),
        }

        return sequences_data, important_sequences_df, metadata
    
    def get_model_dict(self, save_dir):
        pass

    def get_important_sequences(self, sequences_df, new_proxy_max_so_far, new_oracle_max_so_far):
        """
            sequences_df: df of all sequences proposed in this round
        """
        best_proxy_score_this_round = sequences_df.iloc[sequences_df['proxy_fitness'].idxmax()].values.tolist()
        self.proxy_important_sequences['max_this_round_sequences'].append(best_proxy_score_this_round)
        
        best_oracle_score_this_round = sequences_df.iloc[sequences_df['true_fitness'].idxmax()].values.tolist()
        self.oracle_important_sequences['max_this_round_sequences'].append(best_oracle_score_this_round)

        if new_proxy_max_so_far:
            self.proxy_important_sequences['max_so_far_sequences'].append(best_proxy_score_this_round)

        if new_oracle_max_so_far:
            self.oracle_important_sequences['max_so_far_sequences'].append(best_oracle_score_this_round)