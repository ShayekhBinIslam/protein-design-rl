"""Defines the Random explorer class."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch.distributions import Categorical
import flexs
from tqdm import tqdm
from flexs.utils import sequence_utils as s_utils
import editdistance
from flexs.utils.sequence_utils import (
    batched_construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
)



class Random(flexs.Explorer):
    """A simple random explorer.

    Chooses a random previously measured sequence and mutates it.

    A good baseline to compare other search strategies against.

    Since random search is not data-driven, the model is only used to score
    sequences, but not to guide the search strategy.
    """

    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: Optional[str] = None,
    ):
        """
        Create a random search explorer.

        Args:
            mu: Average number of residue mutations from parent for generated sequences.
            elitist: If true, will propose the top `sequences_batch_size` sequences
                generated according to `model`. If false, randomly proposes
                `sequences_batch_size` sequences without taking model score into
                account (true random search).
            seed: Integer seed for random number generator.

        """
        name = f"Random"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.env_batch_size = sequences_batch_size
        self.num_actions = 0
    
    @property
    def starting_state(self):
        return string_to_one_hot(self.starting_sequence, self.alphabet)

    def initialize_data_structures(self):
        """Initialize internal data structures."""
        self.state = self.starting_state.copy()
        self.state = self.state.reshape(1, *self.state.shape)
        self.state = self.state.repeat(self.env_batch_size, 0) # (env_batch_size, seq_len, alphabet_len)
        self.seq_len = len(self.starting_sequence)
        self.num_actions = self.seq_len * self.alphabet_size
        print("Initialize data structures")

    def get_random_action(self, current_states):
        assert (current_states.sum((-2, -1)) == self.seq_len).all()
        flat_states_per_env = current_states.reshape(self.env_batch_size, -1) 
        num_actions = flat_states_per_env.shape[-1]
        num_valid_actions = num_actions - self.seq_len
        valid_actions_mask = 1. - flat_states_per_env
        assert (valid_actions_mask.sum(-1) == num_valid_actions).all()
        random_valid_actions_probs = torch.FloatTensor(valid_actions_mask * (1. / num_valid_actions))
        assert np.allclose(random_valid_actions_probs.sum(-1), 1.)
        random_action_idx = Categorical(probs=random_valid_actions_probs).sample().numpy()
        return random_action_idx
    
    def get_mutant(self, state):
        action_idx = self.get_random_action(state)
        action_matrix = np.zeros(state.shape)
        x = action_idx // self.alphabet_size
        y = action_idx % self.alphabet_size
        action_matrix[np.arange(self.env_batch_size), x, y] = 1
        mutated_state = batched_construct_mutant_from_sample(action_matrix, state)
        return mutated_state
    
    def batched_pick_action(self):
        batched_new_state = self.get_mutant(self.state.copy())
        self.state = batched_new_state
        batched_new_state_strings = [
            one_hot_to_string(new_state, self.alphabet) for new_state in batched_new_state
        ]
        batched_reward, _ = self.model.get_fitness(batched_new_state_strings)
        return batched_new_state_strings, batched_reward

    def propose_sequences(self, measured_sequences: pd.DataFrame, round_idx: int):
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:
            self.initialize_data_structures()

        all_measured_seqs = set(measured_sequences["sequence"].values)
        sequences = {}
        prev_cost = self.model.cost

        pbar = tqdm(total=self.model_queries_per_batch, leave=False, desc="Model Queries")
        p_iter = self.model.cost
        while self.model.cost - prev_cost < self.model_queries_per_batch:
            batched_new_state_string, batched_preds = self.batched_pick_action()
            for new_state_string, proxy_fitness in zip(batched_new_state_string, batched_preds):
                all_measured_seqs.add(new_state_string)
                sequences[new_state_string] = proxy_fitness
            pbar.update(self.model.cost - p_iter)
            p_iter = self.model.cost
        pbar.close()

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[-self.sequences_batch_size:][::-1]

        proposed_seqs = new_seqs[sorted_order]
        proposed_seq_rewards = preds[sorted_order]
        proposed_seq_fitnesses = preds[sorted_order]
        edit_distances = np.array([int(editdistance.eval(seq, self.starting_sequence)) for seq in proposed_seqs])
        return proposed_seqs, proposed_seq_rewards, proposed_seq_fitnesses, edit_distances, None