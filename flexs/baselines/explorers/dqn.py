import collections
import random
import editdistance
from tqdm import tqdm
from typing import Optional, Tuple

import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from torch.distributions import Categorical

import numpy as np
import pandas as pd
import wandb

import flexs
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    batched_construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
)
from helper import calculate_gradient_norm

class QNetwork(nn.Module):
    def __init__(self, state_dims, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dims, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        q_values = self.network(x)
        return q_values


def build_q_network(sequence_len, alphabet_len, device):
    """Build the Q Network."""
    model = QNetwork(sequence_len * alphabet_len, sequence_len * alphabet_len).to(device)
    return model    


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN(flexs.Explorer):
    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        device: str = "cuda",
        lr: float = 2.5e-4,
        memory_size: int = 100000,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.05,
        exploration_fraction: float = 0.5,
        train_epochs: int = 20,
        target_network_frequency: int = 20,
        gamma: float = 0.9,
        tau: float = 1.0,
        allow_degen_actions: bool = True,
    ):
        name = "DQN_Explorer"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file=None,
        )

        self.seq_len = len(starting_sequence)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.best_fitness = 0
        self.device = device
        self.lr = lr
        self.env_batch_size = sequences_batch_size
        self.memory_size = memory_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.exploration_fraction = exploration_fraction
        self.total_loops = (self.model_queries_per_batch // self.env_batch_size) * rounds
        self.train_epochs = train_epochs
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.num_updates = 0
        self.num_actions = 0
        self.counter = 0
        self.allow_degen_actions = allow_degen_actions
        assert self.allow_degen_actions is True

    @property
    def starting_state(self):
        return string_to_one_hot(self.starting_sequence, self.alphabet)
    
    def initialize_data_structures(self):
        """Initialize internal data structures."""
        self.state = self.starting_state.copy()
        self.state = self.state.reshape(1, *self.state.shape)
        self.state = self.state.repeat(self.env_batch_size, 0) # (env_batch_size, seq_len, alphabet_len)
        self.num_actions = self.seq_len * self.alphabet_size

        self.q_network = build_q_network(self.seq_len, len(self.alphabet), self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.target_network = build_q_network(self.seq_len, len(self.alphabet), self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.num_updates = 0
        self.counter = 0
        self.memory = PrioritizedReplayBuffer(
            self.alphabet_size * self.seq_len,
            self.memory_size,
            self.sequences_batch_size,
            0.6,
        )
        tqdm.write("Initialize data structures")

    def q_network_loss(self, batch):
        rewards, actions, states, next_states = (
            torch.tensor(batch["rews"]).to(self.device),
            torch.LongTensor(batch["acts"][:, 0]).to(self.device),
            torch.tensor(batch["obs"]).to(self.device),
            torch.tensor(batch["next_obs"]).to(self.device),
        )
        buffer_batch_size = actions.shape[0]
        
        old_val = self.q_network(states)[torch.arange(buffer_batch_size), actions]
        with torch.no_grad():
            target_max, _ = self.target_network(next_states).max(-1)
            td_target = rewards + self.gamma * target_max

        # Ensure no broadcasting is going on
        assert old_val.shape == target_max.shape
        assert target_max.shape == rewards.shape
        loss = F.mse_loss(old_val, td_target)
        with torch.no_grad():
            metrics = {}
            metrics['loss'] = loss.cpu().numpy()
            metrics['target_max'] = target_max.mean().cpu().numpy()
            metrics['rewards'] = rewards.mean().cpu().numpy()
            metrics['td_target'] = td_target.mean().cpu().numpy()
            metrics['old_val'] = old_val.mean().cpu().numpy()
        return loss, metrics

    def train_actor(self, train_epochs):
        metrics = collections.defaultdict(list)
        for i in range(train_epochs):
            batch = self.memory.sample_batch()
            loss, met = self.q_network_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_updates += 1
            
            for k, v in met.items():
                metrics[k].append(v)

            if self.num_updates % self.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_network_param.data.copy_(
                            self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                        )
        
        metrics = {f"train_metrics/{k}": np.mean(v) for k, v in metrics.items()}
        return metrics

    def get_random_action_no_degen(self, current_states):
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
    
    def get_random_action(self, current_states):
        """
        Get a random action for the current state.
        This is used for the degen action case.
        """
        random_action_idx = np.random.randint(self.seq_len * self.alphabet_size, size=len(current_states))
        return random_action_idx

    def get_action_and_mutant(self, state, epsilon):
        """Return an action and the resulting mutant."""
        # Ensure that staying in place gives no reward
        assert (state.sum(-1) == 1).all()

        explore_vector = np.random.random(len(state)) < epsilon
        
        # Explore: Sample a random action
        if self.allow_degen_actions:
            random_action_idx = self.get_random_action(state)
        else:
            random_action_idx = self.get_random_action_no_degen(state)
        
        # Choose the greedy action
        batched_flat_states = torch.FloatTensor(state.reshape(self.env_batch_size, -1)).to(self.device)
        pred_q_values = self.q_network(batched_flat_states).cpu().detach().numpy()
        assert pred_q_values.shape == (self.env_batch_size, self.seq_len * self.alphabet_size)
        greedy_action_idx = np.argmax(pred_q_values, axis=-1) # argmax across action dimension
        action_idx = np.where(explore_vector, random_action_idx, greedy_action_idx) 
        
        action_matrix = np.zeros(state.shape)
        x = action_idx // self.alphabet_size
        y = action_idx % self.alphabet_size
        action_matrix[np.arange(self.env_batch_size), x, y] = 1
        mutated_state = batched_construct_mutant_from_sample(action_matrix, state)
        return action_idx, mutated_state

    def batched_pick_action(self, all_measured_seqs, counter):
        duration = int(self.exploration_fraction * self.total_loops)
        epsilon = linear_schedule(self.epsilon_max, self.epsilon_min, duration, counter)
        old_state = self.state.copy()
        batched_action, batched_new_state = self.get_action_and_mutant(self.state.copy(), epsilon)
        self.state = batched_new_state
        batched_new_state_strings = [
            one_hot_to_string(new_state, self.alphabet) for new_state in batched_new_state
        ]
        batched_reward, _ = self.model.get_fitness(batched_new_state_strings)

        for old_state_i, action, reward, new_state_i, new_state_string in zip(old_state, batched_action, batched_reward, batched_new_state, batched_new_state_strings):
            if new_state_string not in all_measured_seqs:
                self.best_fitness = max(self.best_fitness, reward)
            self.memory.store(
                old_state_i.ravel(), 
                action, 
                reward, 
                new_state_i.ravel()
            )

        if self.model.cost > 0 and len(self.memory) >= self.sequences_batch_size:
            mets = self.train_actor(self.train_epochs)
            wandb.log(mets)

        return batched_new_state_strings, batched_reward, epsilon

    def propose_sequences(self, measured_sequences_data: pd.DataFrame, round_idx=None) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()

        all_measured_seqs = set(measured_sequences_data["sequence"].values)
        sequences = {}
        prev_cost = self.model.cost

        pbar = tqdm(total=self.model_queries_per_batch, leave=False, desc="Model Queries")
        p_iter = self.model.cost
        while self.model.cost - prev_cost < self.model_queries_per_batch:
            batched_new_state_string, batched_preds, eps = self.batched_pick_action(all_measured_seqs, self.counter)
            self.counter += 1
            for new_state_string, proxy_fitness in zip(batched_new_state_string, batched_preds):
                all_measured_seqs.add(new_state_string)
                sequences[new_state_string] = proxy_fitness
            pbar.update(self.model.cost - p_iter)
            p_iter = self.model.cost
        pbar.close()
        
        mets = {
            'Exploration/epsilon': eps,
        }
        with torch.no_grad():
            temp_state = torch.FloatTensor(self.state.reshape(self.env_batch_size, -1)).to(self.device)
            pred_q_values = self.q_network(temp_state)
            pred_q_probs = F.softmax(pred_q_values - pred_q_values.max(-1).values[..., None], dim=-1)
            entropy = -(pred_q_probs * torch.log(pred_q_probs)).sum(-1).mean()
            mets['train_metrics/pseudo_action_probs'] = pred_q_probs.mean()
            mets['train_metrics/pseudo_entropy'] = entropy
        mets['train_metrics/q_network_gradient_norm'] = calculate_gradient_norm(self.q_network)

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[-self.sequences_batch_size:][::-1]

        proposed_seqs = new_seqs[sorted_order]
        proposed_seq_rewards = preds[sorted_order]
        proposed_seq_fitnesses = preds[sorted_order]
        edit_distances = np.array([int(editdistance.eval(seq, self.starting_sequence)) for seq in proposed_seqs])
        return proposed_seqs, proposed_seq_rewards, proposed_seq_fitnesses, edit_distances, mets