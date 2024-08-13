import os
from typing import Optional

import numpy as np
import pandas as pd
from termcolor import colored
import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import wandb

import flexs
from flexs.utils.replay_buffers import PrioritizedReplayBuffer
from flexs.utils.sequence_utils import (
    batched_construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
)
import editdistance

from helper import calculate_gradient_norm

class Critic(nn.Module):
    """
        Approx Q-function: Q(s, a)
        For the discrete case, the network just gives a vector Q(s) that has `num_action` elements,
        which is the Q value of the state for each possible action.
    """
    def __init__(self, state_dim, seq_len, num_actions, device, name='critic', use_layer_norm=False):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.name = name
        self.device = device

        self.linear1 = nn.Linear(state_dim, state_dim)
        self.linear2 = nn.Linear(state_dim, seq_len)
        self.linear3 = nn.Linear(seq_len, num_actions)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            print(colored("Using Layer Norm in Critic", "light_red"))
            self.norm1 = nn.LayerNorm(state_dim)
            self.norm2 = nn.LayerNorm(seq_len)
        else:
            print(colored("Using Batch Norm in Critic", "light_green"))
            self.bn1 = nn.BatchNorm1d(state_dim)
            self.bn2 = nn.BatchNorm1d(seq_len)

        self.to(self.device)
    
    def forward(self, state):  # pylint: disable=W0221
        """
            Take a forward step.

            state_dim: self.sequence_len * self.alphabet_len
            action_dim: self.sequence_len * self.alphabet_len

            state: (b, state_dim)
        """
        if self.use_layer_norm:
            # Layernorm should be applied before activation function
            action_value = F.relu(self.norm1(self.linear1(state)))
            action_value = F.relu(self.norm2(self.linear2(action_value)))
        else:
            action_value = self.bn1(F.relu(self.linear1(state)))
            action_value = self.bn2(F.relu(self.linear2(action_value)))
        action_value = self.linear3(action_value)
        return action_value


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device, fc1_dims=256, fc2_dims=256, name='actor'):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.action_dim = action_dim # number of discrete actions the agent could take
        self.device = device

        self.linear1 = nn.Linear(state_dim, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.categorical_logits = nn.Linear(self.fc2_dims, self.action_dim)

        self.to(self.device)

    def forward(self, state):
        logits = F.relu(self.linear1(state))
        logits = F.relu(self.linear2(logits))
        logits = self.categorical_logits(logits)
        return logits

    def get_action(self, state):
        logits = self(state)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1) # (env_batch_size, seq_len * alphabet_len)
        return action, log_prob, action_probs


def build_networks(state_dim, seq_len, num_actions, device, use_layer_norm=False):
    actor = ActorNetwork(state_dim, num_actions, device, name='actor')
    qf1 = Critic(state_dim, seq_len, num_actions, device, name='qf1', use_layer_norm=use_layer_norm)
    qf2 = Critic(state_dim, seq_len, num_actions, device, name='qf2', use_layer_norm=use_layer_norm)
    qf1_target = Critic(state_dim, seq_len, num_actions, device, name='qf1_target', use_layer_norm=use_layer_norm)
    qf2_target = Critic(state_dim, seq_len, num_actions, device, name='qf2_target', use_layer_norm=use_layer_norm)
    return actor, qf1, qf2, qf1_target, qf2_target

def sanitize_tensor(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def get_batch_stats(name, tensor):
    tensor = sanitize_tensor(tensor)
    return {
        f"{name}_min": tensor.min(),
        f"{name}_max": tensor.max(),
        f"{name}_mean": tensor.mean(),
    }

def dict_update(dict1: dict, dict2: dict, prefix=None):
    if prefix is None:
        dict1.update(dict2)
    else:
        for k in dict2.keys():
            dict1[f"{prefix}{k}"] = dict2[k]

class BatchedDiscreteSAC(flexs.Explorer):
    def __init__(
        self,
        model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        train_batch_size: int = None,
        log_file: Optional[str] = None,
        memory_size: int = 100000,
        train_epochs: int = 20,
        gamma: float = 0.99,
        device: str = "cuda:0",
        tau: float = 0.005,
        alpha: float = 1.0,
        q_lr: float = 3e-4,
        policy_lr: float = 3e-4,
        target_network_frequency: int = 5,
        autotune: bool = True,
        target_entropy_scale: float = 0.89,
        opt_eps: float = 1e-4,
        use_layer_norm=False,
        allow_degen_actions=True,
        start_seq_batch=None,
    ):

        name = "DSAC_Explorer"
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
        self.memory_size = memory_size
        self.gamma = gamma
        self.best_fitness = 0
        self.train_epochs = train_epochs
        self.num_actions = 0
        self.device = device
        self.env_batch_size = sequences_batch_size
        self.start_seq_batch = start_seq_batch
        self.train_batch_size = train_batch_size if train_batch_size is not None else sequences_batch_size
        print(colored(f"Setting train_batch_size to {train_batch_size}", "red"))

        self.tau = tau
        self.alpha = alpha
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.target_network_frequency = target_network_frequency
        self.autotune = autotune
        self.target_entropy_scale = target_entropy_scale
        self.opt_eps = opt_eps
        self.use_layer_norm = use_layer_norm
        self.allow_degen_actions = allow_degen_actions
        try:
            assert self.allow_degen_actions is True
        except:
            raise NotImplementedError

        self.state = None
        self.seq_len = None
        self.memory = None
        
        self.actor = None
        self.qf1 = None
        self.qf2 = None
        self.qf1_target = None
        self.qf2_target = None

        self.q_optimizer = None
        self.actor_optimizer = None

    @property
    def starting_state(self):
        return string_to_one_hot(self.starting_sequence, self.alphabet)

    def initialize_data_structures(self):
        """Initialize internal data structures."""
        if self.start_seq_batch is None:
            tqdm.write(f"Repeating single state for init env batch")
            self.state = self.starting_state.copy()
            self.state = self.state.reshape(1, *self.state.shape)
            self.state = self.state.repeat(self.env_batch_size, 0) # (env_batch_size, seq_len, alphabet_len)
        else:
            tqdm.write(f"Multiple random state used for init env batch")
            self.state = np.array([string_to_one_hot(sq, self.alphabet) for sq in self.start_seq_batch])
            assert self.state.shape[0] == self.env_batch_size and self.state.shape[1:] == self.starting_state.shape
        
        self.seq_len = len(self.starting_sequence)
        self.state_dims = self.seq_len * self.alphabet_size
        self.num_actions = self.seq_len * self.alphabet_size

        self.memory = PrioritizedReplayBuffer(
            self.alphabet_size * self.seq_len,
            self.memory_size,
            self.sequences_batch_size,
            0.6,
        )

        actor, qf1, qf2, qf1_target, qf2_target = build_networks(
            state_dim=self.state_dims,
            seq_len=self.seq_len,
            num_actions=self.num_actions,
            device=self.device,
            use_layer_norm=self.use_layer_norm
        )

        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf1_target = qf1_target
        self.qf2_target = qf2_target

        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), 
            lr=self.q_lr, 
            eps=self.opt_eps
        )
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr, eps=self.opt_eps)

        if self.autotune:
            self.target_entropy = -self.target_entropy_scale * torch.log(1 / torch.tensor(self.num_actions))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr, eps=self.opt_eps)
        
        print("Initialize data structures")
        
    def update_target_networks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action_and_mutant(self, state):
        # state: #(env_batch_size, seq_len, alphabet_len)
        flat_state_tensor = torch.FloatTensor(state.reshape(self.env_batch_size, -1)).to(self.device)
        action, _, _ = self.actor.get_action(flat_state_tensor)
        action = action.cpu()

        action_matrix = np.zeros(state.shape)
        x = action // self.alphabet_size
        y = action % self.alphabet_size
        action_matrix[np.arange(self.env_batch_size), x, y] = 1

        mutated_state = batched_construct_mutant_from_sample(action_matrix, state)
        return action.cpu().numpy(), mutated_state

    @torch.no_grad()
    def get_action_probs(self, states):
        self.actor.eval()
        if len(states.shape) == 1:
            states = states[None]
        if not torch.is_tensor(states):
            states = torch.FloatTensor(states).to(self.device)
        action_probs = F.softmax(self.actor(states))
        self.actor.train()
        return action_probs

    @torch.no_grad()
    def get_value(self, states):
        self.qf1.eval()
        self.qf2.eval()
        if len(states.shape) == 1:
            states = states[None]
        if not torch.is_tensor(states):
            states = torch.FloatTensor(states).to(self.device)
        qf1_values = self.qf1(states)
        qf2_values = self.qf2(states) 
        self.qf1.train()
        self.qf2.train()
        return torch.min(qf1_values, qf2_values)

    def train_critic(self, batch, return_metrics=False):
        rewards, actions, states, next_states = (
            torch.tensor(batch["rews"]).to(self.device),
            torch.tensor(batch["acts"][:, 0:1]).to(self.device),
            torch.tensor(batch["obs"]).to(self.device),
            torch.tensor(batch["next_obs"]).to(self.device),
        )

        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(next_states)
            qf1_next_target = self.qf1_target(next_states)
            qf2_next_target = self.qf2_target(next_states)
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = rewards + self.gamma * (min_qf_next_target)
        
        # use Q-values only for the taken actions
        qf1_values = self.qf1(states)
        qf2_values = self.qf2(states) 
        qf1_a_values = qf1_values.gather(1, actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, actions.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        if return_metrics:
            metrics = {
                "qf1_loss": sanitize_tensor(qf1_loss),
                "qf2_loss": sanitize_tensor(qf2_loss),
            }
            metrics.update(get_batch_stats("qf1_a_values", qf1_a_values))
            metrics.update(get_batch_stats("qf2_a_values", qf2_a_values))
            metrics.update(get_batch_stats("qf1_values", qf1_values))
            metrics.update(get_batch_stats("qf2_values", qf2_values))
            metrics.update(get_batch_stats("next_q_value", next_q_value))
            metrics.update(get_batch_stats("min_qf_next_target", min_qf_next_target))
            return qf_loss, metrics
        
        return qf_loss, {}

    def train_actor(self, batch, return_metrics=False):
        states = torch.tensor(batch["obs"]).to(self.device)

        _, log_pi, action_probs = self.actor.get_action(states.to(self.device))
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)

        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()
        
        if return_metrics:
            metrics = {
                "actor_loss": sanitize_tensor(actor_loss),
            }
            return actor_loss, log_pi, action_probs, metrics
        return actor_loss, log_pi, action_probs, {}

    def train_actor_critic(self, train_epochs):
        
        # epoch_metrics = []
        for i in range(train_epochs):
            metrics = {}
            batch = self.memory.sample_batch()

            critic_loss, critic_metrics = self.train_critic(batch, return_metrics= i == train_epochs - 1)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self.q_optimizer.step()

            actor_loss, log_pi, action_probs, actor_metrics = self.train_actor(batch, return_metrics= i == train_epochs - 1)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.autotune:
                # re-use action probabilities for temperature loss
                alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

                if i == train_epochs - 1:
                    alpha_metrics = {
                        "alpha_loss": sanitize_tensor(alpha_loss),
                        "alpha": self.alpha
                    }

            # dict_update(metrics, critic_metrics, "rl_metrics/critic/")
            # dict_update(metrics, actor_metrics, "rl_metrics/actor/")
            # epoch_metrics.append(metrics)
            if i % self.target_network_frequency == 0:
                self.update_target_networks()        
        
        dict_update(metrics, critic_metrics, "rl_metrics/critic/")
        dict_update(metrics, actor_metrics, "rl_metrics/actor/")
        if self.autotune:
            dict_update(metrics, alpha_metrics, "rl_metrics/alpha/")
        # keys = metrics.keys()
        # metrics = {k: np.mean([em[k] for em in epoch_metrics], 0) for k in keys}
        with torch.no_grad():
            entropy = -(action_probs * log_pi).sum(-1).mean()
            metrics["rl_metrics/actor/entropy"] = entropy.item()
        return metrics

    def get_model_norms(self):
        return {
            "grad_norm/qf1": calculate_gradient_norm(self.qf1),
            "grad_norm/qf2": calculate_gradient_norm(self.qf2),
            "grad_norm/actor": calculate_gradient_norm(self.actor),
        }
        
    def batched_pick_action(self, all_measured_seqs):
        old_state = self.state.copy()
        batched_action, batched_new_state = self.get_action_and_mutant(self.state)
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
        # print(f"Store in buffer: {time.time() - start_time}")

        if self.model.cost > 0 and len(self.memory) >= self.sequences_batch_size:
            # start_time = time.time()
            self.train_actor_critic(self.train_epochs)
            # print(f"Training actor critic for {self.train_epochs} epochs: {time.time() - start_time}")
        
        return batched_new_state_strings, batched_reward

    def propose_sequences(self, measured_sequences_data: pd.DataFrame, round_idx=None):
        """Propose top `sequences_batch_size` sequences for evaluation."""
        if self.num_actions == 0:   # indicates model was reset
            self.initialize_data_structures()

        all_measured_seqs = set(measured_sequences_data["sequence"].values)
        sequences = {}
        prev_cost = self.model.cost

        pbar = tqdm(total=self.model_queries_per_batch, leave=False, desc="Model Queries")
        p_iter = self.model.cost
        while self.model.cost - prev_cost < self.model_queries_per_batch:
            batched_new_state_string, batched_preds = self.batched_pick_action(all_measured_seqs)
            for new_state_string, proxy_fitness in zip(batched_new_state_string, batched_preds):
                all_measured_seqs.add(new_state_string)
                sequences[new_state_string] = proxy_fitness
            pbar.update(self.model.cost - p_iter)
            p_iter = self.model.cost
        pbar.close()

        if round_idx is not None and round_idx % 100 == 0:
            with torch.no_grad():
                start_qs = self.get_value(self.starting_state.ravel())[0]
                start_qs = start_qs.reshape(self.starting_state.shape).detach().cpu().numpy()
                wandb.log({'heatmaps/start_qs': wandb.plots.HeatMap(self.alphabet, np.arange(start_qs.shape[0]), np.round(start_qs, 3), show_text=False)})
                action_probs = self.get_action_probs(self.starting_state.ravel())[0]
                action_probs = action_probs.reshape(self.starting_state.shape).detach().cpu().numpy()
                wandb.log({'heatmaps/start_action_probs': wandb.plots.HeatMap(self.alphabet, np.arange(action_probs.shape[0]), np.round(action_probs, 4), show_text=False)})

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[-self.sequences_batch_size:][::-1]

        proposed_seqs = new_seqs[sorted_order]
        proposed_seq_rewards = preds[sorted_order]
        proposed_seq_fitnesses = preds[sorted_order]
        edit_distances = np.array([int(editdistance.eval(seq, self.starting_sequence)) for seq in proposed_seqs])
        return proposed_seqs, proposed_seq_rewards, proposed_seq_fitnesses, edit_distances, None
    
    def get_model_dict(self, save_dir):
        ckpt = {
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'qf1_target': self.qf1_target.state_dict(),
            'qf2_target': self.qf2_target.state_dict(),
        }
        ckpt_fname = os.path.join(save_dir, "ckpt.pt")
        torch.save(ckpt, ckpt_fname)
        return ckpt_fname

