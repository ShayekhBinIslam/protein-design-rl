import random
from typing import Optional
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import flexs
from flexs.utils import sequence_utils as s_utils
from flexs.baselines.explorers.environments.esm_env import ESMEnvironment
from flexs.utils.replay_buffers import PrioritizedReplayBuffer

from language.program import ProgramNode
import editdistance


class Q_Network(nn.Module):
    """Q Network implementation, used in DQN Explorer."""

    def __init__(self, dim, sequence_len):
        """Initialize the Q Network."""
        super(Q_Network, self).__init__()
        self.linear1 = nn.Linear(2 * dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, sequence_len)
        self.bn2 = nn.BatchNorm1d(sequence_len)
        self.linear3 = nn.Linear(sequence_len, 1)

    def forward(self, x):  # pylint: disable=W0221
        """Take a forward step."""
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = F.relu(self.linear3(x))
        return x


def build_q_network(dim, sequence_len, device):
    """Build the Q Network."""
    model = Q_Network(dim, sequence_len).to(device)
    return model


class DQN_ESM(flexs.Explorer):
    def __init__(
        self,
        env: ESMEnvironment,
        model: flexs.Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_program: ProgramNode,
        alphabet: str,
        log_file: Optional[str] = None,
        memory_size: int = 100000,
        train_epochs: int = 20,
        gamma: float = 0.9,
        device: str = "cuda",
        is_proxy_oracle: bool = False
    ):
        name = "DQN_ESM_Explorer"

        self.starting_sequence = starting_program.get_sequence_and_set_residue_index_ranges()[0].split(":")[0]
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            self.starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.env = env
        self.memory_size = memory_size
        self.train_epochs = train_epochs
        self.gamma = gamma
        self.device = device
        self.is_proxy_oracle = is_proxy_oracle

        self.best_fitness = 0
        self.epsilon_min = 0.1
        self.top_sequence = []
        self.num_actions = 0

        self.q_network = None
        self.memory = None

    def initialize_data_structures(self):
        """Initialize internal data structures."""
        self.env._reset()

        self.q_network = build_q_network(
            self.env.states.shape[1] * self.env.states.shape[0], 
            self.env.seq_max_len, 
            self.device
        )
        self.q_network.eval() # TODO: Why is this in eval mode?
        self.memory = PrioritizedReplayBuffer(
            self.env.states.shape[1] * self.env.states.shape[0], # self.env.states.shape[0] is self.env.seq_max_len
            self.memory_size,
            self.sequences_batch_size,
            0.6,
        )

    def calculate_next_q_values(self, state_v):
        """Calculate the next Q values."""
        dim = np.prod(self.env.states.shape)
        states_repeated = state_v.repeat(1, dim).reshape(-1, dim)
        actions_repeated = torch.FloatTensor(np.identity(dim)).repeat(len(state_v), 1).to(self.device)
        next_states_actions = torch.cat((states_repeated, actions_repeated), 1)
        next_states_values = self.q_network(next_states_actions)
        next_states_values = next_states_values.reshape(len(state_v), -1)

        # `next_states_values` is of shape (1, self.env.states.shape[0] * self.env.states.shape[1])
        # corresponding to every state-action pair, even including states where we cant take actions
        # For eg., cant take actions from ConstantSequenceSegment. Thus we, care only about the 
        # effective state-space x action-space size which is self.env.num_actions
        next_states_values = next_states_values[:, :self.env.num_actions]
        return next_states_values

    def q_network_loss(self, batch):
        """Calculate MSE.

        Computes between actual state action values, and expected state action values
        from DQN.
        """
        rewards, actions, states, next_states = (
            batch["rews"],
            batch["acts"],
            batch["obs"],
            batch["next_obs"],
        )

        state_action_v = torch.FloatTensor(np.hstack((states, actions))).to(self.device)
        rewards_v = torch.FloatTensor(rewards).to(self.device)
        next_states_v = torch.FloatTensor(next_states).to(self.device)

        state_action_values = self.q_network(state_action_v).view(-1)
        next_state_values = self.calculate_next_q_values(next_states_v)
        next_state_values = next_state_values.max(1)[0].detach()
        expected_state_action_values = next_state_values * self.gamma + rewards_v

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def train_actor(self, train_epochs):
        """Train the Q Network."""
        total_loss = 0.0
        # train Q network on new samples
        optimizer = optim.Adam(self.q_network.parameters())
        for _ in range(train_epochs):
            batch = self.memory.sample_batch()
            optimizer.zero_grad()
            loss = self.q_network_loss(batch)
            loss.backward()
            clip_grad_norm_(self.q_network.parameters(), 1.0, norm_type=1)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / train_epochs

    def get_action_and_mutant(self, state, epsilon):
        """Return an action and the resulting mutant."""

        # Explore-exploit
        if random.random() < epsilon:
            action_idx = self.env.sample_random_action() 
        else:
            state_tensor = torch.FloatTensor([state.ravel()]).to(self.device)
            prediction = self.calculate_next_q_values(state_tensor).cpu().detach().numpy()
            prediction = prediction.reshape((self.env.num_actions))

            # TODO: Is it right to keep this commented? 
            # TODO: Risk: Action that leads to same state will be given rewards
            # TODO: If this is handled in env._step, this can be removed
            # zero_current_state = (self.state - 1) * (-1)
            # moves = np.multiply(prediction, zero_current_state)
            action_idx = np.argmax(prediction) # Choose the greedy action

        i, j = self.env.action_to_state_map[int(action_idx)]
        action = np.zeros(self.env.states.shape)
        action[i, j] = 1

        # ! DO NOT REMOVE THESE 4 LINES; 
        # ! Currently actions are still allowed to be chosen even if they lead to the same state 
        # try:
        #     assert int(state[i, j]) != 1
        # except:
        #     import pdb; pdb.set_trace()

        # Make action, obtain new state
        res = self.env._step(action_idx)
        new_one_hot_state = self.env.states.copy()

        # TODO: Might need more logic here based on termination?
        # - Need to reset or already taken care of by ts.termination?

        return action, new_one_hot_state, res.reward

    def pick_action(self, all_measured_seqs):
        eps = max(
            self.epsilon_min,
            (0.5 - self.model.cost / (self.sequences_batch_size * self.rounds)),
        )
        
        one_hot_state = self.env.states.copy()
        action, new_one_hot_state, reward = self.get_action_and_mutant(one_hot_state, eps)

        new_state_string = s_utils.one_hot_to_string(new_one_hot_state, self.alphabet)
        fitness = self.model.get_fitness([new_state_string])
        if self.is_proxy_oracle is False:
            fitness = fitness.item()
        else:
            assert len(fitness) == 1
            fitness = fitness[0]

        if new_state_string not in all_measured_seqs:
            if fitness >= self.best_fitness:
                self.top_sequence.append((reward, new_one_hot_state, self.model.cost, fitness))

            self.best_fitness = max(self.best_fitness, fitness)

            self.memory.store(one_hot_state.ravel(), action.ravel(), reward, new_one_hot_state.ravel())
        if (
            self.model.cost > 0
            and self.model.cost % self.sequences_batch_size == 0
            and len(self.memory) >= self.sequences_batch_size
        ):
            self.train_actor(self.train_epochs)
        self.num_actions += 1
        
        return new_state_string, fitness, reward
    
    def propose_sequences(self, measured_sequences_data: pd.DataFrame):
        # indicates model was reset
        if self.num_actions == 0:
            self.initialize_data_structures()

        all_measured_seqs = set(measured_sequences_data["sequence"].values)
        reward_sequences = {}
        fitness_sequences = {}

        prev_cost = self.model.cost
        while self.model.cost - prev_cost < self.model_queries_per_batch:
            new_state_string, fitness, reward = self.pick_action(all_measured_seqs)
            all_measured_seqs.add(new_state_string)
            reward_sequences[new_state_string] = reward
            fitness_sequences[new_state_string] = fitness

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(reward_sequences.keys()))
        rewards = np.array(list(reward_sequences.values()))
        fitnesses = np.array(list(fitness_sequences.values()))
        
        sorted_order = np.argsort(fitnesses)[: -self.sequences_batch_size : -1]

        proposed_seqs = new_seqs[sorted_order]
        proposed_seq_rewards = rewards[sorted_order]
        proposed_seq_fitnesses = fitnesses[sorted_order]
        edit_distances = np.array([int(editdistance.eval(seq, self.starting_sequence)) for seq in proposed_seqs])
        return proposed_seqs, proposed_seq_rewards, proposed_seq_fitnesses, edit_distances
