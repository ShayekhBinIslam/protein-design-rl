"""DyNA-PPO environment module."""
import editdistance
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils

import flexs
from flexs.utils import sequence_utils as s_utils


class DynaPPOEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """DyNA-PPO environment based on TF-Agents."""

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        seq_length: int,
        model: flexs.Model,
        landscape: flexs.Landscape,
        batch_size: int,
        penalty_scale = 0.1,
        distance_radius = 2
    ):
        """
        Initialize DyNA-PPO agent environment.

        Based on this tutorial:
        https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent

        Args:
            alphabet: Usually UCGA.
            starting_seq: When initializing the environment,
                the sequence which is initially mutated.
            model: Landscape or model which evaluates
                each sequence.
            landscape: True fitness landscape.
            batch_size: Number of epsisodes to batch together and run in parallel.

        """
        self.alphabet = alphabet
        self._batch_size = batch_size
        self.lam = penalty_scale
        self.dist_radius = distance_radius

        self.seq_length = seq_length
        self.partial_seq_len = 0
        self.states = np.zeros(
            (batch_size, seq_length, len(alphabet) + 1), dtype="float32"
        )
        self.states[:, np.arange(seq_length), -1] = 1

        # model/model/measurements
        self.model = model
        self.landscape = landscape
        self.fitness_model_is_gt = False
        self.previous_fitness = -float("inf")

        # sequence
        self.all_seqs = {}
        self.all_seqs_uncert = {}
        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.integer,
            minimum=0,
            maximum=len(self.alphabet) - 1,
            name="action",
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.seq_length, len(self.alphabet) + 1),
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._time_step_spec = ts.time_step_spec(self._observation_spec)

        super().__init__()

    def _reset(self):
        self.partial_seq_len = 0
        self.states[:, :, :] = 0
        self.states[:, np.arange(self.seq_length), -1] = 1
        return nest_utils.stack_nested_arrays(
            [ts.restart(seq_state) for seq_state in self.states]
        )

    @property
    def batched(self):
        """Tf-agents function that says that this env returns batches of timesteps."""
        return True

    @property
    def batch_size(self):
        """Tf-agents property that return env batch size."""
        return self._batch_size

    def time_step_spec(self):
        """Define time steps."""
        return self._time_step_spec

    def action_spec(self):
        """Define agent actions."""
        return self._action_spec

    def observation_spec(self):
        """Define environment observations."""
        return self._observation_spec

    def sequence_density(self, seq):
        """Get average distance to `seq` out of all observed sequences."""
        dens = 0
        exactly_eq = False
        for s in self.all_seqs:
            dist = int(editdistance.eval(s, seq))
            if dist <= self.dist_radius:
                dens += self.all_seqs[s] / (dist + 1)
            #elif dist == 0:
            #    exactly_eq = True
            #    break

        return dens if not exactly_eq else np.inf

    def get_cached_fitness(self, seq):
        """Get cached sequence fitness computed in previous episodes."""
        return self.all_seqs[seq] if seq else 0.

    def get_cached_uncertainty(self, seq):
        return self.all_seqs_uncert[seq] if seq else 0.

    def set_fitness_model_to_gt(self, fitness_model_is_gt):
        """
        Set the fitness model to the ground truth landscape or to the model.

        Call with `True` when doing an experiment-based training round
        and call with `False` when doing a model-based training round.
        """
        self.fitness_model_is_gt = fitness_model_is_gt

    def _compute_rewards_non_empty(self, seqs):
        if len(seqs) == 0:
            return []

        if self.fitness_model_is_gt:
            fitnesses = self.landscape.get_fitness(seqs)
        else:
            fitnesses, uncerts = self.model.get_fitness(seqs, compute_uncert=True)

        # Reward = fitness - lambda * sequence density
        penalty = np.zeros(len(seqs))
        if not np.isclose(0., self.lam):
            penalty = np.array([
                self.lam * self.sequence_density(seq)
                for seq in seqs
            ])

        self.all_seqs.update(zip(seqs, fitnesses))
        if not self.fitness_model_is_gt:
            self.all_seqs_uncert.update(zip(seqs, uncerts))

        rewards = fitnesses - penalty
        return rewards

    def _compute_reward(self):
        if len(self.states.shape) == 2:
            seqs_to_compute = np.expand_dims(seqs_to_compute, axis=0)

        complete_sequences = [
            s_utils.one_hot_to_string(seq_state[:, :-1], self.alphabet)
            for seq_state in self.states
        ]

        to_compute = list(filter(lambda x: x != '', complete_sequences))
        non_empty_seqs_rewards = self._compute_rewards_non_empty(to_compute)

        i, rewards = 0, np.zeros(len(complete_sequences))
        for j, seq in enumerate(complete_sequences):
            if seq:
                rewards[j] = non_empty_seqs_rewards[i]
                i += 1

        return rewards

    def _step(self, actions):
        """Progress the agent one step in the environment."""
        if self.partial_seq_len != self.seq_length:
            actions = actions.flatten()
            self.states[:, self.partial_seq_len, -1] = 0
            self.states[np.arange(self._batch_size), self.partial_seq_len, actions] = 1

        self.partial_seq_len += 1

        # We have not generated the last residue in the sequence, so continue
        if self.partial_seq_len < self.seq_length:
            return nest_utils.stack_nested_arrays(
                [ts.transition(seq_state, 0) for seq_state in self.states]
            )

        rewards = self._compute_reward()
        return nest_utils.stack_nested_arrays(
            [ts.termination(seq_state, r) for seq_state, r in zip(self.states, rewards)]
        )


class DynaPPOEnvironmentStoppableEpisode(DynaPPOEnvironment):  # pylint: disable=W0223
    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        seq_length: int,
        model: flexs.Model,
        landscape: flexs.Landscape,
        batch_size: int,
        penalty_scale = 0.1,
        distance_radius = 2
    ):
        super().__init__(alphabet, seq_length, model, landscape, batch_size)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.integer,
            minimum=0,
            maximum=len(self.alphabet),
            name="action",
        )

        self.stop_action = len(alphabet)

        self.terminal_transitions = {}
        self.lam = penalty_scale
        self.dist_radius = distance_radius

    def _reset(self):
        self.terminal_transitions = {}
        return super()._reset()

    def sequence_density(self, seq):
        """Get average distance to `seq` out of all observed sequences."""
        dens = 0
        exactly_eq = False
        for s in self.all_seqs:
            dist = int(editdistance.eval(s, seq))
            if dist <= self.dist_radius:
                dens += self.all_seqs[s] / (dist + 1)
            #elif dist == 0:
            #    exactly_eq = True
            #    break

        return dens if not exactly_eq else np.inf

    def _compute_rewards_non_empty(self, seqs):
        if len(seqs) == 0:
            return []

        if self.fitness_model_is_gt:
            fitnesses = self.landscape.get_fitness(seqs)
        else:
            fitnesses, uncerts = self.model.get_fitness(seqs, compute_uncert=True)

        # Reward = fitness - lambda * sequence density
        penalty = np.zeros(len(seqs))
        if not np.isclose(0., self.lam):
            penalty = np.array([
                self.lam * self.sequence_density(seq)
                for seq in seqs
            ])

        self.all_seqs.update(zip(seqs, fitnesses))
        if not self.fitness_model_is_gt:
            self.all_seqs_uncert.update(zip(seqs, uncerts))

        rewards = fitnesses - penalty
        return rewards

    def _compute_reward(self, seqs_to_compute):
        if len(seqs_to_compute) == 0:
            return None

        if len(seqs_to_compute.shape) == 2:
            seqs_to_compute = np.expand_dims(seqs_to_compute, axis=0)

        complete_sequences = [
            s_utils.one_hot_to_string(seq_state, self.alphabet)
            for seq_state in seqs_to_compute
        ]

        to_compute = list(filter(lambda x: x != '', complete_sequences))
        non_empty_seqs_rewards = self._compute_rewards_non_empty(to_compute)

        i, rewards = 0, np.zeros(len(complete_sequences))
        for j, seq in enumerate(complete_sequences):
            if seq:
                rewards[j] = non_empty_seqs_rewards[i]
                i += 1

        return rewards

    def _is_done_transition(self, action):
        return (
            action == self.stop_action or
            self.partial_seq_len == self.seq_length
        )

    def _get_should_compute_reward_idx(self, actions):
        active_idxs = ~self.current_time_step().is_last()

        final_indicator_arr = None
        if self.partial_seq_len == self.seq_length:
            final_indicator_arr = active_idxs
        else:
            final_indicator_arr = active_idxs & (actions == self.stop_action)

        return tf.reshape(tf.where(final_indicator_arr), -1)

    def _step(self, actions):
        """Progress the agent one step in the environment."""
        #if actions[0] == self.stop_action:
        #    print('Agent stopped seq at action %d' % self.partial_seq_len)

        current_time_step = self.current_time_step()
        state_terminal_ind = current_time_step.is_last()

        active_seq_actions = actions.flatten()[~state_terminal_ind]
        if len(active_seq_actions) != 0:
            insert_action_idxs = tf.where(active_seq_actions != self.stop_action)
            insert_action_idxs = insert_action_idxs.numpy().flatten()
            insert_actions = active_seq_actions[insert_action_idxs]

            self.states[insert_action_idxs, self.partial_seq_len, -1] = 0
            self.states[insert_action_idxs, self.partial_seq_len, insert_actions] = 1
            self.partial_seq_len += 1

        compute_rewards_idx = self._get_should_compute_reward_idx(actions)
        rewards = self._compute_reward(self.states[compute_rewards_idx.numpy()])

        transitions = []
        for i in range(len(self.states)):
            transition = None
            if state_terminal_ind[i]:
                transition = self.terminal_transitions[i]
            elif i in compute_rewards_idx:
                reward = rewards[tf.squeeze(tf.where(compute_rewards_idx == i))]
                transition = ts.termination(self.states[i], reward)
                self.terminal_transitions[i] = transition
            else:
                transition = ts.transition(self.states[i], 0)

            transitions.append(transition)

        return nest_utils.stack_nested_arrays(transitions)


class DynaPPOEnvironmentMutative(py_environment.PyEnvironment):  # pylint: disable=W0223
    """
    DyNA-PPO environment based on TF-Agents.

    Note that unlike the other DynaPPO environment, this one is mutative rather than
    constructive.
    """

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        starting_seq: str,
        model: flexs.Model,
        landscape: flexs.Landscape,
        max_num_steps: int,
    ):
        """
        Initialize DyNA-PPO agent environment.

        Based on this tutorial:
        https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent

        Args:
            alphabet: Usually UCGA.
            starting_seq: When initializing the environment,
                the sequence which is initially mutated.
            model: Landscape or model which evaluates
                each sequence.
            max_num_steps: Maximum number of steps before
                episode is forced to terminate. Usually the
                `model_queries_per_batch`.

        """
        self.alphabet = alphabet

        # model/model/measurements
        self.model = model
        self.landscape = landscape
        self.fitness_model_is_gt = False
        self.previous_fitness = -float("inf")

        self.seq = starting_seq
        self._state = {
            "sequence": s_utils.string_to_one_hot(self.seq, self.alphabet).astype(
                np.float32
            ),
            "fitness": self.model.get_fitness([starting_seq]).astype(np.float32),
        }
        self.episode_seqs = set()  # the sequences seen in the current episode
        self.all_seqs = {}
        self.measured_sequences = {}

        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.integer,
            minimum=0,
            maximum=len(self.seq) * len(self.alphabet) - 1,
            name="action",
        )
        self._observation_spec = {
            "sequence": array_spec.BoundedArraySpec(
                shape=(len(self.seq), len(self.alphabet)),
                dtype=np.float32,
                minimum=0,
                maximum=1,
            ),
            "fitness": array_spec.BoundedArraySpec(
                shape=(1,), minimum=0, maximum=1, dtype=np.float32
            ),
        }

        self.num_steps = 0
        self.max_num_steps = max_num_steps

    def _reset(self):
        self.previous_fitness = -float("inf")
        self._state = {
            "sequence": s_utils.string_to_one_hot(self.seq, self.alphabet).astype(
                np.float32
            ),
            "fitness": self.model.get_fitness([self.seq]).astype(np.float32),
        }
        self.episode_seqs = set()
        self.num_steps = 0
        return ts.restart(self._state)

    def action_spec(self):
        """Define agent actions."""
        return self._action_spec

    def observation_spec(self):
        """Define environment observations."""
        return self._observation_spec

    def get_state_string(self):
        """Get sequence representing current state."""
        return s_utils.one_hot_to_string(self._state["sequence"], self.alphabet)

    def sequence_density(self, seq):
        """Get average distance to `seq` out of all observed sequences."""
        dens = 0
        dist_radius = 2
        exactly_eq = False
        for s in self.all_seqs:
            dist = int(editdistance.eval(s, seq))
            if dist <= dist_radius:
                dens += self.all_seqs[s] / (dist + 1)
            #elif dist == 0:
            #    exactly_eq = True
            #    break

        return dens if not exactly_eq else np.inf

    def set_fitness_model_to_gt(self, fitness_model_is_gt):
        """
        Set the fitness model to the ground truth landscape or to the model.

        Call with `True` when doing an experiment-based training round
        and call with `False` when doing a model-based training round.
        """
        self.fitness_model_is_gt = fitness_model_is_gt

    def _step(self, action):
        """Progress the agent one step in the environment.

        The agent moves until the reward is decreasing. The number of sequences that
        can be evaluated at each episode is capped to `self.max_num_steps`.
        """
        # if we've exceeded the maximum number of steps, terminate
        if self.num_steps >= self.max_num_steps:
            return ts.termination(self._state, 0)

        # `action` is an integer representing which residue to mutate to 1
        # along the flattened one-hot representation of the sequence
        pos = action // len(self.alphabet)
        res = action % len(self.alphabet)
        self.num_steps += 1

        # if we are trying to modify the sequence with a no-op, then stop
        if self._state["sequence"][pos, res] == 1:
            return ts.termination(self._state, 0)

        self._state["sequence"][pos] = 0
        self._state["sequence"][pos, res] = 1
        state_string = s_utils.one_hot_to_string(self._state["sequence"], self.alphabet)

        if self.fitness_model_is_gt:
            self._state["fitness"] = self.landscape.get_fitness([state_string]).astype(
                np.float32
            )
        else:
            self._state["fitness"] = self.model.get_fitness([state_string]).astype(
                np.float32
            )
        self.all_seqs[state_string] = self._state["fitness"].item()

        penalty = 0.
        if not np.isclose(0., self.lam):
            penalty = self.lam * self.sequence_density(state_string)

        reward = self._state["fitness"].item() - penalty

        # if we have seen the sequence this episode,
        # terminate episode and punish
        # (to prevent going in loops)
        if state_string in self.episode_seqs:
            return ts.termination(self._state, -1)
        self.episode_seqs.add(state_string)

        # if the reward is not increasing, then terminate
        if reward < self.previous_fitness:
            return ts.termination(self._state, reward=reward)

        self.previous_fitness = reward
        return ts.transition(self._state, reward=reward)
