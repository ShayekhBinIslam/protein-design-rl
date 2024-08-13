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

from language import (ConstantSequenceSegment,
    FixedLengthSequenceSegment,
    VariableLengthSequenceSegment,
)

from copy import deepcopy

class DynaPPOESMEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """DyNA-PPO environment based on TF-Agents."""

    def __init__(  # pylint: disable=W0231
        self,
        alphabet: str,
        starting_program,
        model: flexs.Model,
        landscape: flexs.Landscape,
        batch_size: int,
        max_num_steps: int = 100,
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
        self._batch_size = 1
        self.lam = penalty_scale
        self.dist_radius = distance_radius

        self.num_steps = 0
        self.max_num_steps = max_num_steps

        self.starting_program = starting_program
        # Index 0 is the binder program, we get the segments from it
        regions = self.starting_program.children[0].children
        region_states = []
        num_actions = 0
        self.region_action_boundaries = [num_actions]
        self.action_to_state_map = {}
        pos_offset = 0

        
        self.variable_segs = []
        for region in regions:
            segment = region.sequence_segment
            if isinstance(segment, ConstantSequenceSegment) \
                    or isinstance(segment, FixedLengthSequenceSegment):
                region_len = len(segment.get())
                region_state = np.zeros((region_len, len(alphabet) + 1))
                for i in range(region_len):
                    region_state[i, alphabet.index(segment.get()[i])] = 1
                
                if isinstance(segment, FixedLengthSequenceSegment):
                    num_actions_prev = num_actions
                    num_actions += region_len * len(alphabet) # No Del/Insertion

                    for action in range(num_actions_prev, num_actions):
                        rel_action = action - num_actions_prev
                        pos = pos_offset + (rel_action // len(self.alphabet))
                        res = rel_action % len(self.alphabet)
                        self.action_to_state_map[action] = (pos, res)
                        # print("Action: {}, pos: {}, res: {}".format(action, pos, res))


                
            elif isinstance(segment, VariableLengthSequenceSegment):
                variable_seg = {
                    'left': pos_offset, 
                    'right': pos_offset + segment.max_length, 
                    'none_allowed': segment.max_length - segment.min_length,
                }
                self.variable_segs.append(variable_seg)
                state_len = segment.max_length
                region_len = len(segment.get())
                region_state = np.zeros((state_len, len(alphabet) + 1))
                for idx in range(state_len):
                    if idx < region_len:
                        region_state[idx, alphabet.index(segment.get()[idx])] = 1
                    else:
                        region_state[idx, -1] = 1 # None
                
                num_actions_prev = num_actions
                num_actions += state_len * (len(alphabet) + 1) # Del/Insertion
                # TODO: Punish min, max len violating actions
                for action in range(num_actions_prev, num_actions):
                    # import ipdb; ipdb.set_trace()
                    rel_action = action - num_actions_prev
                    pos = pos_offset + (rel_action // (len(self.alphabet) + 1))
                    res = rel_action % (len(self.alphabet) + 1)
                    self.action_to_state_map[action] = (pos, res)
                    # print("Action: {}, pos: {}, res: {}".format(action, pos, res))
            
            # print(type(segment), segment.get(), '\n', region_state)
            region_states.append(region_state)
            pos_offset += region_state.shape[0]
        
        seq_state = np.vstack(region_states)
        # seq_state = np.tile(seq_state[np.newaxis, ...], (batch_size, 1, 1))
        # self.states[:, np.arange(self.seq_length), -1] = 1
        
        self.starting_state = seq_state.astype("float32")
        self.states = deepcopy(self.starting_state)
        assert int(self.states.sum()) == np.prod(self.states.shape[:-1]), "Wrong state one-hot encoding"
        self.seq_max_len = self.states.shape[0]

        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)

        # model/model/measurements
        self.model = model
        self.landscape = landscape
        self.fitness_model_is_gt = False
        self.previous_fitness = -float("inf")
        self.episode_seqs = set()

        # sequence
        self.all_seqs = {}
        self.all_seqs_uncert = {}
        self.lam = 0.1

        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.integer,
            minimum=0,
            maximum=num_actions - 1,
            name="action",
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.states.shape[0], self.states.shape[1]), # Length X alphabet+1
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._time_step_spec = ts.time_step_spec(self._observation_spec)

        super().__init__()
    
    @classmethod
    def get_seq_len(cls, program):
        regions = program.children[0].children
        seq_len = 0
        for region in regions:
            segment = region.sequence_segment
            if isinstance(segment, ConstantSequenceSegment) \
                    or isinstance(segment, FixedLengthSequenceSegment):
                seq_len += len(segment.get())
            elif isinstance(segment, VariableLengthSequenceSegment):
                seq_len += segment.max_length

        return seq_len

    def _reset(self):
        self.states = deepcopy(self.starting_state)
        self.num_steps = 0
        self.previous_fitness = -float("inf")
        self.episode_seqs = set()
        
        return ts.restart(self.states)

    # @property
    # def batched(self):
    #     """Tf-agents function that says that this env returns batches of timesteps."""
    #     return True

    # @property
    # def batch_size(self):
    #     """Tf-agents property that return env batch size."""
    #     return self._batch_size

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
        # print("***** get_cached_uncertainty", seq)
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
            # fitnesses, uncerts = self.model.get_fitness(seqs, compute_uncert=True)
            fitnesses = self.model.get_fitness(seqs, compute_uncert=False)

        # Reward = fitness - lambda * sequence density
        penalty = np.zeros(len(seqs))
        if not np.isclose(0., self.lam):
            penalty = np.array([
                self.lam * self.sequence_density(seq)
                for seq in seqs
            ])

        self.all_seqs.update(zip(seqs, fitnesses))
        # if not self.fitness_model_is_gt:
        #     # print("*********_compute_rewards_non_empty", seqs, uncerts)
        #     self.all_seqs_uncert.update(zip(seqs, uncerts))

        rewards = fitnesses - penalty
        return rewards

    def _step(self, action):
        """Progress the agent one step in the environment."""
        # if we've exceeded the maximum number of steps, terminate
        if self.num_steps >= self.max_num_steps:
            # print("Termination string v1:", s_utils.one_hot_to_string(self.states, self.alphabet))
            return ts.termination(self.states, 0)
        
        pos, res = self.action_to_state_map[int(action)]

        if self.states[pos, res] == 1:
            # print("Termination string v2:", s_utils.one_hot_to_string(self.states, self.alphabet))
            if self.num_steps == 0:
                self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)
                reward = self._compute_rewards_non_empty([self.seq])[0]
                return ts.termination(self.states, reward)

            return ts.termination(self.states, 0)
    
        self.states[pos, :] = 0
        self.states[pos, res] = 1

        # assert int(self.states.sum()) == np.prod(self.states.shape[:-1]), "Wrong state one-hot encoding"

        self.num_steps += 1        

        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)

        reward = self._compute_rewards_non_empty([self.seq])[0]

        # if we have seen the sequence this episode,
        # terminate episode and punish
        # (to prevent going in loops)
        bad_seq = False
        for variable_seg in self.variable_segs:
            if self.states[variable_seg['left']:variable_seg['right'], -1].sum() > variable_seg['none_allowed']:
                bad_seq = True
                print("*** Bad sequence")
                break

        if (self.seq in self.episode_seqs) or bad_seq:
            return ts.termination(self.states, -1)
        self.episode_seqs.add(self.seq)

        # if the reward is not increasing, then terminate
        if reward < self.previous_fitness:
            return ts.termination(self.states, reward=reward)

        self.previous_fitness = reward

        return ts.transition(self.states, reward=reward)
