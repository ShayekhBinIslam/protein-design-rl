import editdistance

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from copy import deepcopy
import numpy as np
import flexs
from flexs.utils import sequence_utils as s_utils

from language.program import ProgramNode
from language import ConstantSequenceSegment, FixedLengthSequenceSegment, VariableLengthSequenceSegment

class ESMEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """Environment based on TF-Agents compatible with ESM proxies"""

    def __init__(
        self, 
        alphabet: str, 
        starting_program: ProgramNode,
        model: flexs.Model,
        landscape: flexs.Landscape,
        batch_size: int = 1,
        max_num_steps: int = 100,
        penalty_scale = 0.1,
        distance_radius = 2,
        fitness_model_is_gt = False,
        unconditional=False
    ):

        self.alphabet = alphabet
        self.starting_program = starting_program
        self._batch_size = batch_size
        self.lam = penalty_scale
        self.dist_radius = distance_radius
        self.max_num_steps = max_num_steps
        self.landscape = landscape
        self.unconditional = unconditional

        self.num_steps = 0
        self.action_to_state_map = {}
        self.all_seqs = {}
        self.variable_segs = []
        self.previous_fitness = -float("inf")

        self.fitness_model_is_gt = fitness_model_is_gt
        self.model = model

        if self.unconditional:
            assert hasattr(self.starting_program, 'children') is False or self.starting_program.children is None
            self.regions = [self.starting_program]
        else:
            # Index 0 is the binder program, we get the segments from it
            self.regions = self.starting_program.children[0].children

        self._init_starting_state(self.regions)
        self._set_env_properties()

    @classmethod
    def get_seq_len(cls, program):
        seq_len = 0
        if program.children is None:
            segment = program.sequence_segment
            assert isinstance(segment, FixedLengthSequenceSegment)
            seq_len += len(segment.get())
        else:
            regions = program.children[0].children
            for region in regions:
                segment = region.sequence_segment
                if isinstance(segment, ConstantSequenceSegment) \
                        or isinstance(segment, FixedLengthSequenceSegment):
                    seq_len += len(segment.get())
                elif isinstance(segment, VariableLengthSequenceSegment):
                    seq_len += segment.max_length

        return seq_len

    def sequence_density(self, seq):
        """Get average distance to `seq` out of all observed sequences."""
        dens = 0
        for s in self.all_seqs:
            dist = int(editdistance.eval(s, seq))
            if dist <= self.dist_radius:
                dens += self.all_seqs[s] / (dist + 1)

        return dens

    def _init_starting_state(self, regions):
        pos_offset, num_actions = 0, 0
        region_states = []

        for region in regions:
            segment = region.sequence_segment
            region_len = len(segment.get())

            if isinstance(segment, ConstantSequenceSegment) or isinstance(segment, FixedLengthSequenceSegment):
                region_state = np.zeros((region_len, len(self.alphabet) + 1))
                for i in range(region_len):
                    region_state[i, self.alphabet.index(segment.get()[i])] = 1
                
                if isinstance(segment, FixedLengthSequenceSegment):
                    num_actions_prev = num_actions
                    num_actions += region_len * len(self.alphabet) # No Del/Insertion

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
                region_state = np.zeros((state_len, len(self.alphabet) + 1))
                for idx in range(state_len):
                    if idx < region_len:
                        region_state[idx, self.alphabet.index(segment.get()[idx])] = 1
                    else:
                        region_state[idx, -1] = 1 # None
                
                num_actions_prev = num_actions
                num_actions += state_len * (len(self.alphabet) + 1) # Del/Insertion

                # TODO: Punish min, max len violating actions
                for action in range(num_actions_prev, num_actions):
                    rel_action = action - num_actions_prev
                    pos = pos_offset + (rel_action // (len(self.alphabet) + 1))
                    res = rel_action % (len(self.alphabet) + 1)
                    self.action_to_state_map[action] = (pos, res)
                    # print("Action: {}, pos: {}, res: {}".format(action, pos, res))

            region_states.append(region_state)
            pos_offset += region_state.shape[0]

        self.starting_state = np.vstack(region_states).astype("float32")
        self.states = deepcopy(self.starting_state)
        self.seq_max_len = self.states.shape[0]
        assert int(self.states.sum()) == np.prod(self.states.shape[:-1]), "Wrong state one-hot encoding"
        
        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)
        self.num_actions = num_actions

    def _set_env_properties(self):
        # tf_agents environment
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.integer,
            minimum=0,
            maximum=self.num_actions - 1,
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

    def _reset(self):
        self._episode_ended = False
        self.states = deepcopy(self.starting_state)
        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)
        self.num_steps = 0
        self.previous_fitness = -float("inf")
        self.episode_seqs = set()
        return ts.restart(self.starting_state)
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def sample_random_action(self, size=None):
        return np.random.randint(self.num_actions, size=size)

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

        rewards = fitnesses 
        return rewards

    def _step(self, action):
        
        if self._episode_ended:
            return self.reset()

        # if we've exceeded the maximum number of steps, terminate
        if self.num_steps >= self.max_num_steps:
            self._episode_ended = True 
            print("STEPS EXCEEDED")
            return ts.termination(self.states, 0)

        pos, res = self.action_to_state_map[int(action)]

        if self.states[pos, res] == 1: 
            print(f"Same Action | ({pos}, {res})")
            self._episode_ended = True 
            self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)
            reward = self._compute_rewards_non_empty([self.seq])[0]
            return ts.termination(self.states, 0) # TODO: needs explanation. Why terminate here with 0 reward?

        # Modify state according to the action
        old_seq = s_utils.one_hot_to_string(self.states, self.alphabet)
        self.states[pos, :] = 0
        self.states[pos, res] = 1
        self.num_steps += 1
        new_seq = s_utils.one_hot_to_string(self.states, self.alphabet)

        # get new sequence and corresponding reward
        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet) 
        reward = self._compute_rewards_non_empty([self.seq])[0]
        
        bad_seq = False
        for variable_seg in self.variable_segs:
            # If trying to make sequence length longer than (max_length - min_length)
            if self.states[variable_seg['left']:variable_seg['right'], -1].sum() > variable_seg['none_allowed']: 
                bad_seq = True
                print("*** Bad sequence")
                break

        # if we have seen the sequence this episode or `bad_seq`, terminate episode and punish (to prevent going in loops)
        if (self.seq in self.episode_seqs) or bad_seq:
            self._episode_ended = True 
            return ts.termination(self.states, -1)

        self.episode_seqs.add(self.seq)

        # if the reward is not increasing, then terminate
        if reward < self.previous_fitness:
            self._episode_ended = True
            return ts.termination(self.states, reward=reward)

        self.previous_fitness = reward

        return ts.transition(self.states, reward=reward)