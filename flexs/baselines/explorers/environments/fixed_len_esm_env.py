import editdistance

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from copy import deepcopy
import numpy as np
import flexs
from flexs.utils import sequence_utils as s_utils

from language.program import ProgramNode
from language import FixedLengthSequenceSegment

class FixedLengthESMEnvironment(py_environment.PyEnvironment):  # pylint: disable=W0223
    """Environment based on TF-Agents compatible with ESM proxies"""

    def __init__(
        self, 
        alphabet: str, 
        starting_program: ProgramNode,
        model: flexs.Model,
        landscape: flexs.Landscape,
        batch_size: int = 1,
        max_num_steps: int = 100,
        fitness_model_is_gt = False,
        unconditional=False
    ):

        self.alphabet = alphabet
        self.starting_program = starting_program
        self._batch_size = batch_size
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

        self.regions = [self.starting_program]
        self._init_starting_state(self.regions)
        self._set_env_properties()

    def _init_starting_state(self, regions):
        region_states = []

        assert len(regions) == 1
        segment = regions[0].sequence_segment
        region_len = len(segment.get())
        assert isinstance(segment, FixedLengthSequenceSegment)

        region_state = np.zeros((region_len, len(self.alphabet) + 1))
        for i in range(region_len):
            region_state[i, self.alphabet.index(segment.get()[i])] = 1

        num_actions = region_len * len(self.alphabet)        
        self.num_actions = num_actions
        for action in range(0, num_actions):
            rel_action = action
            pos = (rel_action // len(self.alphabet))
            res = rel_action % len(self.alphabet)
            self.action_to_state_map[action] = (pos, res)

        region_states.append(region_state)
        self.starting_state = np.vstack(region_states).astype("float32")
        self.states = deepcopy(self.starting_state)
        self.seq_max_len = self.states.shape[0]
        assert int(self.states.sum()) == np.prod(self.states.shape[:-1]), "Wrong state one-hot encoding"
        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)

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
        # self.states = deepcopy(self.starting_state)
        # self.seq = s_utils.one_hot_to_string(self.states, self.alphabet)
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

    def compute_rewards(self, seq: str):
        if self.fitness_model_is_gt:
            fitnesses = self.landscape.get_fitness([seq])[0]
        else:
            # fitnesses, uncerts = self.model.get_fitness(seqs, compute_uncert=True)
            fitnesses = self.model.get_fitness([seq], compute_uncert=False)[0]

        rewards = fitnesses 
        return rewards

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # if we've exceeded the maximum number of steps, terminate
        if self.num_steps >= self.max_num_steps:
            self._episode_ended = True 
            print(f"STEPS EXCEEDED: Terminated after {self.num_steps} steps")
            return ts.termination(self.states, 0)

        pos, res = self.action_to_state_map[int(action)]

        if self.states[pos, res] == 1: 
            print(f"Same Action | ({pos}, {res})")
            self._episode_ended = True 
            print(f"Already seen sequence. Terminated after {self.num_steps} steps")
            return ts.termination(self.states, -1) # TODO: needs explanation. Why terminate here with 0 reward?

        # Modify state according to the action
        self.states[pos, :] = 0
        self.states[pos, res] = 1
        
        # get new sequence and corresponding reward
        self.seq = s_utils.one_hot_to_string(self.states, self.alphabet) 
        reward = self.compute_rewards(self.seq)
        
        # if we have seen the sequence this episode or `bad_seq`, terminate episode and punish (to prevent going in loops)
        # if (self.seq in self.episode_seqs):
        #     self._episode_ended = True 
        #     print(f"Already seen sequence. Terminated after {self.num_steps} steps")
        #     return ts.termination(self.states, -1)

        self.episode_seqs.add(self.seq)

        # if the reward is not increasing, then terminate
        # if reward < self.previous_fitness:
        #     self._episode_ended = True
        #     print(f"Reward is not increasing. Terminated after {self.num_steps} steps")
        #     return ts.termination(self.states, reward=reward)

        self.previous_fitness = reward
        self.num_steps += 1
        return ts.transition(self.states, reward=reward)