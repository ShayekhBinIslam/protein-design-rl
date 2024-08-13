"""Define TFBinding landscape and problem registry."""
import os
from typing import Dict

import numpy as np
import pandas as pd

import design_bench
import flexs
import flexs.utils.sequence_utils as s_utils
from design_bench.datasets.discrete.tf_bind_8_dataset import TF_BIND_8_FILES
from flexs.types import SEQUENCES_TYPE

class TFBinding(flexs.Landscape):
    """
    A landscape of binding affinity of proposed 8-mer DNA sequences to a
    particular transcription factor.

    We use experimental data from Barrera et al. (2016), a survey of the binding
    affinity of more than one hundred and fifty transcription factors (TF) to all
    possible DNA sequences of length 8.
    """

    _TRANSCRIPTION_FACTORS = None

    @staticmethod
    def get_transcription_factors():
        if TFBinding._TRANSCRIPTION_FACTORS is None:
            tf_bind_8_str = 'tf_bind_8-'

            TFBinding._TRANSCRIPTION_FACTORS = list(map(
                lambda x: x.split('/')[0][len(tf_bind_8_str):],
                design_bench.datasets.discrete.tf_bind_8_dataset.TF_BIND_8_FILES
            ))

        return TFBinding._TRANSCRIPTION_FACTORS

    def get_dataset(self):
        return self

    def get_full_dataset(self):
        return self.x, self.y

    @property
    def length(self):
        return len(self.x[0])

    @property
    def vocab_size(self):
        return len(s_utils.DNAA)

    @property
    def is_dynamic_length(self):
        return False

    def __init__(self, transcription_factor: str = None):
        """
        Create a TFBinding landscape from experimental data .csv file.

        See https://github.com/samsinai/FLSD-Sandbox/tree/stewy-redesign/flexs/landscapes/data/tf_binding  # noqa: E501
        for examples.
        """
        super().__init__(name="TF_Binding")

        dataset_kwargs = None
        if transcription_factor:
            dataset_kwargs = {'transcription_factor': transcription_factor}

        self.task = design_bench.make(
            'TFBind8-Exact-v0',
            dataset_kwargs=dataset_kwargs
        )

        self.char_map = {
            char: idx
            for idx, char in enumerate(s_utils.DNAA)
        }

        self.inv_char_map = {
            idx: char
            for char, idx in self.char_map.items()
        }

        seq_x = self._cat_encode_to_seq(self.task.x)
        self.x, self.y = seq_x, self.task.y.flatten()

    def _cat_encode_to_seq(self, x):
        return np.array([
            ''.join([self.inv_char_map[val] for val in seq_cats])
            for seq_cats in x
        ])

    def add(self, batch):
        samples, scores = batch

        self.x = np.concatenate((self.x, samples), axis=0)
        self.y = np.concatenate((self.y, scores), axis=0)

    def _encode(self, sequences):
        return np.array([
            [self.char_map[char] for char in seq]
            for seq in sequences
        ])

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return self.task.predict(self._encode(sequences)).flatten()

#class TFBinding(flexs.Landscape):
#    """
#    A landscape of binding affinity of proposed 8-mer DNA sequences to a
#    particular transcription factor.
#
#    We use experimental data from Barrera et al. (2016), a survey of the binding
#    affinity of more than one hundred and fifty transcription factors (TF) to all
#    possible DNA sequences of length 8.
#    """
#
#    def __init__(self, landscape_file: str):
#        """
#        Create a TFBinding landscape from experimental data .csv file.
#
#        See https://github.com/samsinai/FLSD-Sandbox/tree/stewy-redesign/flexs/landscapes/data/tf_binding  # noqa: E501
#        for examples.
#        """
#        super().__init__(name="TF_Binding")
#
#        # Load TF pairwise TF binding measurements from file
#        data = pd.read_csv(landscape_file, sep="\t")
#        score = data["E-score"]  # "E-score" is enrichment score
#        norm_score = (score - score.min()) / (score.max() - score.min())
#
#        # The csv file keeps one DNA strand's sequence in "8-mer" and the other in
#        # "8-mer.1".
#        # Since it doesn't really matter which strand we have, we will map the sequences
#        # of both strands to the same normalized enrichment score.
#        self.sequences = dict(zip(data["8-mer"], norm_score))
#        self.sequences.update(zip(data["8-mer.1"], norm_score))
#
#    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
#        return np.array([self.sequences[seq] for seq in sequences])
#
#
#def registry() -> Dict[str, Dict]:
#    """
#    Return a dictionary of problems of the form:
#
#    ```python
#    {
#        "problem name": {
#            "params": ...,
#        },
#        ...
#    }
#    ```
#
#    where `flexs.landscapes.TFBinding(**problem["params"])` instantiates the
#    transcription factor binding landscape for the given set of parameters.
#
#    Returns:
#        Problems in the registry.
#
#    """
#    tf_binding_data_dir = os.path.join(os.path.dirname(__file__), "data/tf_binding")
#
#    problems = {}
#    for fname in os.listdir(tf_binding_data_dir):
#        problem_name = fname.replace("_8mers.txt", "")
#
#        problems[problem_name] = {
#            "params": {"landscape_file": os.path.join(tf_binding_data_dir, fname)},
#            "starts": [
#                "GCTCGAGC",
#                "GCGCGCGC",
#                "TGCGCGCC",
#                "ATATAGCC",
#                "GTTTGGTA",
#                "ATTATGTT",
#                "CAGTTTTT",
#                "AAAAATTT",
#                "AAAAACGC",
#                "GTTGTTTT",
#                "TGCTTTTT",
#                "AAAGATAG",
#                "CCTTCTTT",
#                "AAAGAGAG",
#            ],
#        }
#
#    return problems
