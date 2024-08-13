#"""Defines the AMP landscape."""
import numpy as np
import pandas as pd
import flexs
from collections import namedtuple
from active_learning_pipeline.oracles import nupackScore
from gflownet_generator.lib.dataset.classification import (
    BinaryClassificationDataset as AMPBinaryClassificationDataset
)

_DATASET_CSV_FNAME = '~/repos/ActiveLearningPipeline/data/energies_al_20_60_n200.csv'

_ALPHABET_ORIG = {0: "A", 1: "T", 2: "C", 3: "G"}
_ALPHABET_INV_ORACLE = {v: k + 1 for k, v in _ALPHABET_ORIG.items()}

class DNAAptamerLandscape(flexs.Landscape):
    r"""
    DNA aptamer generation landscape.

    The oracle used in this lanscape is nupack.

    Attributes:
        gfp_wt_sequence (str): Wild-type sequence for jellyfish
            green fluorescence protein.
        starts (dict): A dictionary of starting sequences at different edit distances
            from wild-type with different difficulties of optimization.

    """

    MAX_SEQ_LEN = 60
    MIN_SEQ_LEN = 20

    def __init__(self):
        """
        Create AMP landscape.
        """
        super().__init__(name="DNA Aptamer")

        dataset_df = pd.read_csv(_DATASET_CSV_FNAME)
        self.x = dataset_df['letters'].to_numpy()
        self.y = -dataset_df['scores'].to_numpy()

    def get_dataset(self):
        return self

    def get_full_dataset(self):
        return self.x, self.y

    def add(self, batch):
        samples, scores = batch

        self.x = np.concatenate((self.x, samples), axis=0)
        self.y = np.concatenate((self.y, scores), axis=0)

    def _encode(self, sequences):
        horizon = np.max([len(seq) for seq in sequences])
        encoding = np.zeros((len(sequences), horizon), dtype=np.int)

        for i, seq in enumerate(sequences):
            encoding[i, :len(seq)] = [_ALPHABET_INV_ORACLE[char] for char in seq]

        return encoding

    def _fitness_function(self, sequences):
        return -nupackScore(self._encode(sequences), returnFunc="energy")
