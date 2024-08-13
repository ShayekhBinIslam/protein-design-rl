#"""Defines the AMP landscape."""
import numpy as np
import flexs
import flexs.utils.sequence_utils as s_utils
from collections import namedtuple
from clamp_common_eval.defaults import get_test_oracle
from gflownet_generator.lib.dataset.classification import (
    BinaryClassificationDataset as AMPBinaryClassificationDataset
)


class AMPLandscape(flexs.Landscape):
    r"""
    AMP generation landscape.

    The oracle used in this lanscape is the transformer model
    from clamp-gen/common-evaluation.

    Attributes:
        gfp_wt_sequence (str): Wild-type sequence for jellyfish
            green fluorescence protein.
        starts (dict): A dictionary of starting sequences at different edit distances
            from wild-type with different difficulties of optimization.

    """

    starts = [
        'WGWWTIVTGIRKYMNVDAHH',
        'ATCYFRTGRAAQYESLYGVAEMSCGLYRLAYR',
        'KFAKKRQKQNAQKFEKRFAKKNAP',
        'WDGARQKDES',
        'QLRRMWKWRCIAWD'
    ]

    def __init__(
        self,
        oracle_split,
        oracle_type,
        oracle_features,
        medoid_oracle_norm,
        proxy_data_split,
        num_folds,
        device,
        batch_size=256
    ):
        """
        Create AMP landscape.
        """
        super().__init__(name="AMP")

        self.oracle = get_test_oracle(oracle_split,
                                      model=oracle_type,
                                      feature=oracle_features,
                                      dist_fn="edit",
                                      norm_constant=medoid_oracle_norm)

        self.oracle.to(device)

        self.batch_size = batch_size

        Args = namedtuple('Args', ['save_scores_path'])
        args = Args('/home/mila/j/jarrid.rector-brooks/repos/gflownet-generator/gflownet_generator/cached_scores')
        self.dataset = AMPBinaryClassificationDataset(
            proxy_data_split,
            num_folds,
            args,
            self.oracle
        )

    @property
    def length(self):
        return 50

    @property
    def vocab_size(self):
        return len(s_utils.AAS)

    @property
    def is_dynamic_length(self):
        return True

    def get_dataset(self):
        return self.dataset

    def get_full_dataset(self):
        return self.dataset.get_full_dataset()

    def _fitness_function(self, sequences):
        sequences = np.array(sequences)
        scores = []

        for i in range(int(np.ceil(len(sequences) / self.batch_size))):
            batch = sequences[i * self.batch_size : (i + 1) * self.batch_size]
            s = self.oracle.evaluate_many(batch)

            if type(s) == dict:
                scores += s["confidence"][:, 1].tolist()
            else:
                scores += s.tolist()

        return np.float32(scores)

class DummySeqLenRewardingLandscape(AMPLandscape):
    def _fitness_function(self, sequences):
        scores = [len(seq) for seq in sequences]
        return np.float32(scores)

