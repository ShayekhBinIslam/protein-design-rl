"""Defines base Model class."""
import abc
from typing import Any, List
from sklearn.model_selection import KFold
from collections import OrderedDict
from itertools import product

import numpy as np

import flexs
from flexs.types import SEQUENCES_TYPE


class Model(flexs.Landscape, abc.ABC):
    """
    Base model class. Inherits from `flexs.Landscape` and adds an additional
    `train` method.

    """
    def __init__(self, name, hparam_tune=False, hparams_to_search={}, nfolds=None):
        super().__init__(name)
        self.hparam_tune = hparam_tune
        self.hparams_to_search = OrderedDict(hparams_to_search)
        self.nfolds = nfolds

    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        if not self.hparam_tune:
            return self._train(sequences, labels)

        best_conf, best_score = None, None
        kfold = KFold(n_splits=self.nfolds, shuffle=True)
        for hparam_setting in product(*self.hparams_to_search.values()):
            hparam_keys = self.hparams_to_search.keys()
            hparam_kwargs = {
                key: hparam_setting[i]
                for i, key in enumerate(hparam_keys)
            }
            ### sequences, labels originally list
            ### Converted to np arrays
            r_squareds = np.array([
                self._train_hparam_setting(
                    np.array(sequences)[train_idx],
                    np.array(labels)[train_idx],
                    np.array(sequences)[val_idx],
                    np.array(labels)[val_idx],
                    # sequences[train_idx],
                    # labels[train_idx],
                    # sequences[val_idx],
                    # labels[val_idx],
                    **hparam_kwargs
                )
                for train_idx, val_idx in kfold.split(sequences)
            ])

            mean_r_squared = r_squareds.mean()
            if best_score is None or mean_r_squared >= best_score:
                best_score = mean_r_squared
                best_conf = hparam_kwargs

        print('Had best R squared of %f for %s' % (best_score, self.name))
        self._train(sequences, labels, **best_conf)
        return best_score

    @abc.abstractmethod
    def _train(self, sequences: SEQUENCES_TYPE, labels: List[Any], **hparam_kwargs):
        """
        Train model.

        This function is called whenever you would want your model to update itself
        based on the set of sequences it has measurements for.

        """
        pass

    def _train_hparam_setting(self, seq_train, label_train, seq_val, label_val, **hparams):
        pass


class LandscapeAsModel(Model):
    """
    This simple class wraps a `flexs.Landscape` in a `flexs.Model` to allow running
    experiments against a perfect model.

    This class's `_fitness_function` simply calls the landscape's `_fitness_function`.
    """

    def __init__(self, landscape: flexs.Landscape):
        """
        Create a `flexs.Model` out of a `flexs.Landscape`.

        Args:
            landscape: Landscape to wrap in a model.

        """
        super().__init__(f"LandscapeAsModel={landscape.name}")
        self.landscape = landscape

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return self.landscape._fitness_function(sequences)

    def _train(self, sequences: SEQUENCES_TYPE, labels: List[Any], **hparam_kwargs):
        """No-op."""
        pass
