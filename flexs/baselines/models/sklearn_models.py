"""Define scikit-learn model wrappers as well a few convenient pre-wrapped models."""
import abc

import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics

import flexs
from flexs.utils import sequence_utils as s_utils


class SklearnModel(flexs.Model, abc.ABC):
    """Base sklearn model wrapper."""

    def __init__(
        self,
        model_type,
        alphabet,
        name,
        seq_len,
        hparam_tune=False,
        hparams_to_search={},
        nfolds=None
    ):
        """
        Args:
            model: sklearn model to wrap.
            alphabet: Alphabet string.
            name: Human-readable short model descriptipon (for logging).

        """
        super().__init__(name, hparam_tune, hparams_to_search, nfolds)

        self.model_type = model_type
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.hparam_tune = hparam_tune

        self.model = None

    def _train_hparam_setting(
        self,
        train_seq,
        train_label,
        val_seq,
        val_labels,
        **hparam_kwargs
    ):
        self._train(train_seq, train_label, **hparam_kwargs)
        val_preds = self._fitness_function(val_seq)

        return sklearn.metrics.r2_score(val_labels, val_preds)

    def _seqs_to_one_hot(self, seqs):
        one_hots = np.array([
            s_utils.string_to_one_hot(seq, self.alphabet, self.seq_len)
            for seq in seqs
        ])

        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        return flattened

    def _train(self, sequences, labels, **hparam_kwargs):
        """Flatten one-hot sequences and train model using `model.fit`."""
        self.model = self.model_type(**hparam_kwargs)

        one_hots = self._seqs_to_one_hot(sequences)
        self.model.fit(one_hots, labels)


class SklearnRegressor(SklearnModel, abc.ABC):
    """Class for sklearn regressors (uses `model.predict`)."""

    def _fitness_function(self, sequences):
        
        one_hots = np.array([
            s_utils.string_to_one_hot(seq, self.alphabet, self.seq_len)
            for seq in sequences
        ])
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        # import pdb; pdb.set_trace()

        return self.model.predict(flattened)


class SklearnClassifier(SklearnModel, abc.ABC):
    """Class for sklearn classifiers (uses `model.predict_proba(...)[:, 1]`)."""

    def _fitness_function(self, sequences):
        one_hots = np.array([
            s_utils.string_to_one_hot(seq, self.alphabet, self.seq_len)
            for seq in sequences
        ])
        flattened = one_hots.reshape(
            one_hots.shape[0], one_hots.shape[1] * one_hots.shape[2]
        )

        return self.model.predict_proba(flattened)[:, 1]


class BayesianRidge(SklearnRegressor):
    """Sklearn linear regression."""

    def __init__(
        self,
        alphabet,
        seq_len,
        hparam_tune=False,
        hparams_to_search={},
        nfolds=None
    ):
        """Create linear regression model."""
        model = sklearn.linear_model.BayesianRidge
        self.could_not_fit = False
        super().__init__(
            model,
            alphabet,
            "bayesian_ridge",
            seq_len,
            hparam_tune,
            hparams_to_search,
            nfolds
        )

    def _train_hparam_setting(
        self,
        train_seq,
        train_label,
        val_seq,
        val_labels,
        **hparam_kwargs
    ):
        try:
            return super()._train_hparam_setting(
                train_seq,
                train_label,
                val_seq,
                val_labels,
                **hparam_kwargs
            )
        except:
            self.could_not_fit = True
            return sklearn.metrics.r2_score(val_labels, np.zeros(len(val_labels)))

    def _train(self, sequences, labels, **hparam_kwargs):
        try:
            super()._train(sequences, labels, **hparam_kwargs)
        except:
            self.cound_not_fit = True

    def _fitness_function(self, sequences):
        if not self.could_not_fit:
            return super()._fitness_function(sequences)
        else:
            return np.zeros(len(sequences))


class LinearRegression(SklearnRegressor):
    """Sklearn linear regression."""

    def __init__(
        self,
        alphabet,
        seq_len,
        hparam_tune=False,
        hparams_to_search={},
        nfolds=None
    ):
        """Create linear regression model."""
        model = sklearn.linear_model.LinearRegression
        super().__init__(
            model,
            alphabet,
            "linear_regression",
            seq_len,
            hparam_tune,
            hparams_to_search,
            nfolds
        )


class LogisticRegression(SklearnRegressor):
    """Sklearn logistic regression."""

    def __init__(
        self,
        alphabet,
        seq_len,
        hparam_tune=False,
        hparams_to_search={},
        nfolds=None
    ):
        """Create logistic regression model."""
        model = sklearn.linear_model.LogisticRegression
        super().__init__(
            model,
            alphabet,
            "logistic_regression",
            seq_len,
            hparam_tune,
            hparams_to_search,
            nfolds
        )


class RandomForest(SklearnRegressor):
    """Sklearn random forest regressor."""

    def __init__(
        self,
        alphabet,
        seq_len,
        hparam_tune=False,
        hparams_to_search={},
        nfolds=None
    ):
        """Create random forest regressor."""
        model = sklearn.ensemble.RandomForestRegressor
        super().__init__(
            model,
            alphabet,
            "random_forest",
            seq_len,
            hparam_tune,
            hparams_to_search,
            nfolds
        )
