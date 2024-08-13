"""Define a baseline multilayer perceptron model."""
import tensorflow as tf

import flexs
import flexs.utils.sequence_utils as s_utils
import numpy as np
from . import keras_model


class MLP(keras_model.KerasModel):
    """A baseline MLP with three dense layers and relu activations."""

    def __init__(
        self,
        seq_len,
        hidden_size,
        alphabet,
        loss="MSE",
        name=None,
        batch_size=256,
        num_hidden=3,
        epochs=20,
        lr=None,
    ):
        """Create an MLP."""
        seq_len = 60
        hidden_layers = [tf.keras.layers.Dense(
            hidden_size, input_shape=(seq_len, len(alphabet)), activation="relu"
        )]

        for _ in range(num_hidden - 1):
            hidden_layers.append(tf.keras.layers.Dense(hidden_size, activation="relu"))

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                *hidden_layers,
                tf.keras.layers.Dense(1),
            ]
        )

        optim = tf.keras.optimizers.Adam(learning_rate=lr) if lr else "adam"

        model.compile(loss=loss, optimizer=optim, metrics=["mse"])

        if name is None:
            name = f"MLP_hidden_size_{hidden_size}"

        super().__init__(
            model,
            alphabet=alphabet,
            name=name,
            batch_size=batch_size,
            epochs=epochs,
            seq_len=seq_len
        )

class MLPEnsemble(flexs.Model):
    """A baseline CNN ensemble model with 2 conv layers and 1 dense layer each."""
    def __init__(self, seq_len: int, alphabet: str, acq_class, loss="MSE"):
        self.seq_len = 60#seq_len
        self.alphabet = alphabet
        self.acq_class = acq_class
        self.loss = loss

        super().__init__(name='MLPEnsemble')

    def _create_ensemble(self, lr, ensemble_size=10):
        self.models = [
            MLP(self.seq_len, 512, self.alphabet, self.loss, num_hidden=2, lr=1e-4)
            for _ in range(ensemble_size)
        ]

    def _seqs_to_tensor(self, seqs):
        one_hots = np.array([
            s_utils.string_to_one_hot(seq, self.alphabet, self.seq_len)
            for seq in seqs
        ])

        return tf.convert_to_tensor(one_hots, dtype=tf.float32)

    def _train(self, sequences, labels, **hparam_kwargs):
        epochs, lr = 50, 1e-4
        if 'epochs' in hparam_kwargs and 'lr' in hparam_kwargs:
            epochs, lr = hparam_kwargs['epochs'], hparam_kwargs['lr']
        print('Best epochs: %d, best lr: %f' % (epochs, lr))

        self._create_ensemble(lr)

        for model in self.models:
            model.train(sequences, labels)

        self.acq = self.acq_class(self, sequences)

    def _compute_preds(self, sequences):
        return np.array([model.get_fitness(sequences) for model in self.models])

    def _fitness_function(self, sequences):
        return self._compute_preds(sequences).mean(axis=0)

    def _fitness_function_uncert(self, sequences):
        preds = self._compute_preds(sequences)
        return preds.mean(axis=0), preds.std(axis=0)

    def _fitness_function_acq(self, sequences):
        mean, std = self._fitness_function_uncert(sequences)
        return self.acq(mean, std)
