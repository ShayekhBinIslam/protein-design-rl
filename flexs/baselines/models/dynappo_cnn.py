"""Define a baseline CNN Model."""
import numpy as np
import tensorflow as tf

import flexs
import sklearn.metrics
from flexs.utils import sequence_utils as s_utils
from . import keras_model
from tqdm import tqdm

class CNNEnsemble(flexs.Model):
    """A baseline CNN ensemble model with 2 conv layers and 1 dense layer each."""
    def __init__(self, seq_len: int, alphabet: str):
        self.seq_len = seq_len
        self.alphabet = alphabet

        super().__init__(
            name='DynaPPOCNNEnsemble',
            hparam_tune=True,
            hparams_to_search={
                # 'lr': [1e-2, 1e-3, 1e-4],
                # 'epochs': [5, 10, 20, 40],
                'lr': [1e-3],
                'epochs': [1],
            },
            nfolds=5,
        )

    # def _create_ensemble(self, lr, ensemble_size=10):
    def _create_ensemble(self, lr, ensemble_size=5):
        self.models = [self._create_cnn(lr) for _ in range(ensemble_size)]

    def _seqs_to_tensor(self, seqs):
        one_hots = np.array([
            s_utils.string_to_one_hot(seq, self.alphabet, self.seq_len)
            for seq in seqs
        ])

        return tf.convert_to_tensor(one_hots, dtype=tf.float32)

    def _train_hparam_setting(
        self,
        train_seq,
        train_label,
        val_seq,
        val_label,
        epochs=20,
        lr=1e-3
    ):
        self._create_ensemble(lr, ensemble_size=1)

        train_one_hots = self._seqs_to_tensor(train_seq)
        val_one_hots = self._seqs_to_tensor(val_seq)

        train_label = tf.convert_to_tensor(train_label)
        val_label = tf.convert_to_tensor(val_label)

        r2_vals = []
        for model in tqdm(self.models, desc="CNN ensemble hparam"):
            model.fit(
                x=train_one_hots,
                y=train_label,
                batch_size=256,
                epochs=epochs,
                verbose=0,
            )

            val_preds = model.predict(val_one_hots, batch_size=256, verbose=0).squeeze(axis=1)
            r2_vals.append(
                sklearn.metrics.r2_score(val_label.numpy(), val_preds)
            )

        return np.array(r2_vals).mean()

    def _train(self, sequences, labels, **hparam_kwargs):
        epochs, lr = hparam_kwargs['epochs'], hparam_kwargs['lr']
        print('Best epochs: %d, best lr: %f' % (epochs, lr))

        self._create_ensemble(lr)

        one_hots = self._seqs_to_tensor(sequences)
        labels = tf.convert_to_tensor(labels)

        for model in tqdm(self.models, desc="CNN ensemble train"):
            model.fit(
                x=one_hots,
                y=labels,
                batch_size=256,
                epochs=epochs,
                verbose=0,
            )

    def _create_cnn(self, lr):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(
                32,
                5,
                padding="valid",
                activation="relu",
                strides=1,
                input_shape=(self.seq_len, len(self.alphabet)),
            ),
            tf.keras.layers.Conv1D(
                32,
                5,
                padding="same",
                activation="relu",
                strides=1,
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)#, activation="sigmoid"),
        ])

        model.compile(
            loss='MSE',#tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=['mse']
        )

        return model


    def _fitness_function(self, sequences):
        one_hots = self._seqs_to_tensor(sequences)

        arrs = []
        for model in self.models:
            arrs.append(np.nan_to_num(
                model.predict(one_hots, batch_size=256, verbose=0).T
            ))

        preds = np.vstack(arrs).mean(axis=0)
        return preds
