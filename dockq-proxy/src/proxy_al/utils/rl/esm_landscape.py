"""Defines the BertGFPBrightness landscape."""
import os

import numpy as np
import flexs


class ESMLandscape(flexs.Landscape):
    def __init__(self, proxy):
        """
        Create ESMLandscape.
        """
        super().__init__(name="ESM")
        self.proxy = proxy
        self.x = []
        self.y = []
        
    def get_dataset(self):
        return self

    def train(self, one_hot_sequences, scores):
        return

    def get_full_dataset(self):
        return self.x, self.y

    def add(self, batch):
        samples, scores = batch
        self.x = np.concatenate((self.x, samples), axis=0)
        self.y = np.concatenate((self.y, scores), axis=0)
    
    def _fitness_function(self, sequences):
        return self.proxy.evaluate(sequences)

    def close(self):
        self.proxy.close()
