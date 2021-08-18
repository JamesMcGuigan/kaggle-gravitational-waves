import os
from zipfile import ZipFile

import numpy as np
import tensorflow as tf


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, df, directory, batch_size=32, random_state=42, shuffle=True, target=True, ext='.npy'):
        np.random.seed(random_state)

        self.directory  = directory
        self.df         = df
        self.shuffle    = shuffle
        self.target     = target
        self.batch_size = batch_size
        self.ext        = ext

        self.on_epoch_end()

    def __len__(self):
        return np.ceil(self.df.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        batch = self.df[ start_idx : start_idx + self.batch_size ]

        signals = []

        if self.directory.endswith('.zip'):
            with ZipFile(self.directory) as zipfile:
                for fname in batch.id:
                    with zipfile.open(fname + self.ext) as npfile:
                        data = np.load(npfile)
                        signals.append(data)
        else:
            for fname in batch.id:
                path = os.path.join(self.directory, fname + self.ext)
                data = np.load(path)
                signals.append(data)

        signals = np.stack(signals).astype('float32')

        if self.target:
            return signals, batch.target.values
        else:

            return signals

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
