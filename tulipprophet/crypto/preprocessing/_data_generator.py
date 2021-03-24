from typing import List

import pandas as pd
import tensorflow as tf


class CryptoWindowGenerator:
    def __init__(self, training_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame,
                 batch_size: int,
                 indicator_cols: List[str], word_cols: List[str], label_col: List[str]):
        # Store the raw data.
        self.train_df = training_data
        self.val_df = validation_data
        self.test_df = test_data
        self.full_df = pd.concat([self.train_df, self.val_df], axis=0)

        # Store Column Info
        self.indicator_columns = indicator_cols
        self.word_columns = word_cols
        self.label_column = label_col

        self.batch_size = batch_size

    def __repr__(self):
        pass

    def make_dataset(self, data):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                (
                    data[self.indicator_columns].to_numpy(),
                    data[self.word_columns[0]].to_numpy(),
                    data[self.word_columns[1]].to_numpy()
                ),
                data[self.label_column].to_numpy()
            )
        )
        ds = ds.batch(batch_size=self.batch_size)

        ds = ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def full(self):
        return self.make_dataset(self.full_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
        return result
