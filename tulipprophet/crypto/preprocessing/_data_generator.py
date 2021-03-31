from typing import List

import pandas as pd
import tensorflow as tf


class CryptoDirectionGenerator:
    """
    DataGenerator for the CryptoLstmDirectionModel creating a batched prefetch dataset.

    Parameters
    ----------
    training_data, validation_data, test_data : pd.DataFrame
        DataFrames holding the respective instances for training, validation and testing.
    batch_size : int
        Size of batches for training.
    indicator_cols : List[str]
        Names of technical indicators that should be used by the model.
    word_col_short, word_col_long : str
        Names of the columns that contain the textual information.
    label_cols : List[str]
        Name of the columns that contain the one hot encoded target classes.
    """

    def __init__(self, training_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame,
                 batch_size: int, indicator_cols: List[str], word_col_short: str, word_col_long: str,
                 label_cols: List[str]):
        self.train_df = training_data
        self.val_df = validation_data
        self.test_df = test_data
        self.full_df = pd.concat([self.train_df, self.val_df], axis=0)

        self.indicator_columns = indicator_cols
        self.short_word_col = word_col_short
        self.long_word_col = word_col_long
        self.label_column = label_cols

        self.__batch_size = batch_size

    def _make_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Returns a batched prefetch dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Data that should be processed.

        Returns
        -------
        ds : tf.data.Dataset
            Batched prefetch dataset.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (
                (
                    data[self.indicator_columns].to_numpy(),
                    data[self.short_word_col].to_numpy(),
                    data[self.long_word_col].to_numpy()
                ),
                data[self.label_column].to_numpy()
            )
        )
        ds = ds.batch(batch_size=self.__batch_size)

        ds = ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    @property
    def train(self):
        return self._make_dataset(self.train_df)

    @property
    def val(self):
        return self._make_dataset(self.val_df)

    @property
    def test(self):
        return self._make_dataset(self.test_df)

    @property
    def full(self):
        return self._make_dataset(self.full_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
        return result
