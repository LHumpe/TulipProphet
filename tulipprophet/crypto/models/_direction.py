import os
from typing import Optional, Tuple

import kerastuner as kt
import tensorflow as tf
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from ..preprocessing import CryptoDirectionGenerator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

"""Change these values to change hyper-parameter search space"""
NEWS_EMBEDDINGS_HP_BOUNDS = {
    # Range parameters i.e. [min, max, step]
    'hp_indicator_factor_one': [2, 3, 1],
    'hp_indicator_factor_two': [3, 5, 1],
    'hp_indicator_factor_three': [2, 3, 1],
    'hp_embedding_dim_short': [8, 32, 4],
    'hp_embedding_dim_long': [8, 32, 4],
    'hp_units_long': [22, 46, 4],
    'hp_units_short': [42, 50, 4],
    'hp_units_level_one': [80, 110, 6],
    'hp_dropout_level_one': [0.0, 0.4, 0.2],
    'hp_units_level_two': [20, 32, 4],
    'hp_dropout_level_two': [0.6, 0.8, 0.1],
    # Choice parameters i.e. [option1, option2, option3]
    'hp_dropout_l1': [0.0, 1e-2, 1e-3, 1e-4],
    'hp_dropout_l2': [0.0, 1e-2, 1e-3, 1e-4],
    'hp_learning_rate': [1e-4, 1e-5],
}


class CryptoLstmDirectionModel:
    """Neural Network Model for the classification of a falling, neutral, rising price of currencies.

    This model ingests three different price_data streams:
    1. Technical Indicators
        These are technical indicators that are commonly used in trading environments such as the MACD.
    2. Short Textual Data
        This represents textual price_data that is short in length e.g. titles, descriptions
    3. Long Textual Data
        This represents the body of the news articles

    The network architecture is build to train a simple dense model on the technical features and to train word
    embeddings utilizing bi-directional lstm nodes on the textual features. For each feature there is a separated model
    which gets gets merged in the end.

    Parameters
    ----------
    data_generator : CryptoDirectionGenerator
        Data Generator returning the different datasets created with tf.price_data api.
    seq_short : int
        Maximum sequence length to allow for the short textual input stream.
    seq_long : int
        Maximum sequence length to allow for the long textual input stream.
    short_max_tokens : int
        Maximum tokens for the text-vectorization layer to consider the short textual input stream.
    long_max_tokens : int
        Maximum tokens for the text-vectorization layer to consider the long textual input stream.
    indicator_len : int
        Number of technical indicators that are going to be used.
    max_epochs : int
        Maximum number of possible epochs to train each model for during tuning.
    num_trials : int
        Maximum number of different hyper-parameter combinations to try.
    version_suffix : str
        Number of version to append to the all logging directory paths for versioning.
    tune_dir : str
        Path to directory where tuning price_data should be stored.
    overwrite : bool
        Whether to overwrite existing tuning files. When False starts a complete new hyper-parameter serach. When True
        continues where it left of last time.
    """

    def __init__(self, data_generator: CryptoDirectionGenerator, seq_short: int, seq_long: int, short_max_tokens: int,
                 long_max_tokens: int, indicator_len: int, max_epochs: int, num_trials: int, version_suffix: str,
                 tune_dir: Optional[str] = None, overwrite: bool = True):
        self.train_ds = data_generator.train
        self.val_ds = data_generator.val
        self.final_train_ds = data_generator.full
        self.test_ds = data_generator.test

        self.long_max_tokens = long_max_tokens
        self.short_max_tokens = short_max_tokens

        self.indicator_len = indicator_len

        self.seq_short = seq_short
        self.seq_long = seq_long

        self.vec_layer_short, self.vec_layer_long = self._initialize_layers()
        self.tuner = None

        self.max_epochs = max_epochs
        self.hp_space = NEWS_EMBEDDINGS_HP_BOUNDS
        self.num_trials = num_trials
        self.tune_dir = tune_dir
        self.version_suffix = version_suffix

        self.best_hyper_parameters = None

        self.overwrite = overwrite

    def _initialize_layers(self) -> Tuple[TextVectorization, TextVectorization]:
        """
        Initializes the the TextVectorisation layers with their vocabulary

        Returns
        -------
        vec_layer_short, vec_layer_long : TextVectorization
            TextVectorization layers that are initialized and adapted according to their vocabulary
        """
        # initialize and adapt layer for short price_data
        vec_layer_short = TextVectorization(
            standardize=None,
            pad_to_max_tokens=True,
            max_tokens=self.short_max_tokens,
            output_mode='int',
            output_sequence_length=self.seq_short,
            name='short_text_vec'
        )
        short_adapt = self.train_ds.map(lambda x, y: x[1])
        vec_layer_short.adapt(short_adapt)

        # initialize and adapt layer for long price_data
        vec_layer_long = TextVectorization(
            standardize=None,
            pad_to_max_tokens=True,
            max_tokens=self.long_max_tokens,
            output_mode='int',
            output_sequence_length=self.seq_long,
            name='long_text_vec'
        )
        long_adapt = self.train_ds.map(lambda x, y: x[2])
        vec_layer_long.adapt(long_adapt)

        return vec_layer_short, vec_layer_long

    def _build_model(self, hp: HyperParameters) -> tf.keras.Model:
        """
        Build the model with respect to the hyper-parameter combinations.

        Parameters
        ----------
        hp : HyperParameters class instance from Keras Tuner
            Hyper-Parameter combinations for the specific trial

        Returns
        -------
        model : tf.keras.Model
            The built and compiled model
        """
        """Indicator Model"""
        hp_indicator_factor_one = hp.Int(
            'indicator_factor_one',
            min_value=self.hp_space['hp_indicator_factor_one'][0],
            max_value=self.hp_space['hp_indicator_factor_one'][1],
            step=self.hp_space['hp_indicator_factor_one'][2]
        )
        hp_indicator_factor_two = hp.Int(
            'indicator_factor_two',
            min_value=self.hp_space['hp_indicator_factor_two'][0],
            max_value=self.hp_space['hp_indicator_factor_two'][1],
            step=self.hp_space['hp_indicator_factor_two'][2]
        )
        hp_indicator_factor_three = hp.Int(
            'indicator_factor_three',
            min_value=self.hp_space['hp_indicator_factor_three'][0],
            max_value=self.hp_space['hp_indicator_factor_three'][1],
            step=self.hp_space['hp_indicator_factor_three'][2]
        )
        indicator_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.indicator_len,
                activation='relu',
                name='indicator_dense_one'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * hp_indicator_factor_one,
                activation='relu',
                name='indicator_dense_2'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * hp_indicator_factor_two,
                activation='relu',
                name='indicator_dense_3'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * hp_indicator_factor_three,
                activation='relu',
                name='indicator_dense_4'
            ),
            tf.keras.layers.Dense(
                self.indicator_len,
                activation='relu',
                name='indicator_dense_5'
            ),
        ], name='Indicator')

        """Short Model"""
        hp_embedding_dim_short = hp.Int(
            'embedding_dim_short',
            min_value=self.hp_space['hp_embedding_dim_short'][0],
            max_value=self.hp_space['hp_embedding_dim_short'][1],
            step=self.hp_space['hp_embedding_dim_short'][2]
        )
        hp_units_short = hp.Int(
            'units_short',
            min_value=self.hp_space['hp_units_short'][0],
            max_value=self.hp_space['hp_units_short'][1],
            step=self.hp_space['hp_units_short'][2]
        )

        short_model = tf.keras.Sequential([
            self.vec_layer_short,
            tf.keras.layers.Embedding(
                input_dim=len(self.vec_layer_short.get_vocabulary()),
                output_dim=hp_embedding_dim_short,
                mask_zero=True,
                name="short_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=hp_embedding_dim_short,
                    return_sequences=False,
                ),
                name='short_bi_lstm'
            ),
            tf.keras.layers.Dense(
                hp_units_short,
                activation='relu',
                name='short_dense'
            ),
        ], name='short')

        """Long Model"""
        hp_embedding_dim_long = hp.Int(
            'embedding_dim_long',
            min_value=self.hp_space['hp_embedding_dim_long'][0],
            max_value=self.hp_space['hp_embedding_dim_long'][1],
            step=self.hp_space['hp_embedding_dim_long'][2]
        )
        hp_units_long = hp.Int(
            'units_long',
            min_value=self.hp_space['hp_units_long'][0],
            max_value=self.hp_space['hp_units_long'][1],
            step=self.hp_space['hp_units_long'][2]
        )

        long_model = tf.keras.Sequential([
            self.vec_layer_long,
            tf.keras.layers.Embedding(
                input_dim=len(self.vec_layer_long.get_vocabulary()),
                output_dim=hp_embedding_dim_long,
                mask_zero=True,
                name="long_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=hp_embedding_dim_long,
                    return_sequences=False,
                ),
                name='long_bi_lstm'
            ),
            tf.keras.layers.Dense(
                hp_units_long,
                activation='relu',
                name='long_dense'
            ),
        ], name='long')

        """Model Inputs"""
        indicator_model_input = tf.keras.layers.Input(
            shape=self.indicator_len,
            dtype=tf.float32,
            name='indicator_input'
        )
        short_model_input = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.string,
            name='short_input'
        )
        long_model_input = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.string,
            name='long_input'
        )

        """Merged Model"""
        hp_units_level_one = hp.Int(
            'units_level_one',
            min_value=self.hp_space['hp_units_level_one'][0],
            max_value=self.hp_space['hp_units_level_one'][1],
            step=self.hp_space['hp_units_level_one'][2]
        )
        hp_dropout_level_one = hp.Float(
            'dropout_level_one',
            min_value=self.hp_space['hp_dropout_level_one'][0],
            max_value=self.hp_space['hp_dropout_level_one'][1],
            step=self.hp_space['hp_dropout_level_one'][2]
        )
        hp_units_level_two = hp.Int(
            'units_level_two',
            min_value=self.hp_space['hp_units_level_two'][0],
            max_value=self.hp_space['hp_units_level_two'][1],
            step=self.hp_space['hp_units_level_two'][2]
        )
        hp_dropout_level_two = hp.Float(
            'dropout_level_two',
            min_value=self.hp_space['hp_dropout_level_two'][0],
            max_value=self.hp_space['hp_dropout_level_two'][1],
            step=self.hp_space['hp_dropout_level_two'][2]
        )
        hp_l1_reg = hp.Choice('l1_reg', values=self.hp_space['hp_dropout_l1'])
        hp_l2_reg = hp.Choice('l2_reg', values=self.hp_space['hp_dropout_l2'])
        hp_learning_rate = hp.Choice('learning_rate', values=self.hp_space['hp_learning_rate'])

        merged = tf.keras.layers.concatenate([
            indicator_model(indicator_model_input),
            short_model(short_model_input),
            long_model(long_model_input)
        ], name='Concatenation')

        dense_level_one = tf.keras.layers.Dense(
            units=hp_units_level_one,
            activation='relu',
            name='merged_dense_one'
        )(merged)

        dropout_level_one = tf.keras.layers.Dropout(
            hp_dropout_level_one,
            name='merged_dropout_one'
        )(dense_level_one)

        dense_level_two = tf.keras.layers.Dense(
            units=hp_units_level_two,
            activation='relu',
            name='merged_dense_two'
        )(dropout_level_one)

        dropout_level_two = tf.keras.layers.Dropout(
            hp_dropout_level_two,
            name='merged_dropout_two'
        )(dense_level_two)

        output = tf.keras.layers.Dense(
            3,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            bias_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            name='final_output'
        )(dropout_level_two)

        model = tf.keras.Model([
            indicator_model_input,
            short_model_input,
            long_model_input,
        ],
            output
        )

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=hp_learning_rate),
            metrics='accuracy'
        )
        return model

    def tune(self, log_dir: Optional[str] = None):
        """
        Start hyper-parameter search.

        Parameters
        ----------
        log_dir : str
            Path to directory where logging price_data should be stored.
        """
        callbacks = list()

        if log_dir:
            logging_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, self.version_suffix))
            callbacks.append(logging_callback)

        self.tuner = kt.Hyperband(
            self._build_model,
            objective='val_accuracy',
            overwrite=self.overwrite,
            directory=self.tune_dir,
            project_name=self.version_suffix,
            max_epochs=self.max_epochs
        )

        self.tuner.search(
            self.train_ds,
            validation_data=self.val_ds,
            callbacks=callbacks,
            epochs=self.max_epochs,
        )
        self.best_hyper_parameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        return self

    def final_model(self, max_epochs: int, model_dir: Optional[str] = None, log_dir: Optional[str] = None,
                    patience: Optional[int] = None) -> None:
        """
        Train and save final model.

        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs to allow during final model training.
        model_dir : str
            Path to where model files should be saved. If None model will not be saved.
        log_dir : str
            Path to directory where logging price_data should be stored. If None training will not be logged.
        patience : int
            Number of epochs to wait for improvement in validation accuracy before terminating the training process.
            If None training will be done for the specified number of epochs.
        """
        callbacks = list()

        if log_dir:
            logging_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, self.version_suffix))
            callbacks.append(logging_callback)

        if patience:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', min_delta=0, patience=patience, verbose=1, mode='auto',
                baseline=None, restore_best_weights=True
            )
            callbacks.append(early_stopping)

        if model_dir:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, self.version_suffix),
                save_weights_only=False,
                monitor='val_accuracy',
                mode='auto',
                save_format='tf',
                save_best_only=True
            )

            callbacks.append(model_checkpoint_callback)

        final_model = self.tuner.hypermodel.build(self.best_hyper_parameters)

        final_model.fit(
            self.final_train_ds,
            validation_data=self.test_ds,
            callbacks=callbacks,
            epochs=max_epochs
        )
