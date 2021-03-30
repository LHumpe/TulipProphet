from typing import Optional, Tuple

import kerastuner as kt
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from ..preprocessing import CryptoWindowGenerator

NEWS_EMBEDDINGS_HP_BOUNDS = {
    'hp_units_long': [22, 46, 4],
    'hp_units_short': [42, 50, 4],
    'hp_units_level_one': [80, 110, 6],
    'hp_dropout_level_one': [0.0, 0.4, 0.2],
    'hp_units_level_two': [20, 32, 4],
    'hp_dropout_level_two': [0.6, 0.8, 0.1],
    'hp_dropout_l1': [0.0, 1e-2, 1e-3, 1e-4],
    'hp_dropout_l2': [0.0, 1e-2, 1e-3, 1e-4],
    'hp_learning_rate': [1e-4, 1e-5],
}


class CryptoDirectionModel:

    def __init__(self, data_generator: CryptoWindowGenerator, seq_short: int, seq_long: int, short_max_tokens: int,
                 long_max_tokens, embedding_dim: int, indicator_len: int, max_epochs: int, version_suffix: str,
                 num_trials: int = 1, tune_dir: Optional[str] = None, project_name: str = 'base',
                 activation: str = 'sigmoid', overwrite: bool = True):
        self.train_df = data_generator.train
        self.val_df = data_generator.val
        self.final_train_df = data_generator.full
        self.test_df = data_generator.test

        self.long_max_tokens = long_max_tokens
        self.short_max_tokens = short_max_tokens
        self.embedding_dim = embedding_dim

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
        self.project_name = project_name

        self.best_hyper_parameters = None

        self.final_activation = activation
        self.overwrite = overwrite

    def _initialize_layers(self) -> Tuple[TextVectorization, TextVectorization]:
        vec_layer_short = TextVectorization(
            standardize=None,
            pad_to_max_tokens=True,
            max_tokens=self.short_max_tokens,
            output_mode='int',
            output_sequence_length=self.seq_short,
            name='short_text_vec'
        )
        short_adapt = self.train_df.map(lambda x, y: x[1])
        vec_layer_short.adapt(short_adapt)

        vec_layer_long = TextVectorization(
            standardize=None,
            pad_to_max_tokens=True,
            max_tokens=self.long_max_tokens,
            output_mode='int',
            output_sequence_length=self.seq_long,
            name='long_text_vec'
        )
        long_adapt = self.train_df.map(lambda x, y: x[2])
        vec_layer_long.adapt(long_adapt)

        return vec_layer_short, vec_layer_long

    def _build_model(self, hp) -> tf.keras.Model:

        """Body Model"""
        long_model_input = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.string,
            name='long_input'
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
                output_dim=self.embedding_dim,
                mask_zero=True,
                name="long_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.embedding_dim,
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

        """Title Model"""
        short_model_input = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.string,
            name='short_input'
        )

        hp_units_short = hp.Int(
            'units_head',
            min_value=self.hp_space['hp_units_short'][0],
            max_value=self.hp_space['hp_units_short'][1],
            step=self.hp_space['hp_units_short'][2]
        )
        short_model = tf.keras.Sequential([
            self.vec_layer_short,
            tf.keras.layers.Embedding(
                input_dim=len(self.vec_layer_short.get_vocabulary()),
                output_dim=self.embedding_dim,
                mask_zero=True,
                name="short_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.embedding_dim,
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

        """Indicator Model"""
        indicator_model_input = tf.keras.layers.Input(
            shape=self.indicator_len,
            dtype=tf.float32,
            name='indicator_input'
        )

        indicator_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.indicator_len,
                activation='relu',
                name='indicator_dense_one'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * 2,
                activation='relu',
                name='indicator_dense_2'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * 4,
                activation='relu',
                name='indicator_dense_3'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * 2,
                activation='relu',
                name='indicator_dense_4'
            ),
            tf.keras.layers.Dense(
                self.indicator_len,
                activation='relu',
                name='indicator_dense_5'
            ),
        ], name='Indicator')

        """Merged Model"""
        merged = tf.keras.layers.concatenate([
            indicator_model(indicator_model_input),
            short_model(short_model_input),
            long_model(long_model_input)
        ], name='Concatenation')

        hp_units_level_one = hp.Int(
            'units_level_one',
            min_value=self.hp_space['hp_units_level_one'][0],
            max_value=self.hp_space['hp_units_level_one'][1],
            step=self.hp_space['hp_units_level_one'][2]
        )
        dense1 = tf.keras.layers.Dense(units=hp_units_level_one, activation='relu', name='merged_dense_one')(merged)

        hp_dropout_level_one = hp.Float(
            'dropout_level_one',
            min_value=self.hp_space['hp_dropout_level_one'][0],
            max_value=self.hp_space['hp_dropout_level_one'][1],
            step=self.hp_space['hp_dropout_level_one'][2]
        )
        dropout1 = tf.keras.layers.Dropout(hp_dropout_level_one, name='merged_dropout_one')(dense1)

        hp_units_level_two = hp.Int(
            'units_level_two',
            min_value=self.hp_space['hp_units_level_two'][0],
            max_value=self.hp_space['hp_units_level_two'][1],
            step=self.hp_space['hp_units_level_two'][2]
        )
        dense2 = tf.keras.layers.Dense(units=hp_units_level_two, activation='relu', name='merged_dense_two')(dropout1)

        hp_dropout_level_two = hp.Float(
            'dropout_level_two',
            min_value=self.hp_space['hp_dropout_level_two'][0],
            max_value=self.hp_space['hp_dropout_level_two'][1],
            step=self.hp_space['hp_dropout_level_two'][2]
        )
        dropout2 = tf.keras.layers.Dropout(hp_dropout_level_two, name='merged_dropout_two')(dense2)

        hp_l1_reg = hp.Choice('l1_reg', values=self.hp_space['hp_dropout_l1'])
        hp_l2_reg = hp.Choice('l2_reg', values=self.hp_space['hp_dropout_l2'])
        output = tf.keras.layers.Dense(
            3,
            activation=self.final_activation,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            bias_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            name='final_output'
        )(dropout2)

        model = tf.keras.Model([
            indicator_model_input,
            short_model_input,
            long_model_input,
        ],
            output
        )

        hp_learning_rate = hp.Choice('learning_rate', values=self.hp_space['hp_learning_rate'])
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=hp_learning_rate),
            metrics='accuracy'
        )
        return model

    def _tune(self, callbacks: list):
        self.tuner = kt.Hyperband(self._build_model,
                                  objective='val_accuracy',
                                  overwrite=self.overwrite,
                                  directory=self.tune_dir,
                                  project_name=self.project_name,
                                  max_epochs=self.max_epochs
                                  )
        self.tuner.search(
            self.train_df,
            validation_data=self.val_df,
            callbacks=callbacks,
            epochs=self.max_epochs,
        )
        self.best_hyper_parameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        return self

    def tune(self, callbacks: list = None):
        if not callbacks:
            callbacks = []
        else:
            pass
        return self._tune(callbacks=callbacks)

    def _train_final_model(self, callbacks: list, max_epochs: int) -> None:

        final_model = self.tuner.hypermodel.build(self.best_hyper_parameters)

        final_model.fit(
            self.final_train_df,
            validation_data=self.test_df,
            callbacks=callbacks,
            epochs=max_epochs
        )

    def final_model(self, max_epochs: int, callbacks: Optional[list] = None) -> None:
        if not callbacks:
            callbacks = list()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='../local_output/models/' + self.version_suffix,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_format='tf',
            save_best_only=True)

        callbacks.append(model_checkpoint_callback)

        self._train_final_model(max_epochs=max_epochs, callbacks=callbacks)
