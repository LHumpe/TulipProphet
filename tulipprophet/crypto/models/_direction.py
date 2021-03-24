import kerastuner as kt
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from ..preprocessing import CryptoWindowGenerator

NEWS_EMBEDDINGS_HP_BOUNDS = {
    'hp_units_bdy': [36, 56, 4],
    'hp_units_ttl': [40, 56, 4],
    'hp_units_level_one': [64, 128, 8],
    'hp_dropout_level_one': [0.2, 0.6, 0.2],
    'hp_units_level_two': [16, 32, 4],
    'hp_dropout_level_two': [0.4, 0.8, 0.2],
}

class CryptoDirectionModel:

    def __init__(self, data_generator: CryptoWindowGenerator,
                 seq_title: int, seq_body: int,
                 bdy_max_tokens: int, ttl_max_tokens: int, embedding_dim: int,
                 indicator_len: int,
                 max_epochs: int,
                 num_trials: int = 1,
                 tune_dir: str = '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_output/tf/',
                 project_name: str = 'base'):
        self.train_df = data_generator.train
        self.val_df = data_generator.val
        self.test_df = data_generator.test
        self.full_df = data_generator.full

        self.bdy_max_tokens = bdy_max_tokens
        self.ttl_max_tokens = ttl_max_tokens
        self.embedding_dim = embedding_dim

        self.indicator_len = indicator_len

        self.seq_ttl = seq_title
        self.seq_bdy = seq_body

        self.vec_layer_bdy, self.vec_layer_ttl = self._initialize_layers()
        self.model = None
        self.tuner = None
        self.history = None

        self.max_epochs = max_epochs
        self.hp_space = NEWS_EMBEDDINGS_HP_BOUNDS
        self.num_trials = num_trials
        self.tune_dir = tune_dir
        self.project_name = project_name

        self.best_hyper_parameters = None

    def _initialize_layers(self):
        vec_layer_bdy = TextVectorization(
            max_tokens=self.bdy_max_tokens,
            output_mode='int',
            output_sequence_length=self.seq_bdy)

        vec_layer_ttl = TextVectorization(
            max_tokens=self.ttl_max_tokens,
            output_mode='int',
            output_sequence_length=self.seq_ttl)

        bdy_adapt = self.train_df.map(lambda x, y: x[1])
        ttl_adapt = self.train_df.map(lambda x, y: x[2])

        vec_layer_bdy.adapt(bdy_adapt)
        vec_layer_ttl.adapt(ttl_adapt)

        return vec_layer_bdy, vec_layer_ttl

    def _build_model(self, hp):

        """Body Model"""
        hp_units_bdy = hp.Int(
            'units_bdy',
            min_value=self.hp_space['hp_units_bdy'][0],
            max_value=self.hp_space['hp_units_bdy'][1],
            step=self.hp_space['hp_units_bdy'][2]
        )

        bdy_model = tf.keras.Sequential([
            self.vec_layer_bdy,
            tf.keras.layers.Embedding(
                input_dim=len(self.vec_layer_bdy.get_vocabulary()),
                output_dim=self.embedding_dim,
                name="bdy_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.embedding_dim,
                    return_sequences=False,
                ),
                name='ttl_bi_lstm'
            ),
            tf.keras.layers.Dense(
                hp_units_bdy,
                activation='relu',
                name='bdy_dense'
            ),
        ], name='Body')

        bdy_model_input = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.string,
            name='bdy_input'
        )

        """Title Model"""
        hp_units_ttl = hp.Int(
            'units_head',
            min_value=self.hp_space['hp_units_ttl'][0],
            max_value=self.hp_space['hp_units_ttl'][1],
            step=self.hp_space['hp_units_ttl'][2]
        )
        ttl_model = tf.keras.Sequential([
            self.vec_layer_ttl,
            tf.keras.layers.Embedding(
                input_dim=len(self.vec_layer_ttl.get_vocabulary()),
                output_dim=self.embedding_dim,
                name="ttl_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.embedding_dim,
                    return_sequences=False,
                ),
                name='ttl_bi_lstm'
            ),
            tf.keras.layers.Dense(
                hp_units_ttl,
                activation='relu',
                name='ttl_dense'
            ),
        ], name='Title')

        ttl_model_input = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.string,
            name='ttl_input'
        )

        """Indicator Model"""
        indicator_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.indicator_len,
                activation='relu',
                name='Idctr_dense_one'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * 2,
                activation='relu',
                name='Idctr_dense_2'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * 4,
                activation='relu',
                name='Idctr_dense_3'
            ),
            tf.keras.layers.Dense(
                self.indicator_len * 2,
                activation='relu',
                name='Idctr_dense_4'
            ),
            tf.keras.layers.Dense(
                self.indicator_len,
                activation='relu',
                name='Idctr_dense_5'
            ),
        ], name='Indicator')

        indicator_model_input = tf.keras.layers.Input(
            shape=(4),
            dtype=tf.float32,
            name='Idctr_input'
        )

        """Merged Model"""
        merged = tf.keras.layers.concatenate([
            indicator_model(indicator_model_input),
            bdy_model(bdy_model_input),
            ttl_model(ttl_model_input)
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

        hp_l1_reg = hp.Choice('l1_reg', values=[1e-1, 1e-2, 1e-3, 1e-4])
        hp_l2_reg = hp.Choice('l2_reg', values=[1e-1, 1e-2, 1e-3, 1e-4])
        output = tf.keras.layers.Dense(
            3,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            bias_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=hp_l1_reg, l2=hp_l2_reg),
            name='final_output'
        )(dropout2)

        model = tf.keras.Model([
            indicator_model_input,
            bdy_model_input,
            ttl_model_input
        ],
            output
        )

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=hp_learning_rate),
            metrics='accuracy'
        )
        return model

    def _tune(self, callbacks: list, store_partial: bool = False):
        self.tuner = kt.BayesianOptimization(self._build_model,
                                             objective='val_accuracy',
                                             overwrite=True,
                                             directory=self.tune_dir,
                                             project_name=self.project_name,
                                             max_trials=self.max_epochs
                                             )
        self.tuner.search(
            self.train_df,
            validation_data=self.val_df,
            callbacks=callbacks,
            epochs=self.max_epochs,
        )
        self.best_hyper_parameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        if store_partial:
            self.model = self.tuner.get_best_models(num_models=1)

        return self

    def tune(self, callbacks: list = None, store_partial: bool = False):
        if not callbacks:
            callbacks = []
        else:
            pass
        return self._tune(callbacks=callbacks, store_partial=store_partial)
