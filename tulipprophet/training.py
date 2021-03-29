import pandas as pd
import tensorflow as tf

from crypto.io import read_kraken_history, read_json_news
from crypto.models import CryptoDirectionModel
from crypto.preprocessing import CryptoHistoryProphetProcessor, CryptoWindowGenerator, \
    CryptoNewsProphetProcessor

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import configparser

config = configparser.ConfigParser()
config.read('crypto/crypto_config.ini')
TECH_INDICATORS = config['preprocessing']['technical_indicators'].split(',')
TEXT_COL_S = config['preprocessing']['text_cols_short']
TEXT_COL_L = config['preprocessing']['text_col_long']

if __name__ == '__main__':
    tel_data = read_json_news(config['PATHS']['telegraph'], word_cols=[TEXT_COL_S, TEXT_COL_L], with_empty_str=False)
    desk_data = read_json_news(config['PATHS']['desk'], word_cols=[TEXT_COL_S, TEXT_COL_L], with_empty_str=True)

    news_data = pd.concat([tel_data, desk_data], axis=0, ignore_index=True)

    news_processor = CryptoNewsProphetProcessor(
        data=news_data,
        text_col_short=TEXT_COL_L,
        text_col_long=TEXT_COL_L
    ).preprocess_data()

    data = read_kraken_history(config['PATHS']['price_history'], tic='BTC')

    engineer = CryptoHistoryProphetProcessor(
        data=data,
        tech_indicators=TECH_INDICATORS,
        news_data=news_processor.prep_data
    ).preprocess_data()

    train_df, val_df, test_df = engineer.train_test_split(train_size=0.7, val_size=0.2, shuffle_data=True)

    windows = CryptoWindowGenerator(
        training_data=train_df,
        validation_data=val_df,
        test_data=test_df,
        batch_size=52,
        indicator_cols=TECH_INDICATORS,
        word_cols=[TEXT_COL_S, TEXT_COL_L],
        label_col=['fall', 'neutral', 'rise'],
    )

    bdy_max_tokens, ttl_max_tokens, bdy_seq_len, ttl_seq_len = news_processor.calc_vocab_stats()

    predictor = CryptoDirectionModel(data_generator=windows, seq_short=ttl_seq_len, seq_long=bdy_seq_len,
                                     long_max_tokens=bdy_max_tokens, short_max_tokens=ttl_max_tokens, embedding_dim=16,
                                     indicator_len=len(TECH_INDICATORS), max_epochs=5, num_trials=10,
                                     tune_dir='../local_output/tf/', version_suffix="V2")

    logging_callback = tf.keras.callbacks.TensorBoard(log_dir="../local_output/logdir/lstm/")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0, patience=20, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )

    predictor.tune(callbacks=[logging_callback])

    predictor.final_model(max_epochs=1000, callbacks=[early_stopping])
