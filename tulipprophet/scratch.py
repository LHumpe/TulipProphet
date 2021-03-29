import os

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

if __name__ == '__main__':
    tf.get_logger().setLevel('WARN')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    news_data = read_json_news(
        '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_input/cointelegraph.json')

    news_processor = CryptoNewsProphetProcessor(data=news_data, cols_to_prepare=['title', 'body'])
    news_processor.preprocess_data()
    prepped_news_data = news_processor.prep_data

    data = read_kraken_history(
        '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_input/XBTEUR_1440.csv', tic='BTC')

    engineer = CryptoHistoryProphetProcessor(
        data=data,
        tech_indicators=TECH_INDICATORS,
        news_data=prepped_news_data
    )
    engineer.preprocess_data()

    train_df, val_df, test_df = engineer.train_test_split(train_size=0.7, val_size=0.2, shuffle_data=True)

    windows = CryptoWindowGenerator(
        training_data=train_df,
        validation_data=val_df,
        test_data=test_df,
        batch_size=52,
        indicator_cols=TECH_INDICATORS,
        word_cols=['title', 'body'],
        label_col=['fall', 'neutral', 'rise'],
    )

    bdy_max_tokens = news_processor.get_max_tokens(data=prepped_news_data['body'], threshold=15)
    ttl_max_tokens = news_processor.get_max_tokens(data=prepped_news_data['title'], threshold=15)
    bdy_seq_len = news_processor.get_sequence_len(data=prepped_news_data['body'], threshold=75)
    ttl_seq_len = news_processor.get_sequence_len(data=prepped_news_data['title'], threshold=75)

    predictor = CryptoDirectionModel(
        data_generator=windows,
        bdy_max_tokens=bdy_max_tokens,
        ttl_max_tokens=ttl_max_tokens,
        seq_title=ttl_seq_len,
        seq_body=bdy_seq_len,
        indicator_len=len(TECH_INDICATORS),
        embedding_dim=16,
        max_epochs=60,
        num_trials=100,
        tune_dir='/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_output/tf/second/',
    )

    predictor.tune(
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir="/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_output/logdir/tune/second/",
            ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto',
            #     baseline=None, restore_best_weights=False
            # ),
        ]
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    ),
    predictor.final_model(callbacks=[early_stopping])
