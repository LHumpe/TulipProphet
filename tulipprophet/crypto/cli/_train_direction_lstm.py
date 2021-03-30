import configparser

import click
import pandas as pd

from ..io import read_kraken_history, read_json_news
from ..models import CryptoDirectionModel
from ..preprocessing import CryptoHistoryProphetProcessor, CryptoWindowGenerator, \
    CryptoNewsProphetProcessor


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True))
def _train_direction_lstm(config_path: str):

    # Import the config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Extract parameters for readability
    TECH_INDICATORS = config['PREPROCESSING']['technical_indicators'].split(',')
    TEXT_COL_S = config['PREPROCESSING']['text_cols_short']
    TEXT_COL_L = config['PREPROCESSING']['text_col_long']

    NEWS_PATHS = config['PATHS']['news_data'].split(',')
    TUNE_DIR = config['PATHS']['tune_dir']
    LOG_DIR = config['PATHS']['log_dir']
    MODEL_DIR = config['PATHS']['model_dir']

    VERSION_SUFFIX = config['MODEL_CONFIG']['version_suffix']
    MAX_EPOCHS = config.getint('MODEL_CONFIG', 'max_epochs')
    MAX_FINAL_EPOCHS = config.getint('MODEL_CONFIG', 'max_final_epochs')
    NUM_TRIALS = config.getint('MODEL_CONFIG', 'num_trials')
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'batch_size')
    OVERWRITE = config.getboolean('MODEL_CONFIG', 'overwrite')
    PATIENCE = config.getboolean('MODEL_CONFIG', 'early_stopping_patience')

    single_news_frames = []
    for path in NEWS_PATHS:
        single_news_frames.append(read_json_news(path, word_cols=[TEXT_COL_S, TEXT_COL_L]))

    news_data = pd.concat(single_news_frames, axis=1, ignore_index=True)

    data = read_kraken_history(config['PATHS']['price_history'], tic='BTC')

    news_processor = CryptoNewsProphetProcessor(
        data=news_data,
        text_col_short=TEXT_COL_L,
        text_col_long=TEXT_COL_L
    ).preprocess_data()

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
        batch_size=BATCH_SIZE,
        indicator_cols=TECH_INDICATORS,
        word_cols=[TEXT_COL_S, TEXT_COL_L],
        label_col=['fall', 'neutral', 'rise'],
    )

    bdy_max_tokens, ttl_max_tokens, bdy_seq_len, ttl_seq_len = news_processor.calc_vocab_stats()

    predictor = CryptoDirectionModel(data_generator=windows, seq_short=ttl_seq_len, seq_long=bdy_seq_len,
                                     long_max_tokens=bdy_max_tokens, short_max_tokens=ttl_max_tokens,
                                     indicator_len=len(TECH_INDICATORS), max_epochs=MAX_EPOCHS, num_trials=NUM_TRIALS,
                                     tune_dir=TUNE_DIR, overwrite=OVERWRITE, version_suffix=VERSION_SUFFIX)

    predictor.tune(log_dir=LOG_DIR)

    predictor.final_model(max_epochs=MAX_FINAL_EPOCHS, model_dir=MODEL_DIR, log_dir=LOG_DIR, patience=PATIENCE)
