import configparser
import os
import sys

import tensorflow as tf
from flask import Flask

from crypto.io import read_kraken_history, read_json_news
from crypto.preprocessing import CryptoHistoryProphetProcessor, CryptoWindowGenerator, \
    CryptoNewsProphetProcessor

config = configparser.ConfigParser()
config.read('code/crypto/crypto_config.ini')
TECH_INDICATORS = config['preprocessing']['technical_indicators'].split(',')

app = Flask(__name__)


@app.route('/predict_btc/')
def hello_world():
    # TODO: get data from database and enable config
    news_data = read_json_news('local_input/cointelegraph.json')
    news_processor = CryptoNewsProphetProcessor(data=news_data, cols_to_prepare=['title', 'body'])
    news_processor.preprocess_data()

    data = read_kraken_history('local_input/XBTEUR_1440.csv', tic='BTC')
    engineer = CryptoHistoryProphetProcessor(data=data,
                                             tech_indicators=TECH_INDICATORS,
                                             news_data=news_processor.prep_data)
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
    model = tf.keras.models.load_model(os.environ.get('MODEL_DIR'))
    results = model.predict(windows.test)
    print(results, file=sys.stdout)
    return "Done"
