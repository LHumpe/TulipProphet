import configparser
import json

import requests
from flask import Flask

from crypto.io import read_kraken_history, read_json_news
from crypto.preprocessing import CryptoHistoryProphetProcessor, CryptoNewsProphetProcessor

config = configparser.ConfigParser()
config.read('code/crypto/crypto_config.ini')
TECH_INDICATORS = config['PREPROCESSING']['technical_indicators'].split(',')
TEXT_COL_S = config['PREPROCESSING']['text_cols_short']
TEXT_COL_L = config['PREPROCESSING']['text_col_long']

app = Flask(__name__)


@app.route('/predict_direction_btc/')
def hello_world():
    # TODO: get data from database and enable config
    tel_data = read_json_news(config['PATHS']['telegraph'], word_cols=[TEXT_COL_S, TEXT_COL_L], with_empty_str=False)
    news_processor = CryptoNewsProphetProcessor(
        data=tel_data,
        text_col_short=TEXT_COL_L,
        text_col_long=TEXT_COL_L
    ).preprocess_data()

    data = read_kraken_history(config['PATHS']['price_history'], tic='BTC')

    engineer = CryptoHistoryProphetProcessor(
        data=data,
        tech_indicators=TECH_INDICATORS,
        news_data=news_processor.prep_data
    ).preprocess_data()

    train_df, val_df, test_df = engineer.train_test_split(train_size=0.7, val_size=0.29, shuffle_data=True)

    data = {
        "signature_name": "serving_default",
        "inputs": {
            'indicator_input': test_df[TECH_INDICATORS].values.tolist(),
            'short_input': test_df['title'].values.reshape(-1, 1).tolist(),
            'long_input': test_df['body'].values.reshape(-1, 1).tolist()
        }
    }

    response = requests.post('http://tensorflow-servings:8501/v1/models/direction:predict', data=json.dumps(data))

    return response.json()
