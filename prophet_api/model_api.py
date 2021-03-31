import json

import requests
from flask import Flask

app = Flask(__name__)


@app.route('/predict_direction_btc/')
def hello_world():
    data = {
        "signature_name": "serving_default",
        "inputs": {
            'indicator_input': [[1, 2, 3, 4, 5, 6, 7, 8]],
            'short_input': [['hallo']],
            'long_input': [['hallo']]
        }
    }

    response = requests.post('http://tensorflow-servings:8501/v1/models/direction:predict', data=json.dumps(data))

    return response.json()


# @app.route('/predict_direction_btc/')
# def hello_world():
#     # TODO: get price_data from database and enable config
#     tel_data = read_json_news(config['PATHS']['telegraph'], word_cols=[TEXT_COL_S, TEXT_COL_L], with_empty_str=False)
#     news_processor = CryptoNewsProphetProcessor(
#         price_data=tel_data,
#         text_col_short=TEXT_COL_L,
#         text_col_long=TEXT_COL_L
#     ).preprocess_data()
#
#     price_data = read_kraken_history(config['PATHS']['price_history'], tic='BTC')
#
#     engineer = CryptoHistoryProphetProcessor(
#         price_data=price_data,
#         tech_indicators=TECH_INDICATORS,
#         news_data=news_processor.prep_data
#     ).preprocess_data()
#
#     train_df, val_df, test_df = engineer.train_test_split(train_size=0.7, val_size=0.29, shuffle_data=True)
#
#     price_data = {
#         "signature_name": "serving_default",
#         "inputs": {
#             'indicator_input': test_df[TECH_INDICATORS].values.tolist(),
#             'short_input': test_df['title'].values.reshape(-1, 1).tolist(),
#             'long_input': test_df['body'].values.reshape(-1, 1).tolist()
#         }
#     }
#
#     response = requests.post('http://tensorflow-servings:8501/v1/models/direction:predict', price_data=json.dumps(price_data))
#
#     return response.json()