import tensorflow as tf

from tulipprophet.crypto.io import read_kraken_history, read_json_news
from tulipprophet.crypto.preprocessing import CryptoHistoryProphetProcessor, CryptoWindowGenerator, \
    CryptoNewsProphetProcessor

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

news_data = read_json_news(
    '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_input/cointelegraph.json')

news_processor = CryptoNewsProphetProcessor(data=news_data, cols_to_prepare=['title', 'body'])
news_processor.preprocess_data()
prepped_news_data = news_processor.prep_data

data = read_kraken_history(
    '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_input/XBTEUR_1440.csv', tic='BTC')

engineer = CryptoHistoryProphetProcessor(
    data=data,
    tech_indicators=['macd', 'rsi_30', 'cci_30', 'dx_30']
)
engineer.preprocess_data()

final_data = engineer.merge_with_news(news_data=prepped_news_data)
train_df, val_df, test_df = engineer.train_test_split(final_data, train_size=0.7, val_size=0.2, shuffle_data=True)

windows = CryptoWindowGenerator(
    training_data=train_df,
    validation_data=val_df,
    test_data=test_df,
    batch_size=52,
    indicator_cols=['macd', 'rsi_30', 'cci_30', 'dx_30'],
    word_cols=['title', 'body'],
    label_col=['fall', 'neutral', 'rise'],
)
model = tf.keras.models.load_model('/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/tulipprophet/weights/crypto_model')
model.evaluate(windows.test)
