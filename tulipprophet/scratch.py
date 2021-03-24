import tensorflow as tf

from crypto.io import read_kraken_history, read_json_news
from crypto.models import CryptoDirectionModel
from crypto.preprocessing import CryptoHistoryProphetProcessor, CryptoWindowGenerator, \
    CryptoNewsProphetProcessor
if __name__ == '__main__':
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
    engineer.slice_data(start='2018-01-01').preprocess_data()

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

    max_tokens = news_processor.get_max_tokens(threshold=15)

    predictor = CryptoDirectionModel(
        data_generator=windows,
        max_tokens=max_tokens,
        seq_title=50,
        seq_body=800,
        indicator_len=len(['macd', 'rsi_30', 'cci_30', 'dx_30']),
        embedding_dim=16,
        max_epochs=100,
        num_trials=100,
    )

    predictor.tune(
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir="/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipProphet/local_output/logdir/tune/"),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto',
                baseline=None, restore_best_weights=False
            ),
        ],
        store_partial=False,
    )
