import json

import pandas as pd


def read_json_news(path: str) -> pd.DataFrame:
    # Load json file
    with open(path) as file:
        json_data = json.load(file)

    # Convert json to DF
    df = pd.DataFrame.from_dict(data=json_data)

    # Cast time to datetime
    df['date'] = pd.to_datetime(df['date'])

    return df


def read_kraken_history(path: str, tic: str) -> pd.DataFrame:
    # Import data
    df = pd.read_csv(path, names=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]

    # Add currency identifier
    df['tic'] = tic

    # Cast time to datetime
    df['date'] = pd.to_datetime(df['date'], unit='s')

    return df
