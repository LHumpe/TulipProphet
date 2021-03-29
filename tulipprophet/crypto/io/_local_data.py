import json
from typing import List

import numpy as np
import pandas as pd


def read_json_news(path: str, word_cols=List[str], with_empty_str: bool = False) -> pd.DataFrame:
    # Load json file
    with open(path) as file:
        json_data = json.load(file)

    # Convert json to DF
    df = pd.DataFrame.from_dict(data=json_data)

    if with_empty_str:
        df = df.replace('', np.nan).dropna(subset=word_cols)

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
