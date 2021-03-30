import json
from typing import List

import numpy as np
import pandas as pd


def read_json_news(path: str, word_cols=List[str]) -> pd.DataFrame:
    """
    Import and format news data from .json files.

    Parameters
    ----------
    path : str
        Path to the .json file.
    word_cols :
        Columns that contain the relevant textual data.

    Returns
    -------
    df : pd.DataFrame
        DataFrame that contains properly formatted news data
    """
    # load json file
    with open(path) as file:
        json_data = json.load(file)

    # convert json to DF
    df = pd.DataFrame.from_dict(data=json_data)

    # remove rows containing empty strings in relevant columns
    df = df.replace('', np.nan).dropna(subset=word_cols)

    # cast time to datetime
    df['date'] = pd.to_datetime(df['date'])

    return df


def read_kraken_history(path: str, tic: str = 'Coin') -> pd.DataFrame:
    """
    Import and format historical price data from the kraken exchange.

    Parameters
    ----------
    path : str
        Path to the historic price data
    tic : str
        Identifier for the currency. Important when training the model on many currencies e.g. in indicator calculation

    Returns
    -------
    df : pd.DataFrame
        DataFrame that contains properly formatted historic price data
    """
    # import data
    df = pd.read_csv(path, names=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]

    # add currency identifier
    df['tic'] = tic

    # cast time to datetime
    df['date'] = pd.to_datetime(df['date'], unit='s')

    return df
