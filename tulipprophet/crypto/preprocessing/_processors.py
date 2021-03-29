import re
import unicodedata
from collections import Counter
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from stockstats import StockDataFrame as Sdf


class CryptoHistoryProphetProcessor:

    def __init__(self, data: pd.DataFrame, news_data=pd.DataFrame, tech_indicators: Optional[list] = None):
        self.__data = data
        self._prep_data = self.__data.copy()
        self.tech_indicators = tech_indicators
        self.news_data = news_data

    def preprocess_data(self):
        if self.tech_indicators:
            self.__add_tech_indicators()

        self.__create_label()

        return self

    def __create_label(self):
        percentage_change = (self._prep_data['close'] - self._prep_data['open']) / self._prep_data['open']

        q1 = np.quantile(percentage_change, 0.33)
        q2 = np.quantile(percentage_change, 0.66)

        self._prep_data.loc[percentage_change <= q1, 'fall'] = 1
        self._prep_data.loc[percentage_change > q1, 'fall'] = 0
        self._prep_data['fall'] = self._prep_data['fall'].shift(-1)

        self._prep_data.loc[(percentage_change > q1) & (percentage_change <= q2), 'neutral'] = 1
        self._prep_data.loc[(percentage_change <= q1) | (percentage_change > q2), 'neutral'] = 0
        self._prep_data['neutral'] = self._prep_data['neutral'].shift(-1)

        self._prep_data.loc[percentage_change > q2, 'rise'] = 1
        self._prep_data.loc[percentage_change <= q2, 'rise'] = 0
        self._prep_data['rise'] = self._prep_data['rise'].shift(-1)

        return self

    def __add_tech_indicators(self):
        temp_data = Sdf.retype(self._prep_data.copy())
        unique_ticker = temp_data.tic.unique()

        for indicator in self.tech_indicators:
            indicator_df = pd.DataFrame()
            for u_tick in unique_ticker:
                temp_indicator = pd.DataFrame(temp_data[temp_data.tic == u_tick][indicator])
                indicator_df = indicator_df.append(temp_indicator, ignore_index=True)

            self._prep_data[indicator] = indicator_df
            self._prep_data.dropna(inplace=True)

        return self

    def slice_data(self, start: str, end: Optional[str] = None):
        if end:
            temp_data = self._prep_data[(self._prep_data.date >= start) & (self._prep_data.date < end)]
        else:
            temp_data = self._prep_data[self._prep_data.date >= start]
        self._prep_data = temp_data.sort_values(['date', 'tic'], ignore_index=True)
        self._prep_data.index = self._prep_data.date.factorize()[0]

        return self

    def train_test_split(self,
                         train_size: float, val_size: float,
                         shuffle_data: bool = False) -> Tuple[Any, Any, Any]:
        if shuffle_data:
            data = shuffle(self._prep_data, random_state=2)
        else:
            data = self._prep_data.copy()

        data_len = len(data)

        train = self._merge_with_news(data.iloc[0:int(train_size * data_len)])
        validation = self._merge_with_news(
            data.iloc[int(train_size * data_len):int(data_len * (train_size + val_size))])
        test = self._merge_with_news(data.iloc[int(data_len * (train_size + val_size)):])

        train.index = train.date.factorize()[0]
        validation.index = validation.date.factorize()[0]
        test.index = test.date.factorize()[0]

        return train, validation, test

    def _merge_with_news(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(data, self.news_data, how='left', on='date').dropna()

    @property
    def prep_data(self) -> pd.DataFrame:
        return self._prep_data.dropna()


class CryptoNewsProphetProcessor:
    def __init__(self, data: pd.DataFrame, text_col_short: str, text_col_long: str):
        self.__data = data
        self._prep_data = self.__data.copy()

        self.rel_cols = [text_col_short, text_col_long]
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are"
        }

    def preprocess_data(self):
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.normalize_text)
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.replace_contractions)
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.replace_punctuation)
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.replace_numbers)
        return self

    # TODO: Remove single letters

    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.replace(b'\xe2\x80\x94'.decode('utf-8'), '--')
        text = re.sub(r'\n\s*\n', '', text)
        return text

    def replace_contractions(self, text: str) -> str:
        for key, value in self.contractions.items():
            text = text.replace(key, value)
        return text

    @staticmethod
    def replace_punctuation(text: str) -> str:
        for punct in "/-'":
            text = text.replace(punct, ' ')
        for punct in '&':
            text = text.replace(punct, 'and')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~‘' + '“”’':
            text = text.replace(punct, '')

        text = re.sub(' +', ' ', text)
        return text

    @staticmethod
    def replace_numbers(text: str) -> str:
        text = re.sub('[0-9]{5,}', '#####', text)
        text = re.sub('[0-9]{4}', '####', text)
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)
        return text

    @staticmethod
    def get_max_tokens(data: pd.DataFrame, threshold: int) -> int:
        results = Counter()

        data.str.lower().str.split().apply(results.update)

        rel_words = [word for word, count in results.items() if count >= threshold]

        return len(rel_words)

    @staticmethod
    def get_sequence_len(data: pd.DataFrame, threshold: int) -> int:
        return int(np.percentile(data.str.split().apply(len), threshold))

    def calc_vocab_stats(self,
                         threshold_short: List[int] = [0, 100],
                         threshold_long: List[int] = [0, 100]) -> Tuple[int, int, int, int]:

        max_t_short = self.get_max_tokens(self._prep_data[self.rel_cols[0]], threshold=threshold_short[0])
        seq_len_short = self.get_max_tokens(self._prep_data[self.rel_cols[1]], threshold=threshold_short[1])

        max_t_long = self.get_sequence_len(self._prep_data[self.rel_cols[0]], threshold=threshold_long[0])
        seq_len_long = self.get_sequence_len(self._prep_data[self.rel_cols[1]], threshold=threshold_long[1])

        return max_t_short, seq_len_short, max_t_long, seq_len_long

    @property
    def prep_data(self) -> pd.DataFrame:
        return self._prep_data.dropna()
