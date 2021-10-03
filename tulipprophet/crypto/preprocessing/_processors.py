import re
import unicodedata
from collections import Counter
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from stockstats import StockDataFrame as Sdf


class CryptoHistoryProphetProcessor:
    """
    Instance providing preprocessing for price price_data and the merging of news price_data.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing price price_data.
        Columns must be ['date', 'open', 'high', 'low', 'close', 'volume', 'amount'].
    news_data : pd.DataFrame
        DataFrame containing the news price_data
    tech_indicators : List[str]
        List of names for technical indicators supported by the stockstats package.
    fall_quantile : float
        Quantile which marks the upper bound for the class "fall". E.g. when 0.25 then the lowest 25% of the
        observations regarding their value of percentage change will be considered as "fall".
    rise_quantile : float
        Quantile which marks the upper bound for the class "neutral". E.g. when 0.75  and fall quantile 0.25
        observations which fall into the range of 0.25% and 0.75% quantiles regarding their percentage change will be
        labeled as "neutral".
    """

    def __init__(self, price_data: pd.DataFrame, news_data: pd.DataFrame, tech_indicators: List[str],
                 fall_quantile: float, rise_quantile: float):
        self.price_data = price_data
        self.news_data = news_data

        self.tech_indicators = tech_indicators

        self.fall_quantile = fall_quantile
        self.rise_quantile = rise_quantile

        self._prep_data = self.price_data.copy()

    def preprocess_data(self):
        """Create Technical indicators and create the labels."""

        self._add_tech_indicators()

        self._create_label()

        return self

    def _create_label(self):
        """Create the label according to the quantiles specified and shift."""

        percentage_change = (self._prep_data['close'] - self._prep_data['open']) / self._prep_data['open']

        q1 = np.quantile(percentage_change, self.fall_quantile)
        q2 = np.quantile(percentage_change, self.rise_quantile)

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

    def _add_tech_indicators(self):
        """Calculate and add technical indicators per currency pair."""
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
        """
        Slice data according to date. Creates a subset only containing observations from the specified time frame.

        Parameters
        ----------
        start : str
            String representing the first observation that should be used. Must be in the same DateTime format as
            the date column.
        end : str
            String representing the last observation that should be used. Must be in the same DateTime format as
            the date column.
        """
        if end:
            temp_data = self._prep_data[(self._prep_data.date >= start) & (self._prep_data.date < end)]
        else:
            temp_data = self._prep_data[self._prep_data.date >= start]

        self._prep_data = temp_data.sort_values(['date', 'tic'], ignore_index=True)
        # self._prep_data.index = self._prep_data.date.factorize()[0]

        return self

    def make_subsets(self, train_size: float, val_size: float, shuffle_data: bool = False) -> Tuple[Any, Any, Any]:
        """
        Split the data into train, validation and test sets and merge the textual data. The train and validation set
        will be sized in regards to the specified fractions. The test set will contain the remainder of the data.

        Parameters
        ----------
        train_size : float
            Fraction of the data that should be used for the training set.
        val_size : float
            Fraction of the data that should be used as the validation set.
        shuffle_data : bool
            Whether to shuffle observations before splitting.

        Returns
        -------
        train, validation, test : pd.DataFrame
            Three DataFrames containing the corresponding subsets.
        """
        if shuffle_data:
            temp_data = shuffle(self._prep_data, random_state=2)
        else:
            temp_data = self._prep_data.copy()

        # Split the data
        data_len = len(temp_data)
        train_subset = temp_data.iloc[0:int(train_size * data_len)]
        train = pd.merge(train_subset, self.news_data, how='left', on='date').dropna()
        # train.index = train.date.factorize()[0]

        val_subset = temp_data.iloc[int(train_size * data_len):int(data_len * (train_size + val_size))]
        validation = pd.merge(val_subset, self.news_data, how='left', on='date').dropna()
        # validation.index = validation.date.factorize()[0]

        test_subset = temp_data.iloc[int(data_len * (train_size + val_size)):]
        test = pd.merge(test_subset, self.news_data, how='left', on='date').dropna()
        # test.index = test.date.factorize()[0]

        return train, validation, test

    @property
    def prep_data(self) -> pd.DataFrame:
        return self._prep_data.dropna()


class CryptoNewsProphetProcessor:
    """
    Instance providing preprocessing for news data such as replacements for punctuation and contractions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the news information.
    text_col_short : str
        Name of the column for short textual data.
    text_col_long : str
        Name of the column for long textual data.
    """

    def __init__(self, data: pd.DataFrame, text_col_short: str, text_col_long: str):
        self.data = data
        self._prep_data = self.data.copy()

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
        """Call all cleaning functions."""
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.normalize_text)
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.replace_contractions)
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.replace_punctuation)
        self._prep_data[self.rel_cols] = self._prep_data[self.rel_cols].applymap(self.replace_numbers)
        return self

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize html data and remove unwanted symbols and signs."""

        text = unicodedata.normalize("NFKD", text)
        text = text.replace(b'\xe2\x80\x94'.decode('utf-8'), '--')
        text = re.sub(r'\n\s*\n', '', text)
        text = re.sub(r'\r', '', text)
        text = re.sub(r'\n', '', text)
        return text

    def replace_contractions(self, text: str) -> str:
        """Replace contractions with un-contracted versions to reduce the amount of unique vocabulary."""
        for key, value in self.contractions.items():
            text = text.replace(key, value)
        return text

    @staticmethod
    def replace_punctuation(text: str) -> str:
        """Replace punctuation with whitespaces."""
        for punct in "/-'":
            text = text.replace(punct, '')
        for punct in '&':
            text = text.replace(punct, 'and')
        for punct in '?!.,"#$%\'()*+–/:;<=>@[\\]^_`{|}~‘' + '““’':
            text = text.replace(punct, '')

        text = re.sub(' +', ' ', text)
        text = re.sub('” ', ' ', text)
        return text

    @staticmethod
    def replace_numbers(text: str) -> str:
        """Replace numbers of different length with a hashtag representation to reduce vocabulary."""
        text = re.sub('[0-9]{5,}', '#####', text)
        text = re.sub('[0-9]{4}', '####', text)
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)
        return text

    @staticmethod
    def get_max_tokens(data: pd.DataFrame, threshold: int) -> int:
        """
        Get the maximum number of unique words that the model should be able to learn.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing only one column storing either long or short textual data.
        threshold : int
            How often a word has to occur to be considered as relevant to learn.

        Returns
        -------
        n_rel_tokens : int
            Number of relevant tokens.
        """
        results = Counter()

        data.str.lower().str.split().apply(results.update)

        rel_words = [word for word, count in results.items() if count >= threshold]

        n_rel_tokens = len(rel_words)

        return n_rel_tokens

    @staticmethod
    def get_sequence_len(data: pd.DataFrame, threshold: int) -> int:
        """
        Get the sequence length for a specific word column which is later used in the vectorization layer.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing only one column storing either long or short textual data.
        threshold : int
            Fraction that specifies which len to return in regards to how much percent of the observations have smaller
            or equal length of sequences.

        Returns
        -------
        max_sequence_length : int
            Number specifying the maximum sequence length to consider.
        """
        max_sequence_length = int(np.percentile(data.str.split().apply(len), threshold))
        return max_sequence_length

    def calc_vocab_stats(self, threshold_short: Optional[List[int]] = None,
                         threshold_long: Optional[List[int]] = None) -> Tuple[int, int, int, int]:
        """
        Wrapper to return the relevant textual parameters during training. Thresholds are used to reduce unnecessary
        training time e.g. when only one sequence in the short word column is 1000 words long and the rest only five
        words long.

        Parameters
        ----------
        threshold_short, threshold_long : List[token_threshold, sequence_threshold]
            Thresholds for short and long word columns.
            token_threshold: Specifies the number of occurrences needed for a word to be considered.
            sequence_threshold: Specifies the fraction of how many of the sequences can be shorter or equal to the
                                resulting percentile.
        """

        if threshold_short is None:
            threshold_short = [0, 100]

        if threshold_long is None:
            threshold_long = [0, 100]

        max_t_short = self.get_max_tokens(self._prep_data[self.rel_cols[0]], threshold=threshold_short[0])
        seq_len_short = self.get_max_tokens(self._prep_data[self.rel_cols[1]], threshold=threshold_short[1])

        max_t_long = self.get_sequence_len(self._prep_data[self.rel_cols[0]], threshold=threshold_long[0])
        seq_len_long = self.get_sequence_len(self._prep_data[self.rel_cols[1]], threshold=threshold_long[1])

        return max_t_short, seq_len_short, max_t_long, seq_len_long

    @property
    def prep_data(self) -> pd.DataFrame:
        return self._prep_data.dropna()
