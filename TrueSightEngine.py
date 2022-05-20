import tensorflow
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import numpy as np

PREPOSISI: list = ['di', 'dan', 'yang', 'atau', 'dari', 'pada', 'sejak', 'ke', 'untuk', 'buat',
                   'akan', 'bagi', 'oleh', 'tentang', 'yaitu', 'ala', 'kepada', 'daripada', 'dalam']


def RemoveStopWords(words: list, preposisi=PREPOSISI) -> list:
    """Remove puchtuation and preposisi words"""

    return [s.strip(string.punctuation) for s in words if s.lower().strip(string.punctuation) not in preposisi]


def search(keywords: str, data, search_accuracy: float = 0.5) -> list:
    """
    Search keywords in dataframe

    Returns:
    Returning array of tuple (float accuracy, str text)
    """
    data = list(data)
    search_words = RemoveStopWords(keywords.split())
    filtered_keywords = ' '.join(search_words)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    X = X.T.toarray()
    data_frame = pd.DataFrame(X, index=vectorizer.get_feature_names_out())

    word_vect = vectorizer.transform(
        [filtered_keywords]).toarray().reshape(data_frame.shape[0],)
    search_rate = {}

    for i in range(len(data)):
        search_rate[i] = np.dot(data_frame.loc[:, i].values, word_vect) / np.linalg.norm(
            data_frame.loc[:, i]) * np.linalg.norm(word_vect)
        print(np.dot(data_frame.loc[:, i].values, word_vect))

    rate_sorted = sorted(
        search_rate.items(), key=lambda x: x[1], reverse=True)
    result = []

    for k, v in rate_sorted:
        word_found = 0
        if v != 0.0:
            for word in search_words:
                if word in data[k]:
                    word_found += 1

            if (word_found / len(search_words) > search_accuracy):
                result.append((v, data[k]))

    return result
