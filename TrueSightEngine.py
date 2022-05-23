from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
session = tf.compat.v1.Session(config=configuration)

stop_words = stopwords.words('indonesian')


class TimeExecution:
    def init(self):
        self.timestamp = datetime.now().timestamp()

    def end(self):
        print()
        print(int((datetime.now().timestamp() - self.timestamp) * 1000), 'ms')
        print()


class SearchEngine:

    # PREPOSISI: list = ['di', 'dan', 'yang', 'atau', 'dari', 'pada', 'sejak', 'ke', 'untuk', 'buat',
    #                    'akan', 'bagi', 'oleh', 'tentang', 'yaitu', 'ala', 'kepada', 'daripada', 'dalam']

    def addDataToDictionary(new_data: dict, dictionary: dict):
        for header in list(new_data.keys()):
            total_data = 0
            if header in dictionary:
                total_data = len(dictionary[header])
            else:
                dictionary[header] = {}
            dictionary[header][total_data] = new_data[header]

        return dictionary

    def RemoveStopWords(words: list, stop_words=stop_words) -> list:
        """Remove puchtuation and preposisi words"""

        return [s.strip(string.punctuation) for s in words if s.lower().strip(string.punctuation) not in stop_words]

    def search(keywords: str, data, search_accuracy: float = 0.5, use_stopwords=True) -> list:
        """
        Search keywords in data of string

        Returns:
        Returning array of tuple (float accuracy, str text)
        """
        data = list(data)
        search_words = SearchEngine.RemoveStopWords(
            keywords.split()) if use_stopwords else keywords.split()
        filtered_keywords = ' '.join(
            search_words) if use_stopwords else keywords

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

    def search_from_dict(keywords: str, data, lookupHeader, search_accuracy: float = 0.5, use_stopwords=True) -> list:
        """
        Search keywords in dict

        lookupHeader: Array of header name to lookup

        Returns:
        Returning array of tuple (float accuracy, dict item)
        """
        datalist = list()
        result = []

        timecount = TimeExecution()
        timecount.init()
        print("Convert dictionary to list")

        for header in list(lookupHeader):
            for i, (_, item) in enumerate(data[header].items()):
                if len(datalist) <= i:
                    datalist.append(item)
                else:
                    datalist[i] += " " + item
        timecount.end()

        timecount.init()
        print("Remove stopWords")

        search_words = SearchEngine.RemoveStopWords(
            keywords.split()) if use_stopwords else keywords.split()
        filtered_keywords = ' '.join(
            search_words) if use_stopwords else keywords

        timecount.end()

        timecount.init()
        print("Vectorization")

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(datalist)
        X = X.T.toarray()
        data_frame = pd.DataFrame(X, index=vectorizer.get_feature_names_out())

        timecount.end()

        timecount.init()
        print("Transform")

        word_vect = vectorizer.transform(
            [filtered_keywords]).toarray().reshape(data_frame.shape[0],)
        search_rate = {}

        timecount.end()

        timecount.init()
        print("Calculating rate")

        norm_vector = np.linalg.norm(word_vect)
        for i in range(len(datalist)):
            search_rate[i] = np.dot(data_frame.loc[:, i].values, word_vect) / np.linalg.norm(
                data_frame.loc[:, i]) * norm_vector

        timecount.end()

        timecount.init()
        print("Sorting")

        rate_sorted = sorted(
            search_rate.items(), key=lambda x: x[1], reverse=True)
        result = []

        timecount.end()

        timecount.init()
        print("Generate Result")
        for k, v in rate_sorted:
            word_found = 0
            if v != 0.0:
                for word in search_words:
                    if word in datalist[k]:
                        word_found += 1

                if (word_found / len(search_words) > search_accuracy):
                    data_row = {}
                    for header in list(data.keys()):
                        data_row[header] = data[header][k]
                    result.append((v, data_row))

        timecount.end()

        return result


class TensorHelper:

    def __init__(self, threshold) -> None:
        self.THRESHOLD = threshold
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "indobenchmark/indobert-base-p1")

    def openModel(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict_claim(self, claimtext: str, predict_text_length):
        result = self.model.predict(
            self._bert_encode([claimtext], predict_text_length))
        return {
            'claim': claimtext,
            'accuracy': result[0],
            'fake': result[0] > self.THRESHOLD
        }

    def saveModel(self, path):
        self.model.save(path)

    def _bert_encode(self, data, max_len):
        input_ids = []
        attention_masks = []

        for i in range(len(data)):
            encoded = self.bert_tokenizer.encode_plus(data[i],
                                                      add_special_tokens=True,
                                                      max_length=max_len,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True)

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        return np.array(input_ids), np.array(attention_masks)
