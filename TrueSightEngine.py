from __future__ import annotations
import pathlib
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


class SearchEngine:

    PREPOSISI: list = ['di', 'dan', 'yang', 'atau', 'dari', 'pada', 'sejak', 'ke', 'untuk', 'buat',
                       'akan', 'bagi', 'oleh', 'tentang', 'yaitu', 'ala', 'kepada', 'daripada', 'dalam']

    def RemoveStopWords(words: list, preposisi=PREPOSISI) -> list:
        """Remove puchtuation and preposisi words"""

        return [s.strip(string.punctuation) for s in words if s.lower().strip(string.punctuation) not in preposisi]

    def search(keywords: str, data, search_accuracy: float = 0.5) -> list:
        """
        Search keywords in data of string

        Returns:
        Returning array of tuple (float accuracy, str text)
        """
        data = list(data)
        search_words = SearchEngine.RemoveStopWords(keywords.split())
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

    def search_from_dict(keywords: str, data, lookupHeader, search_accuracy: float = 0.5) -> list:
        """
        Search keywords in dict

        lookupHeader: Array of header name to lookup

        Returns:
        Returning array of tuple (float accuracy, dict item)
        """
        datalist = list()
        result = []

        for item in list(data):
            strbuild = ""
            for header in lookupHeader:
                strbuild += str(item[header]) + " "
            datalist.append(strbuild)

        datalist = list(datalist)
        search_words = SearchEngine.RemoveStopWords(keywords.split())
        filtered_keywords = ' '.join(search_words)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(datalist)
        X = X.T.toarray()
        data_frame = pd.DataFrame(X, index=vectorizer.get_feature_names_out())

        word_vect = vectorizer.transform(
            [filtered_keywords]).toarray().reshape(data_frame.shape[0],)
        search_rate = {}

        for i in range(len(datalist)):
            search_rate[i] = np.dot(data_frame.loc[:, i].values, word_vect) / np.linalg.norm(
                data_frame.loc[:, i]) * np.linalg.norm(word_vect)

        rate_sorted = sorted(
            search_rate.items(), key=lambda x: x[1], reverse=True)
        result = []

        for k, v in rate_sorted:
            word_found = 0
            if v != 0.0:
                for word in search_words:
                    if word in datalist[k]:
                        word_found += 1

                if (word_found / len(search_words) > search_accuracy):
                    result.append((v, data[k]))

        return result


class TensorHelper:

    def openModel(path) -> TFModel:
        model = TFModel()
        model.LoadTFLiteModel(path)
        return model

    def train_new_claim():
        pass

    def predict_claim(model: TFModel, claimtext: str):
        lbl, acc = model.predict(claimtext, {0: "fake", 1: "fact"})
        return lbl, acc


class TFModel:
    def __init__(self):
        self.model = None
        pass

    def loadTFLiteModel(self, path: str):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        pass

    def saveTFLiteModel(self, path: str):
        parentDir = os.pathsep.join(path.split(os.pathsep)[:-1])
        filename = path.split(os.pathsep)[-1]
        tf.saved_model.save(self.model, os.path.join(parentDir, filename))
        converter = tf.lite.TFLiteConverter.from_saved_model(
            os.path.join(parentDir, filename))

        tf_model_file = pathlib.Path(path + '.tflite')
        tf_model_file.write_bytes(converter.convert())

    def predict(self, test, classname: dict):
        self._prediction = []

        self.interpreter.set_tensor(self._input_details[0]['index'], test)
        self.interpreter.invoke()
        self._prediction.append(self.interpreter.get_tensor(
            self._output_details[0]['index']))

        predicted_label = np.argmax(self._prediction)

        return {
            "label": classname[predicted_label],
            "accuracy": np.max(self._prediction)
        }
