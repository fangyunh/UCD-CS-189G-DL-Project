from code.base_class.dataset import dataset
import numpy as np
import os, re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.utils import shuffle

class Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_raw_data(self):
        data = {}
        labels = {}
        root_path = self.dataset_source_folder_path
        for group in ["train", "test"]:
            reviews = []
            review_labels = []
            for lab in ['pos', 'neg']:
                path = root_path + f"/{group}/{lab}"
                for file in os.listdir(path):
                    if os.fsdecode(file).endswith(".txt"):
                        with open(path + "/" + file, 'r', encoding='utf-8') as text_file:
                            text = text_file.read()
                            label = {"pos" : 1, "neg" : 0}[lab]
                            reviews.append(text)
                            review_labels.append(label)
                data[group] = np.array(reviews)
                labels[group] = np.array(review_labels)


        data["train"], labels["train"] = shuffle(data["train"], labels["train"])
        data["test"], labels["test"] = shuffle(data["test"], labels["test"])

        return data, labels

    def reg_expression(self, sentence):
        sentence = re.sub(r"[^a-zA-Z0-9\sentence]", "", sentence)
        sentence = re.sub(r"\sentence+", "", sentence)
        sentence = re.sub(r"\d", "", sentence)
        return sentence

    def prep_sentences(self, data_dict, labels_dict):
        words_set = []
        stop_words = set(stopwords.words("english"))
        data_size = 25000
        for sentence in data_dict["train"][:data_size]:
            for word in sentence.lower().split():
                word = self.reg_expression(word)
                if word not in stop_words and word != '':
                    words_set.append(word)

        vocab = Counter(words_set).most_common(1000)
        vocab = [item[0] for item in vocab]
        encoder_dict = {word:i+1 for i, word in enumerate(vocab)}

        X_train, X_test = [], []
        for sentence in data_dict["train"][:data_size]:
            encoded_sequence = [encoder_dict[self.reg_expression(word)] for word in sentence.lower().split()
                                if self.reg_expression(word) in encoder_dict.keys()]
            X_train.append(encoded_sequence)

        for sentence in data_dict["test"][:data_size]:
            encoded_sequence = [encoder_dict[self.reg_expression(word)] for word in sentence.lower().split()
                                if self.reg_expression(word) in encoder_dict.keys()]
            X_test.append(encoded_sequence)

        return {"X_train" : np.array(X_train), "X_test" : np.array(X_test),
                "y_train" : np.array(labels_dict["train"])[:data_size],
                "y_test" : np.array(labels_dict["test"])[:data_size]}, encoder_dict

    def padding(self, data, max_length=250):
        padded_inputs = np.zeros((len(data), max_length), dtype=int)
        for i, sentence in enumerate(data):
            if len(sentence) != 0:
                padded_inputs[i, -len(sentence):] = sentence[:max_length]
        return padded_inputs

    def load(self):
        raw_data, raw_labels = self.load_raw_data()
        cleaned_data, vocab = self.prep_sentences(raw_data, raw_labels)
        X_train = self.padding(cleaned_data["X_train"])
        X_test = self.padding(cleaned_data["X_test"])
        y_train = cleaned_data["y_train"]
        y_test = cleaned_data["y_test"]

        return {"X_train" : np.array(X_train), "X_test" : np.array(X_test),
                "y_train" : np.array(y_train), "y_test" : np.array(y_test)}, vocab
