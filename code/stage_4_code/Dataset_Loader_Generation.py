from code.base_class.dataset import dataset
from collections import Counter
import itertools

class Dataset_Loader(dataset):
    dataset_name = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_w_to_ind = None
    vocab_ind_to_w = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_raw(self):

        root_path = self.dataset_source_folder_path + self.dataset_source_file_name
        jokes = []
        jokes_lengths = []
        with open(root_path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.split(",", 1)[1].replace("\n", "").replace("\"", "").lower()
                line = line.split()
                jokes.append(line)
                jokes_lengths.append(len(line))
        return jokes, jokes_lengths

    def create_vocabulary(self, cleaned_strings):

        words_set = list(itertools.chain.from_iterable(cleaned_strings))
        counts = Counter(words_set)

        vocabulary_w_to_i = {}
        vocabulary_i_to_w = {}
        for i, item in enumerate(counts.most_common()):
            word = item[0]
            vocabulary_w_to_i[word] = i
            vocabulary_i_to_w[i] = word

        self.vocab_w_to_ind = vocabulary_w_to_i
        self.vocab_ind_to_w = vocabulary_i_to_w

        return vocabulary_w_to_i, vocabulary_i_to_w

    def prepare_input(self, sequences, vocab, sequence_length=3):

        prev = []
        next_sent = []

        for sentence in sequences:
            encoded_sentence = [vocab[i] for i in sentence]
            for i in range(0, len(encoded_sentence) - sequence_length):
                prev_code = encoded_sentence[i:i+sequence_length]
                next_code = encoded_sentence[i+sequence_length]
                prev.append(prev_code)
                next_sent.append(next_code)
        return prev, next_sent

    def load(self):
        jokes, jokes_len = self.load_raw()
        vocab_w_i, vocab_i_w = self.create_vocabulary(jokes)
        context, targets = self.prepare_input(jokes, vocab_w_i)
        return context, targets, vocab_w_i, vocab_i_w, jokes, jokes_len





