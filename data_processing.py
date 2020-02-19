import os

import numpy as np
import pickle as pkl
import time
from tqdm import tqdm

from collections import defaultdict, Counter


import download_ap
import read_ap


class APDataset():

    def __init__(self, window_size=2, vocab_file_name='vocabulary'):
        # ensure dataset is downloaded
        download_ap.download_dataset()
        # pre-process the text
        docs_by_id = read_ap.get_processed_docs()

        vocab_size, vocab_list = self.create_vocabulary(docs_by_id)

        self.docs_by_id = docs_by_id
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.vocab_list = vocab_list
        self.word_to_ix = {word: index for index,
                           word in enumerate(vocab_list)}

    def create_vocabulary(self, docs, vocab_file_name='vocabulary'):
        vocab_path = "./vocabulary.pkl"
        if os.path.exists(vocab_path):

            with open(vocab_path, "rb") as reader:
                vocab = pkl.load(reader)

            vocab_len = vocab["vocab_len"]
            vocab_words = vocab["vocab_words"]
            return vocab_len, vocab_words
        else:
            word_counter = Counter()

            doc_ids = list(docs.keys())

            print("Building Vocabulary")
            for doc_id in tqdm(doc_ids):
                doc = docs[doc_id]
                word_counter.update(doc)

            thresholded_vocab = Counter(
                {x: word_counter[x] for x in word_counter if word_counter[x] >= 50})

            vocab_words = [word for word, _ in thresholded_vocab.most_common()]
            vocab_len = len(vocab_words)

            with open(vocab_path, "wb") as writer:
                vocab = {
                    "vocab_len": vocab_len,
                    "vocab_words": vocab_words
                }
                pkl.dump(vocab, writer)
            return vocab_len, vocab_words

    def word_to_one_hot(self, word):
        one_hot = np.zeros(self.vocab_size)
        one_hot[self.word_to_ix[word]] = 1
        return np.array(one_hot).transpose()

    def get_generator(self):
        doc_ids = list(self.docs_by_id.keys())
        for doc_id in doc_ids:
            doc = self.docs_by_id[doc_id]
            n_words = len(doc)
            for index, word in enumerate(doc):
                word = self.word_to_one_hot(word)
                for window in range(1, self.window_size + 1):
                    try:
                        if index-window >= 0 and index-window <= n_words:
                            context = self.word_to_one_hot(doc[index-window])
                            yield (word, context)
                    except:
                        pass
                    try:
                        if index+window >= 0 and index+window <= n_words:
                            context = self.word_to_one_hot(doc[index-window])
                            yield (word, context)
                    except:
                        pass
