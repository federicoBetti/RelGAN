import random

from real.real_gan.loaders.real_loader import RealDataLoader
import pandas as pd
import numpy as np


class RealDataCustomerReviewsLoader(RealDataLoader):
    """
    This is a custom data loader for real data that it is used all over the code,
    we can work here to add new data (topic-related data)
    """

    def __init__(self, batch_size, seq_length, end_token=0):
        super().__init__(batch_size, seq_length)
        self.model_index_word_dict = None
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

        # initialization
        self.num_batch = None
        self.token_stream = None
        self.sequence_batches_train = None
        self.sequence_batches_validation = None
        self.pointer = None

    def create_batches(self, data_file):
        train_file = data_file[0]
        self.token_stream = []

        df = pd.read_csv(train_file)
        self.token_stream = df[['sentiment', 'tokenized_text']].values
        for el in self.token_stream:
            el[1] = np.asarray([int(s) for s in el[1][1:-1].split(", ") if "\n" not in s])

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches_train = np.split(np.array(self.token_stream), self.num_batch, axis=0)

        df_negative = df[df.sentiment == 0]
        self.token_stream = df_negative[['tokenized_text']].values
        for el in self.token_stream:
            el[0] = np.asarray([int(s) for s in el[0][1:-1].split(", ") if "\n" not in s])

        self.num_batch_neg = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.batch_size * self.num_batch_neg]
        self.sequence_batches_negative = np.split(np.array(self.token_stream), self.num_batch_neg, axis=0)

        df_positive = df[df.sentiment == 1]
        self.token_stream = df_positive[['tokenized_text']].values
        for el in self.token_stream:
            el[0] = np.asarray([int(s) for s in el[0][1:-1].split(", ") if "\n" not in s])

        self.num_batch_pos = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.batch_size * self.num_batch_pos]
        self.sequence_batches_positive = np.split(np.array(self.token_stream), self.num_batch_pos, axis=0)

        self.pointer = 0

    def next_batch(self, only_text=True):
        ret = self.sequence_batches_train[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        sentiment, sentence = ret[:, 0], ret[:, 1]
        return sentiment, sentence  # batch_size x

    def random_batch(self, only_text=True, dataset="train"):
        rn_pointer = random.randint(0, self.num_batch - 1)
        if dataset == "train":
            ret = self.sequence_batches_train[rn_pointer]
        elif dataset == "validation":
            ret = self.sequence_batches_validation[rn_pointer]
        else:
            raise ValueError
        sentiment, sentence = ret[:, 0], ret[:, 1]
        return sentiment, sentence

    def get_positive_negative_batch(self):
        rn_pointer = random.randint(0, self.num_batch_pos - 1)
        pos = self.sequence_batches_positive[rn_pointer]
        rn_pointer = random.randint(0, self.num_batch_neg - 1)
        neg = self.sequence_batches_negative[rn_pointer]
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches_train[rn_pointer]
        sentiment, sentence = ret[:, 0], ret[:, 1]
        return sentiment, sentence, pos, neg

    def reset_pointer(self):
        self.pointer = 0
