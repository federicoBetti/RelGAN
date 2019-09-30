import random

from real.real_gan.real_loader import RealDataLoader
import pandas as pd
import numpy as np


class RealDataAmazonLoader(RealDataLoader):
    """
    This is a custom data loader for real data that it is used all over the code,
    we can work here to add new data (topic-related data)
    """

    def __init__(self, batch_size, seq_length, end_token=0):
        super().__init__(batch_size, seq_length)
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

        # initialization
        self.num_batch = None
        self.token_stream = None
        self.sequence_batches = None
        self.pointer = None

    def create_batches(self, data_file):
        train_file, dev_file, test_file = data_file[0], data_file[1], data_file[2]
        self.token_stream = []

        df = pd.read_csv(train_file)
        self.token_stream = df[['user_id', 'product_id', 'rating', 'tokenized_text']].values
        for el in self.token_stream:
            el[3] = np.asarray([int(s) for s in el[3].split(" ") if "\n" not in s])

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, axis=0)
        self.pointer = 0

    def next_batch(self, only_text=True):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        user, product, rating, sentence = ret[:, 0], ret[:, 1], ret[:, 2], ret[:, 3]
        return user, product, rating, sentence  # batch_size x

    def random_batch(self, only_text=True):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        if only_text:
            return ret
        return ret, None

    def reset_pointer(self):
        self.pointer = 0
