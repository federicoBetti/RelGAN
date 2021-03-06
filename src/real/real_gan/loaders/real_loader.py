import random
import time
from typing import Dict, List

import gc
import numpy as np

from path_resolution import resources_path
from topic_modelling.lda_utils import get_perc_sent_topic, process_texts, get_perc_few_sent_topic
from topic_modelling.lda_topic import train_specific_LDA, get_corpus


# from gensim.utils import lemmatize


class RealDataLoader:
    """
    This is a custom data loader for real data that it is used all over the code,
    we can work here to add new data (topic-related data)
    """

    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

        # initialization
        self.num_batch = None
        self.token_stream = None
        self.sequence_batches = None
        self.pointer = None
        self.lda = None

    def create_batches(self, data_file):
        self.token_stream = []
        self.data_file = data_file

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, axis=0)
        self.pointer = 0

    def next_batch(self, only_text=True):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        if only_text:
            return ret
        return ret, None

    def random_batch(self, only_text=True):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        if only_text:
            return ret
        return ret, None

    def reset_pointer(self):
        self.pointer = 0

    def set_dictionaries(self, word_index_dict: Dict[str, int], index_word_dict: Dict[str, str]):
        self.model_word_index_dict = word_index_dict
        self.model_index_word_dict = index_word_dict
        assert len(word_index_dict) == len(index_word_dict)
        self.vocab_size = len(self.model_index_word_dict)

    def get_LDA(self, data_file):
        self.vocab_size = len(self.model_index_word_dict)

        print("Computation of topic model started...")
        t = time.time()

        # Create LDA model for the dataset, given parameters
        coco = True if "coco" in data_file else False  # Now it is just coco or not coco just for name saving reasons
        corpus_raw = get_corpus(coco, datapath=data_file)
        self.lda = train_specific_LDA(corpus_raw, num_top=self.topic_num, passes=2, iterations=2, chunksize=2000,
                                      dataset_name=self.dataset)

        # Get percentage of each topic in each sentence

        self.topic_matrix = self.lda.lda_model.get_topics()  # num_topic x num_words

        # get model lemmatized version of the words, it's needed because LDA does it and model processing doesn't
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        self.texts = [lemmatizer.lemmatize(word) for word in self.model_word_index_dict.keys()]
        self.lda_index_word_dict = self.lda.dictionary.id2token

        self.inverse_indexes = [self.get_model_index(i) for i in range(len(self.lda_index_word_dict))]
        print("number of LDA words: {}".format(len(self.lda_index_word_dict)))

        print("Topic model computed in {} sec!".format(time.time() - t))
        gc.collect()

    def get_model_index(self, lda_index) -> List[int]:
        word = self.lda_index_word_dict[lda_index]
        from_lemmatize = [text_index for text_index in range(len(self.texts)) if self.texts[text_index] == word]
        return from_lemmatize

    def get_topic(self, sentences):
        if self.lda is None:
            self.get_LDA(self.data_file)
        t = time.time()
        tmp = process_texts(sentences, self.lda.stops)
        corpus_bow = [self.lda.dictionary.doc2bow(i) for i in tmp]
        df = get_perc_few_sent_topic(ldamodel=self.lda.lda_model, corpus_bow=corpus_bow, topic_num=self.topic_num)
        topic_weights = df.values[:, 1:self.topic_num + 1]  # num_sentences x num_topic (each row sum to 1)
        topic_sentences = np.dot(topic_weights, self.topic_matrix)  # num_sentences x num_lda_word

        real_vector = np.zeros(
            (topic_sentences.shape[0], len(self.model_word_index_dict) + 1))  # sentence_number x vocab_size

        for ind, invere_index in enumerate(self.inverse_indexes):
            for x in invere_index:
                real_vector[:, x] = topic_sentences[:, ind]
        gc.collect()
        real_vector = np.divide(real_vector, np.sum(real_vector, axis=1, keepdims=True))
        return real_vector

    def set_files(self, data_file, lda_file):
        """
        set different files to train lda model and GAN model \n
        :param data_file: file with sentences used to train GAN
        :param lda_file: file with sentences used to train LDA
        :return: None
        """
        self.lda_model_file = lda_file
        self.data_file = data_file

    def set_dataset(self, dataset_name):
        self.dataset = dataset_name


class RealDataTopicLoader(RealDataLoader):
    model_index_word_dict: Dict[str, str]
    model_word_index_dict: Dict[str, int]

    def __init__(self, batch_size, seq_length):
        super().__init__(batch_size, seq_length)
        self.vocab_size = None
        self.model_word_index_dict = None
        self.model_index_word_dict = None
        self.sentence_topic_array = None
        self.topic_batches = None
        self.topic_matrix = None
        self.batches_shape = None
        self.topic_num = 3
        self.lda = None
        self.lda_model_file = None
        self.data_file = None
        self.dataset = None

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        gc.collect()
        self.sentence_topic_array = self.get_sentence_topic_array()
        self.num_batch = int(len(self.token_stream) / self.batch_size)

        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sentence_topic_array = np.asarray(self.sentence_topic_array)[:self.num_batch * self.batch_size]

        print(self.num_batch, self.batch_size, len(self.sentence_topic_array))
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, axis=0)
        self.topic_batches = np.split(np.array(self.sentence_topic_array), self.num_batch, axis=0)

        self.pointer = 0

    def next_batch(self, only_text=True):
        # with the parameter I can change only when needed
        ret_sent = self.sequence_batches[self.pointer]
        ret_topic = self.topic_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        if only_text:
            return ret_sent
        else:
            return ret_sent, ret_topic

    def random_batch(self, only_text=True):
        # with the parameter I can change only when needed
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret_sent = self.sequence_batches[rn_pointer]
        ret_topic = self.topic_batches[rn_pointer]
        if only_text:
            return ret_sent
        else:
            return ret_sent, ret_topic

    def set_dictionaries(self, word_index_dict: Dict[str, int], index_word_dict: Dict[str, str]):
        self.model_word_index_dict = word_index_dict
        self.model_index_word_dict = index_word_dict
        assert len(word_index_dict) == len(index_word_dict)
        self.vocab_size = len(self.model_index_word_dict)
        # self.sentence_topic_array = self.get_sentence_topic_array(data_file)

    def get_sentence_topic_array(self):
        """
        compute for each sentence in the corpus its topic vector.
        It does it extracting the topics in the dataset with LDA, it checks the influence of each topic
        in each sentence and compute the topic of each sentence with a weighted sum over the topics. \n
        :return: array of dim (sentence_number, word_count in the model vocabulary)
        """
        print("Computation of topic model started...")
        t = time.time()

        # Create LDA model for the dataset, given parameters
        corpus_raw = get_corpus(datapath=self.lda_model_file)
        self.lda = train_specific_LDA(corpus_raw, num_top=self.topic_num, passes=2, iterations=2, chunksize=2000,
                                      dataset_name=self.dataset)

        self.topic_matrix = self.lda.lda_model.get_topics()  # num_topic x num_words

        # get model lemmatized version of the words, it's needed because LDA does it and model processing doesn't
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        self.texts = [lemmatizer.lemmatize(word) for word in self.model_word_index_dict.keys()]
        self.lda_index_word_dict = self.lda.dictionary.id2token

        self.inverse_indexes = [self.get_model_index(i) for i in range(len(self.lda_index_word_dict))]
        print("number of LDA words: {}".format(len(self.lda_index_word_dict)))

        print("Topic model computed in {} sec!".format(time.time() - t))
        gc.collect()

        return self.compute_real_vector()

    def compute_real_vector(self):
        t = time.time()
        df = get_perc_sent_topic(lda=self.lda, topic_num=self.topic_num, data_file=self.data_file)
        topic_weights = df.values[:, 1:self.topic_num + 1]  # num_sentences x num_topic (each row sum to 1)
        topic_sentences = np.dot(topic_weights, self.topic_matrix)  # num_sentences x num_word
        topic_sentences = np.divide(topic_sentences,
                                    np.sum(topic_sentences, axis=1, keepdims=True))  # rowwise normalization

        real_vector = np.zeros(
            (topic_sentences.shape[0], len(self.model_word_index_dict) + 1))  # sentence_number x vocab_size

        for ind, invere_index in enumerate(self.inverse_indexes):
            # more than one index in the model because of lemmatization
            for x in invere_index:
                real_vector[:, x] = topic_sentences[:, ind]

        real_vector = np.divide(real_vector, np.sum(real_vector, axis=1, keepdims=True))
        gc.collect()
        print("Compute real vector time: {}".format(time.time() - t))
        return real_vector

    def random_topic(self):
        """
        It returns a random topic, it could be completely random or sampled randomly from the existing ones
        :return: an array of batch size topic vectors taken randomly from the real one present in the system
        """
        idx = np.random.randint(self.sentence_topic_array.shape[0], size=self.batch_size)
        return self.sentence_topic_array[idx, :]

    def get_model_index(self, lda_index) -> List[int]:
        word = self.lda_index_word_dict[lda_index]
        from_lemmatize = [text_index for text_index in range(len(self.texts)) if self.texts[text_index] == word]
        return from_lemmatize

    def get_LDA(self, word_index_dict, index_word_dict, data_file):
        self.vocab_size = len(self.model_index_word_dict)

        print("Computation of topic model started...")
        t = time.time()

        # Create LDA model for the dataset, given parameters
        coco = True if "coco" in data_file else False  # Now it is just coco or not coco just for name saving reasons
        corpus_raw = get_corpus(coco, datapath=data_file)
        self.lda = train_specific_LDA(corpus_raw, num_top=self.topic_num, passes=2, iterations=2, chunksize=2000,
                                      dataset_name=self.dataset)

        # Get percentage of each topic in each sentence

        self.topic_matrix = self.lda.lda_model.get_topics()  # num_topic x num_words

        # get model lemmatized version of the words, it's needed because LDA does it and model processing doesn't
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        self.texts = [lemmatizer.lemmatize(word) for word in self.model_word_index_dict.keys()]
        self.lda_index_word_dict = self.lda.dictionary.id2token

        self.inverse_indexes = [self.get_model_index(i) for i in range(len(self.lda_index_word_dict))]
        print("number of LDA words: {}".format(len(self.lda_index_word_dict)))

        print("Topic model computed in {} sec!".format(time.time() - t))
        gc.collect()

    def get_topic(self, sentences):
        t = time.time()
        tmp = process_texts(sentences, self.lda.stops)
        corpus_bow = [self.lda.dictionary.doc2bow(i) for i in tmp]
        df = get_perc_few_sent_topic(ldamodel=self.lda.lda_model, corpus_bow=corpus_bow, topic_num=self.topic_num)
        topic_weights = df.values[:, 1:self.topic_num + 1]  # num_sentences x num_topic (each row sum to 1)
        topic_sentences = np.dot(topic_weights, self.topic_matrix)  # num_sentences x num_lda_word

        real_vector = np.zeros(
            (topic_sentences.shape[0], len(self.model_word_index_dict) + 1))  # sentence_number x vocab_size

        for ind, invere_index in enumerate(self.inverse_indexes):
            for x in invere_index:
                real_vector[:, x] = topic_sentences[:, ind]
        gc.collect()
        real_vector = np.divide(real_vector, np.sum(real_vector, axis=1, keepdims=True))
        return real_vector

    def set_files(self, data_file, lda_file):
        """
        set different files to train lda model and GAN model \n
        :param data_file: file with sentences used to train GAN
        :param lda_file: file with sentences used to train LDA
        :return: None
        """
        self.lda_model_file = lda_file
        self.data_file = data_file

    def set_dataset(self, dataset_name):
        self.dataset = dataset_name
