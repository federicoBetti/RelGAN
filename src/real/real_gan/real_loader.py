import numpy as np
import random

from gensim.utils import lemmatize

from topic_modelling.lda_topic import train_lda, train_specific_LDA
from topic_modelling.lda_utils import get_corpus, get_perc_sent_topic


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

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, axis=0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def random_batch(self):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        return ret

    def reset_pointer(self):
        self.pointer = 0


def get_LDA(data_file):
    pass


class RealDataTopicLoader(RealDataLoader):
    def __init__(self, batch_size, seq_length):
        super().__init__(batch_size, seq_length)
        self.model_word_index_dict = None
        self.sentence_topic_array = None

    def create_batches(self, data_file):
        lda_model = train_lda(datapath=data_file)
        # or, for a specific model
        # corpus = get_corpus(datapath=data_file)
        # lda = train_specific_LDA(corpus, num_top=3, passes=2, iterations=2, chunksize=2000, coco=coco)

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

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, axis=0)
        self.pointer = 0

    def set_model_word_index_dict(self, word_index_dict):
        self.model_word_index_dict = word_index_dict
        self.sentence_topic_array = self.get_sentence_topic_array()

    def get_sentence_topic_array(self):
        """
        compute for each sentence in the corpus its topic vector.
        It does it extracting the topics in the dataset with LDA, it checks the influence of each topic
        in each sentence and compute the topic of each sentence with a weighted sum over the topics. \n
        :return: array of dim (sentence_number, word_count in the model vocabulary)
        """
        from src.topic_modelling.lda_topic import train_specific_LDA, get_corpus, LDA

        # Create LDA model for the dataset, given parameters
        topic_num = 3
        coco = True # Now it is just coco or not coco just for name saving reasons, it's already possible to integrate any dataset in the data-dir folder
        corpus_raw = get_corpus(coco)
        lda = train_specific_LDA(corpus_raw, num_top=topic_num, passes=2, iterations=2, chunksize=2000, coco=coco)

        # Get percentage of each topic in each sentence
        df = get_perc_sent_topic(ldamodel=lda.lda_model, corpus=lda.corpus_bow, texts=lda.corpus_text, stops=lda.stops)
        topic_matrix = lda.lda_model.get_topics()  # num_topic x num_words
        topic_weights = df.values[:, 1:topic_num + 1]  # num_sentences x num_topic (each row sum to 1)
        sentences_number = topic_weights.shape[0]
        word_count = topic_matrix.shape[1]
        size = (int(sentences_number), int(word_count))
        topic_sentences = np.zeros(
            size)  # contains the word influence for each sentence, considering the topic of the sentence
        print("Topic sentnece dim: {}".format(size))

        for index in range(topic_sentences.shape[0]):
            # multiply the percentage of the topic in the sentence with the topic itself
            topic_sentences[index] = np.dot(np.expand_dims(topic_weights[index], 0), topic_matrix).squeeze()

        # get model lemmatized version of the words, it's needed because LDA does it and model processing doesn't
        texts = [str(lemmatize(word, min_length=2)).split('/')[0].split("b\'")[-1] for word in
                 self.model_word_index_dict.keys()]
        lda_index_word_dict = lda.dictionary.id2token

        def get_model_index(lda_index):
            word = lda_index_word_dict[lda_index]
            try:
                return self.model_word_index_dict[word]
            except KeyError:
                return [text_index for text_index in range(len(texts)) if texts[text_index] == word]

        real_vector = np.zeros((topic_sentences.shape[0], len(self.model_word_index_dict)))
        inverse_indexes = [get_model_index(i) for i in range(len(lda_index_word_dict))]

        # since the parallelism is the same for each sentence, it is done word by word for all sentences all together.
        # It is possible that a word in the LDA corresponds to more words in the model due to lemmatization procedure
        for ind, invere_index in enumerate(inverse_indexes):
            if isinstance(invere_index, list):
                # more than one index in the model because of lemmatization
                for x in invere_index:
                    real_vector[:, x] = topic_sentences[:, ind]
            else:
                invere_index = int(invere_index)
                real_vector[:, invere_index] = topic_sentences[:, ind]

        return real_vector
