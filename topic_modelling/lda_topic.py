import pickle
from multiprocessing.spawn import freeze_support

from gensim.corpora import Dictionary
from nltk.corpus import stopwords

from topic_modelling.lda_utils import *


class LDA:
    def __init__(self, lda_train, corpus, stops):
        self.lda_model = lda_train
        self.corpus = corpus
        self.stops = stops

    def __str__(self):
        return "This is the class with this LDA model: {}".format(self.lda_model)


def create_LDA_model(texts, limit, chunksize, iterations, passes, random_state_lda, stops):
    tmp = process_texts(texts, stops)
    dictionary = Dictionary(tmp)
    corpus = [dictionary.doc2bow(i) for i in tmp]
    lm_list, c_v = evaluate_num_topics(dictionary, corpus, tmp, limit, passes, iterations, random_state_lda)
    num_top = c_v.index(max(c_v)) + 1
    lda_train = LdaModel(corpus=corpus, num_topics=num_top, id2word=dictionary,
                         eval_every=1, passes=passes,  # chunksize=chunksize,
                         iterations=iterations, random_state=random_state_lda)
    lda = LDA(lda_train=lda_train, corpus=corpus, stops=stops)
    return lda_train


def get_dominant_topic_and_contribution(lda_model, corpus, texts, stops):
    """
    get dominant topic for each document considering the ones in the lda model with the given corpus \n
    :param lda_model: lda model
    :param corpus: corpus on which lda is trained
    :param texts: text to evaluate the dominant topic on
    :param stops: stop words
    :return:
    """
    tmp = process_texts(texts, stops)
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=tmp)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(10)


def train_lda():
    stops = set(stopwords.words('english'))
    print("Stops: {}".format(stops))
    stops.add(u"amp")

    corpus = get_corpus()
    lda_train = create_LDA_model(corpus, limit=50, chunksize=2, iterations=2, passes=2, random_state_lda=3, stops=stops)
    print(lda_train)

    print("Computation finished")

    with open(os.path.join("topic_modelling", 'lda_model.pkl'), 'wb') as handle:
        pickle.dump(lda_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return lda_train


def use_lda():
    with open(os.path.join("topic_modelling", 'lda_model.pkl'), 'rb') as handle:
        lda_train = pickle.load(handle)

    return lda_train


if __name__ == '__main__':
    freeze_support()
    lda_train_result = train_lda()

    # lda_train_result = use_lda()
    # get_dominant_topic_and_contribution(lda_model=lda_train_result.lda_train, corpus=lda_train_result.corpus, stops=lda_train_result.stops)
