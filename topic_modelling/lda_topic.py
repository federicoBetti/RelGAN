import pickle
from multiprocessing.spawn import freeze_support

from gensim.corpora import Dictionary
from nltk.corpus import stopwords

from topic_modelling.lda_utils import *


class LDA:
    def __init__(self, lda_train, corpus_text, corpus_bow, stops):
        self.lda_model = lda_train
        self.corpus_text = corpus_text
        self.corpus_bow = corpus_bow
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
    return df_dominant_topic.head(10)


def train_lda():
    stops = set(stopwords.words('english'))
    print("Stops: {}".format(stops))
    stops.add(u"amp")

    corpus = get_corpus()
    lda_train = create_LDA_model(corpus, limit=50, chunksize=2, iterations=2, passes=2, random_state_lda=3, stops=stops)
    print(lda_train)

    print("Computation finished")

    with open(os.path.join("topic_models", 'lda_model.pkl'), 'wb') as handle:
        pickle.dump(lda_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return lda_train


def use_lda(model_name='lda_model.pkl'):
    with open(os.path.join("topic_models", model_name), 'rb') as handle:
        lda_train = pickle.load(handle)

    return lda_train


def train_specific_LDA(corpus, num_top, passes, iterations, random_state_lda=3, chunksize=2000) -> LDA:
    '''
    create an LDA model if needed, if it has been already computed and saved, use that one \n
    :param corpus: corpus (plain text)
    :param num_top: topic number
    :param passes: passes
    :param iterations: iterations
    :param random_state_lda: random_state_lda
    :param chunksize: chunksize
    :return:
    '''
    try:
        with open(os.path.join("topic_models",
                               'lda_model_ntop_{}_iter_{}_pass_{}_chunk_{}.pkl'.format(num_top, iterations, passes,
                                                                                       chunksize)),
                  'rb') as handle:
            lda = pickle.load(handle)
        return lda
    except FileNotFoundError:
        print("No model found")
    stops = set(stopwords.words('english'))
    tmp = process_texts(corpus, stops)
    dictionary = Dictionary(tmp)
    corpus_bow = [dictionary.doc2bow(i) for i in tmp]
    lda_train = LdaModel(corpus=corpus_bow, num_topics=num_top, id2word=dictionary,
                         eval_every=1, passes=passes, chunksize=chunksize,
                         iterations=iterations, random_state=random_state_lda)
    lda = LDA(lda_train=lda_train, corpus_text=corpus, corpus_bow=corpus_bow, stops=stops)

    with open(os.path.join("topic_models",
                           'lda_model_ntop_{}_iter_{}_pass_{}.pkl'.format(num_top, iterations, passes)),
              'wb') as handle:
        pickle.dump(lda, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("New model saved")
    return lda


if __name__ == '__main__':
    freeze_support()
    # lda_train_result = train_lda()

    corpus = get_corpus()
    lda = train_specific_LDA(corpus, num_top=42, passes=2, iterations=2, chunksize=2000)

    # lda_train_result = use_lda()
    print("Texts: {}".format(lda.corpus_text[:10]))
    df = get_dominant_topic_and_contribution(lda_model=lda.lda_model, corpus=lda.corpus_bow, texts=lda.corpus_text[:10],
                                        stops=lda.stops)
    a = 1
