
import re

import gensim

from gensim.models import CoherenceModel, LdaModel

from gensim.corpora import Dictionary
from gensim.utils import lemmatize



def process_texts(input_texts):
    final = []
    for i in input_texts:
        texts = (re.sub((r"http\S+"), "", i))
        # tokenize
        texts = gensim.utils.simple_tokenize(texts)
        # lower case
        texts = [word.lower() for word in texts]
        # remove stopwords
        texts = [word for word in texts if word not in stops]

        # automatically detect common phrases
        texts = [str(word).split('/')[0].split('b\'')[1] for word in lemmatize(' '.join(texts), allowed_tags=re.compile('(NN)'), min_length=3)]

        final.append(texts)
    return final


def evaluate_num_topics(dictionary, corpus, texts, limit, passes, iterations, random_state):
    c_v = []
    c_uci = []
    u_mass = []
    perplexity = []
    lm_list = []
    for num_top in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_top, id2word=dictionary, eval_every=1, \
                      passes=passes, iterations=iterations, random_state=random_state)
        lm_list.append(lm)
        cm_cv = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm_cv.get_coherence())

    # Show graph
    return lm_list, c_v



def create_LDA_model(texts, limit, chunksize, iterations, passes, random_state_lda):
    tmp = process_texts(texts)
    dictionary = Dictionary(tmp)
    corpus = [dictionary.doc2bow(i) for i in tmp]
    lm_list, c_v = evaluate_num_topics(dictionary, corpus, tmp, limit, chunksize, iterations, random_state_lda)
    num_top = c_v.index(max(c_v)) + 1
    lda_train = LdaModel(corpus=corpus, num_topics=num_top, id2word=dictionary, \
                         eval_every=1, passes=passes, chunksize=chunksize, \
                         iterations=iterations, random_state=random_state_lda)
    return lda_train