import os
import re
import time
from typing import List
import pandas as pd

from gensim.models import LdaModel, CoherenceModel
from gensim.utils import lemmatize, simple_tokenize


def process_texts(input_texts, stops):
    final = []
    for i in input_texts:
        texts = (re.sub(r"http\S+", "", i))
        # tokenize
        texts = simple_tokenize(texts)
        # lower case
        texts = [word.lower() for word in texts]
        # remove stopwords
        texts = [word for word in texts if word not in stops]

        # automatically detect common phrases
        texts = [str(word).split('/')[0].split('b\'')[1] for word in
                 lemmatize(' '.join(texts), allowed_tags=re.compile('(NN)'), min_length=3)]

        final.append(texts)
    return final


def evaluate_num_topics(dictionary, corpus, texts, limit, passes, iterations, random_state):
    c_v = []
    c_uci = []
    u_mass = []
    perplexity = []
    lm_list = []
    for num_top in range(1, limit):
        t = time.time()
        print("Check with {} topics".format(num_top), end=" ")
        lm = LdaModel(corpus=corpus, num_topics=num_top, id2word=dictionary, eval_every=1,
                      passes=passes, iterations=iterations, random_state=random_state)
        lm_list.append(lm)
        cm_cv = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm_cv.get_coherence())
        print("with coherence of {} in {} sec".format(cm_cv.get_coherence(), time.time() - t))

    # Show graph
    return lm_list, c_v


def get_corpus() -> List[str]:
    fname = os.path.join("..", "data", "image_coco.txt")
    with open(fname) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df
