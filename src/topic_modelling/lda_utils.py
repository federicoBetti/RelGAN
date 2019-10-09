import math
import re
import string
import time
from typing import List

import nltk
from nltk.corpus import wordnet
import pandas as pd
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_tokenize
import numpy as np

from path_resolution import resources_path

from tqdm import tqdm

from utils.static_file_manage import load_pickle, write_pickle


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def process_texts(input_texts, stops):
    """
    This function is processing the input text, removing stop words and lemmatizing it \n
    :param input_texts:
    :param stops: stop words that must be delated
    :return:
    """
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    final = []
    for i in tqdm(input_texts):
        # remove http and lower case
        texts = (re.sub(r"http\S+", "", i)).lower()
        # tokenize
        texts = nltk.word_tokenize(texts)

        # remove stopwords and lemmatize the sentence
        texts = [lemmatizer.lemmatize(word) for word in texts if word not in stops and word.isalpha()]

        # this is another version considering also postag, must be revised in case
        # final_t = []
        # for word, pos in nltk.pos_tag(texts):
        #     if word not in stops:
        #         final_t.append(lemmatizer.lemmatize(word, pos=pos))
        # texts = [lemmatizer.lemmatize(

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


def get_corpus(coco=True, datapath=None) -> List[str]:
    if datapath is None:
        if coco:
            fname = resources_path("data", "image_coco.txt")
        else:
            fname = resources_path("data", "emnlp_news.txt")
    else:
        fname = datapath

    with open(fname) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines


def format_topics_sentences(lda_model: LdaModel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list

        # row = sorted(row, key=lambda x: (x[1]), reverse=True) # sort list to get dominant topic
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        to_append = []
        for j, (topic_num, prop_topic) in enumerate(row):
            # if j == 0:  # => dominant topic
            #     wp = ldamodel.show_topic(topic_num)
            #     topic_keywords = ", ".join([word for word, prop in wp])
            #     sent_topics_df = sent_topics_df.append(
            #         pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            # else:
            #     break
            to_append.append(prop_topic)
        sent_topics_df = sent_topics_df.append(
            pd.Series(to_append), ignore_index=True)
    sent_topics_df.columns = ["Topic {}".format(topic_number) for topic_number in range(len(to_append))]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def word_cloud(lda):
    lda_model = lda.lda_model
    stop_words = lda.stops
    # 1. Wordcloud of Top N words in each topic
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud
    import matplotlib.colors as mcolors

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(math.ceil(lda.topic_num / 2), 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        try:
            topic_words = dict(topics[i][1])
        except IndexError:
            continue
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


def get_perc_few_sent_topic(ldamodel, corpus_bow, topic_num):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus_bow]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        to_append = np.zeros(topic_num)
        for j, (topic_n, prop_topic) in enumerate(row):
            to_append[topic_n] = prop_topic
        sent_topics_df = sent_topics_df.append(pd.Series(to_append), ignore_index=True)

    sent_topics_df.columns = ["Topic {}".format(topic_number) for topic_number in range(topic_num)]

    # Add original text to the end of the output
    sent_topics_df = sent_topics_df.reset_index()
    return sent_topics_df


def get_perc_sent_topic(lda, topic_num, data_file):
    file_path = "{}-{}-{}".format(lda.lda_model, topic_num, data_file[-10:])
    file_path = resources_path("topic_models", file_path)
    try:
        sent_topics_df = load_pickle(file_path)
    except FileNotFoundError:
        print("get perc sent topic not found")
        ldamodel = lda.lda_model
        with open(data_file) as f:
            sentences = [line.rstrip('\n') for line in f]

        tmp = process_texts(sentences, lda.stops)
        corpus_bow = [lda.dictionary.doc2bow(i) for i in tmp]
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(tqdm(ldamodel[corpus_bow])):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            # row = sorted(row, key=lambda x: (x[1]), reverse=True) # sort list to get dominant topic
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            to_append = np.zeros(topic_num)
            for j, (topic_n, prop_topic) in enumerate(row):
                to_append[topic_n] = prop_topic
            sent_topics_df = sent_topics_df.append(pd.Series(to_append), ignore_index=True)

        sent_topics_df.columns = ["Topic {}".format(topic_number) for topic_number in range(len(to_append))]

        # Add original text to the end of the output
        contents = pd.Series(sentences)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        sent_topics_df = sent_topics_df.reset_index()

        write_pickle(file_path, sent_topics_df)

    return sent_topics_df
