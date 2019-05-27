import os

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from path_resolution import resources_path
from real.real_gan.real_topic_train_utils import get_accuracy, get_train_ops, get_metric_summary_op, get_losses
from utils.text_process import code_to_text
from utils.utils import CustomSummary
# from translate import Translator
from googletrans import Translator
import numpy as np


def get_sentences(path):
    sentences = []
    with open(path, 'r') as outfile:
        for line in outfile:
            sentences.append(line)

    # translator = Translator()
    # sentences = [translator.translate(s, dest='en') for s in sentences]

    return sentences


def fix_size(x_topic, batch_size):
    x_topic_shape = x_topic.shape
    if x_topic_shape[0] > batch_size:
        return x_topic[:batch_size, :]
    else:
        fake_ret = np.zeros((batch_size, x_topic_shape[1]))
        fake_ret[:x_topic_shape[0], :] = x_topic
        return fake_ret


def inference_main(oracle_loader, config, model_path, input_path):
    tf.reset_default_graph()
    batch_size = config['batch_size']

    sentences = get_sentences(input_path)

    sent_number = len(sentences)
    topic_sentences = oracle_loader.get_topic(sentences)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                model_path,
            )

            x_topic = graph.get_tensor_by_name('x_topic:0')
            gen_x = graph.get_tensor_by_name("generator/gen_x_trans:0")

            print(topic_sentences[0])
            topic_sentences = fix_size(topic_sentences, batch_size)
            res = sess.run(gen_x, feed_dict={x_topic: topic_sentences})

    print("FINITO!!")
    for index in range(sent_number):
        sent = res[index]
        print(sent)
        print(code_to_text(codes=[sent], dictionary=oracle_loader.model_index_word_dict))


