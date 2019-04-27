import time

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pprint
from utils.text_process import *

pp = pprint.PrettyPrinter()


def generate_samples(sess, gen_x, batch_size, generated_num, output_file=None,
                     get_code=True):
    # Generate Samples
    print("Generating Samples...", end="  ")
    generated_samples = []
    max_gen = int(generated_num / batch_size)  # - 155  # 156
    for ii in range(max_gen):
        if ii % 50 == 0:
            print("generated {} over {}".format(ii, max_gen))
        generated_samples.extend(sess.run(gen_x))
    print("Samples Generated")
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for sent in generated_samples:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(sent)
        return np.array(codes)
    codes = ""
    for sent in generated_samples:
        buffer = ' '.join([str(x) for x in sent]) + '\n'
        codes += buffer

    return codes


def generate_samples_topic(sess, gen_x, batch_size, generated_num, oracle_loader=None, x_topic=None, output_file=None,
                           get_code=True):
    # Generate Samples
    print("Generating Samples with Topic...", end="  ")
    generated_samples = []
    sentence_generated_from = []
    max_gen = int(generated_num / batch_size) - 155  # 156
    for ii in range(max_gen):
        text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
        feed = {x_topic: topic_batch}
        sentence_generated_from.extend(text_batch)
        if ii % 50 == 0:
            print("generated {} over {}".format(ii, max_gen))
        generated_samples.extend(sess.run(gen_x, feed_dict=feed))
    print("Samples Generated")
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for sent in generated_samples:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(sent)
        return np.array(codes), sentence_generated_from
    codes = ""
    for sent in generated_samples:
        buffer = ' '.join([str(x) for x in sent]) + '\n'
        codes += buffer

    return codes, sentence_generated_from


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        if np.mod(it, 50) == 0:
            print("Trained the batch {} over {}".format(it, data_loader.num_batch))
        batch = data_loader.next_batch()
        _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: batch})
        supervised_g_losses.append(g_loss)

    tf.summary.scalar("loss/generator/pretrain_loss", tf.reduce_mean(supervised_g_losses))

    return np.mean(supervised_g_losses)
#
# def pre_train_discriminator(sess, data_loader):
#     # Pre-train the generator using MLE for one epoch
#     supervised_g_losses = []
#     data_loader.reset_pointer()
#
#     for it in range(data_loader.num_batch):
#         if np.mod(it, 50) == 0:
#             print("Trained the batch {} over {}".format(it, data_loader.num_batch))
#         batch = data_loader.next_batch()
#
#         text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
#         sess.run(d_train_op, feed_dict={x_real: text_batch, x_topic: topic_batch,
#                                         x_topic_random: oracle_loader.random_topic()}
#         _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: batch})
#         supervised_g_losses.append(g_loss)
#
#     tf.summary.scalar("loss/generator/pretrain_loss", tf.reduce_mean(supervised_g_losses))
#
#     return np.mean(supervised_g_losses)

def plot_csv(csv_file, pre_epoch_num, metrics, method):
    names = [str(i) for i in range(len(metrics) + 1)]
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=0, skip_footer=0, names=names)
    for idx in range(len(metrics)):
        metric_name = metrics[idx].get_name()
        plt.figure()
        plt.plot(data[names[0]], data[names[idx + 1]], color='r', label=method)
        plt.axvline(x=pre_epoch_num, color='k', linestyle='--')
        plt.xlabel('training epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plot_file = os.path.join(os.path.dirname(csv_file), '{}_{}.pdf'.format(method, metric_name))
        print(plot_file)
        plt.savefig(plot_file)


def get_oracle_file(data_file, oracle_file, seq_len):
    """
    It generates the oracle_file made by tokens and output the dictionary from tokens to real words \n
    :param data_file: real corpus file
    :param oracle_file: file path where to store tokens
    :param seq_len: max len of the sequence
    :return: dictionary from tokens to real words
    """
    tokens = get_tokenized(data_file)
    word_set = get_word_list(tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)
    with open(oracle_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return index_word_dict


def get_real_test_file(generator_file, gen_save_file, iw_dict):
    codes = get_tokenized(generator_file)
    with open(gen_save_file, 'w') as outfile:
        outfile.write(code_to_text(codes=codes, dictionary=iw_dict))


def gen_real_test_file_not_file(codes: str, sentence_generated_from, file, iw_dict):
    """
    Save in the file in this format:
    'sentence generated + Taken from: sentence from which the topic was taken' \n
    :param codes: codes of the text generated by the generator, it is a long string with \n
    :param sentence_generated_from: sentences from which the topic was taken
    :param file: file where to save things
    :param iw_dict: index to word dictionary to convert from codes to words
    """
    raw = codes.split('\n')[:-1]
    tokenized = []
    for text in raw:
        text = nltk.word_tokenize(text.lower())
        tokenized.append(text)
    assert len(tokenized) == len(sentence_generated_from), \
        "Codes and sentence generated from have different lengths: {} and {}".format(len(tokenized),
                                                                                     len(sentence_generated_from))
    with open(file, 'w') as outfile:
        for r, s in zip(tokenized, sentence_generated_from):
            outfile.write(code_to_text(codes=[r], dictionary=iw_dict))
            outfile.write("\t Taken from: {}".format(code_to_text(codes=[s], dictionary=iw_dict)))


def take_sentences(gen_text_file):
    strings = []
    with open(gen_text_file, 'r') as outfile:
        for line in outfile:
            strings.append(line)
    return strings


def take_sentences_topic(gen_text_file):
    strings, all_strings = [], []
    with open(gen_text_file, 'r') as outfile:
        for line in outfile:
            all_strings.append(line)
    for line, taken_from_line in zip(all_strings[0::2], all_strings[1::2]):
        strings.append(line + taken_from_line)
    return strings
