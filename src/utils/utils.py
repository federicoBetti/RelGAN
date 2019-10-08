import argparse
import random
import time

import numpy as np
import tensorflow as tf
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pprint
from utils.text_process import *

pp = pprint.PrettyPrinter()


def generate_samples(sess, gen_x, batch_size, generated_num, output_file=None,
                     get_code=True):
    # Generate Samples
    print("Generating Samples...")
    generated_samples = []
    max_gen = int(generated_num / batch_size)  # - 155  # 156
    for ii in range(max_gen):
        generated_samples.extend(sess.run(gen_x))
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


def generate_samples_topic(sess, gen_x, batch_size, generated_num, lambda_values=None, oracle_loader=None,
                           gen_x_no_lambda=None, x_topic=None, get_code=True):
    generated_samples = []
    generated_samples_lambda = []
    sentence_generated_from = []
    generated_samples_no_lambda_words = []

    max_gen = int(generated_num / batch_size)  # - 155 # 156
    for ii in range(max_gen):
        text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
        feed = {x_topic: topic_batch}
        sentence_generated_from.extend(text_batch)
        gen_x_res, lambda_values_res, gen_x_no_lambda_res = sess.run([gen_x, lambda_values, gen_x_no_lambda],
                                                                     feed_dict=feed)

        assert len(gen_x_res) == len(lambda_values_res)
        assert len(gen_x_res[0]) == len(lambda_values_res[0])
        generated_samples.extend(gen_x_res)
        generated_samples_lambda.extend(lambda_values_res)
        generated_samples_no_lambda_words.extend(gen_x_no_lambda_res)

    codes = ""
    codes_with_lambda = ""
    json_file = {'sentences': []}
    for sent, lambda_value_sent, no_lambda_words, start_sentence in zip(generated_samples, generated_samples_lambda,
                                                                        generated_samples_no_lambda_words,
                                                                        sentence_generated_from):
        sent_json = []
        for x, y, z in zip(sent, lambda_value_sent, no_lambda_words):
            sent_json.append({
                'word_code': int(x),
                'word_text': '' if x == len(oracle_loader.model_index_word_dict) else
                oracle_loader.model_index_word_dict[str(x)],
                'lambda': float(y),
                'no_lambda_word': '' if z == len(oracle_loader.model_index_word_dict) else
                oracle_loader.model_index_word_dict[str(z)]
            })
            codes_with_lambda += "{} ({:.4f};{}) ".format(x, y, z)
            codes += "{} ".format(x)
        json_file['sentences'].append({
            'generated': sent_json,
            'real_starting': " ".join([
                oracle_loader.model_index_word_dict[str(el)] for el in start_sentence if
                el < len(oracle_loader.model_index_word_dict)]),
            'generated_sentence': " ".join([
                oracle_loader.model_index_word_dict[str(el)] for el in sent if
                el < len(oracle_loader.model_index_word_dict)])
        })

        codes_with_lambda = codes_with_lambda[:-1] + '\n'
        codes = codes[:-1] + '\n'
        # print(codes_with_lambda)
        # print(json_file['sentences'][-1])

    return codes_with_lambda, sentence_generated_from, codes, json_file


def generate_amazon(sess, gen_x, batch_size, generated_num, oracle_loader=None, **tensors):
    generated_samples = []
    sentence_generated_from = []

    max_gen = int(generated_num / batch_size)  # - 155 # 156
    for ii in range(max_gen):
        user, product, rating, sentences = oracle_loader.random_batch(dataset=tensors['dataset'])
        feed_dict = {tensors['x_user']: user,
                     tensors['x_product']: product,
                     tensors['x_rating']: rating}
        sentence_generated_from.extend(sentences)
        gen_x_res = sess.run([gen_x], feed_dict=feed_dict)

        generated_samples.extend([x for a in gen_x_res for x in a])

    json_file = {'sentences': []}
    for sent, start_sentence in zip(generated_samples, sentence_generated_from):
        json_file['sentences'].append({
            'real_starting': " ".join([
                oracle_loader.model_index_word_dict[str(el)] for el in start_sentence if
                el < len(oracle_loader.model_index_word_dict)]),
            'generated_sentence': " ".join([
                oracle_loader.model_index_word_dict[str(el)] for el in sent if
                el < len(oracle_loader.model_index_word_dict)])
        })

    return json_file


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    return sess


def pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, data_loader, x_topic=None):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in tqdm(range(data_loader.num_batch)):
        if x_topic is not None:
            text_batch, topic_batch = data_loader.next_batch(only_text=False)
            _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: text_batch, x_topic: topic_batch})
        else:
            text_batch = data_loader.next_batch()
            _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: text_batch})

        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_discriminator(sess, d_topic_op, d_topic_loss, d_topic_accuracy, x_real, x_topic, x_topic_random,
                            data_loader, d_topic_out_real_pos, d_topic_out_real_neg):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    supervised_accuracy = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        text_batch, topic_batch = data_loader.next_batch(only_text=False)
        _, topic_loss, accuracy = sess.run([d_topic_op, d_topic_loss, d_topic_accuracy],
                                           feed_dict={x_real: text_batch, x_topic: topic_batch,
                                                      x_topic_random: data_loader.random_topic()})
        supervised_g_losses.append(topic_loss)
        supervised_accuracy.append(accuracy)

    return np.mean(supervised_g_losses), np.mean(supervised_accuracy)


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


def gen_real_test_file_not_file(codes: str, sentence_generated_from, file, iw_dict, json_file,
                                generator_sentences=False):
    """
    Save in the file in this format:
    'sentence generated + Taken from: sentence from which the topic was taken' \n
    :param codes: codes of the text generated by the generator, it is a long string with \n
    :param sentence_generated_from: sentences from which the topic was taken
    :param file: file where to save things
    :param iw_dict: index to word dictionary to convert from codes to words
    :param generator_sentences: boolean, if to write the generator sentence from which the topic was taken
    """
    with open(file, 'w') as outfile:
        for sent in json_file['sentences']:
            outfile.write(sent['generated_sentence'] + "\n")
            if generator_sentences:
                word_with_no_lambda = []
                for letter in sent['generated']:
                    generated_word, real_word = letter['word_text'], letter['no_lambda_word']
                    if generated_word:
                        word_with_no_lambda.append("{} ({}, {})".format(generated_word, letter['lambda'], real_word))
                word_with_no_lambda = " ".join(word_with_no_lambda)
                outfile.write("\t ---- {} ---- {}".format(word_with_no_lambda, sent['real_starting']) + "\n")


def take_sentences(gen_text_file):
    strings = []
    with open(gen_text_file, 'r') as outfile:
        for line in outfile:
            strings.append(line)
    return strings


def take_sentences_json(json_object, first_elem='sentiment', second_elem='generated_sentence'):
    sentences = json_object['sentences']
    sent_number = 10
    sent = random.sample(sentences, sent_number)
    all_sentences = []
    for s in sent:
        to_print = "{}".format(str(s[first_elem]))
        if second_elem is not None:
            to_print += " --- {}".format(s[second_elem])
        all_sentences.append(to_print)
    return all_sentences


def take_sentences_attribute(json_object):
    sentences = json_object['sentences']
    sent_number = 10
    sent = random.sample(sentences, sent_number)
    all_sentences = []
    for s in sent:
        all_sentences.append("{} --- {}".format(str(s['generated_sentence']), s['real_starting']))
    return all_sentences


def take_sentences_topic(gen_text_file):
    """
    :param gen_text_file: file containing generated sentences
    :return: list of strings with generated sentences and sentences from which they have been taken from
    """
    strings, all_strings = [], []
    with open(gen_text_file, 'r') as outfile:
        for line in outfile:
            all_strings.append(line)
    for line, taken_from_line in zip(all_strings[0::2], all_strings[1::2]):
        strings.append(line + taken_from_line)
    return strings


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parameters_division():
    # count parameters
    variables_dict = {}
    for v in tf.trainable_variables():
        name_scope = v.name.split('/')
        d = variables_dict
        params_number = np.prod(v.get_shape().as_list())
        for name in name_scope:
            d[name] = d.get(name, {})
            d = d[name]
            d['total_param'] = d.get('total_param', 0) + params_number
    print("Total paramter number: {}".format(
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    return variables_dict


class CustomSummary(object):
    """
    This class is used to create custom summaries and does everything by itself
    """

    def __init__(self, name: str, scope: str, summary_type=tf.summary.scalar, item_type=tf.float32):
        self.name = name
        self.scope = scope
        self.summary_type = summary_type
        self.item_type = item_type

        self.placeholder = tf.placeholder(self.item_type, name=self.name + '_placeholder')
        self.final_summary = self.summary_type(self.scope + '/' + self.name, self.placeholder)

        self.file_writer = None
        self.sess = None

    def set_file_writer(self, sum_writer, sess):
        self.file_writer = sum_writer
        self.sess = sess

    def write_summary(self, value, epoch):
        if self.file_writer is None or self.sess is None:
            raise ValueError("file writer or session weren't defined")
        fake_summary = self.sess.run(self.final_summary, feed_dict={self.placeholder: value})
        self.file_writer.add_summary(fake_summary, epoch)
