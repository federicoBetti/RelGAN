import concurrent.futures
import multiprocessing
import random
import time
from itertools import repeat

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

from utils.metrics.Metrics import Metrics
from utils.static_file_manage import load_json


class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, name='Bleu', portion=1):
        super().__init__()
        self.name = name
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = 200  # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.reference = None
        self.is_first = True
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_name(self):
        return self.name

    def get_score(self, is_fast=False, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        return self.get_bleu()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(tuple(text))

            # randomly choose a portion of test data
            # In-place shuffle
            random.shuffle(reference)
            len_ref = len(reference)
            reference = reference[:int(self.portion * len_ref)]

            self.reference = tuple(reference)

            return self.reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        json_obj = load_json(self.test_data)
        t = time.time()
        # for i, hypothesis in enumerate(json_obj['sentences']):
        #     hypothesis = hypothesis['generated_sentence']
        #     if i >= self.sample_size:
        #         break
        #     hypothesis = nltk.word_tokenize(hypothesis)
        #     bleu.append(calc_bleu(reference, hypothesis, weight))
        #     i += 1
        # return sum(bleu) / len(bleu)
        json_cropped = json_obj['sentences'][:self.sample_size]
        # we can swap out ProcessPoolExecutor for ThreadPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for bleu_res in executor.map(procedure, json_cropped, repeat(reference), repeat(weight)):
                bleu.append(bleu_res)
        # print("Bleu executed in {}".format(time.time() - t))
        return sum(bleu) / len(bleu)



def calc_stuff(parameter=None):  # these are examples.
    sleep_time = random.choice([0, 1, 2, 3, 4, 5])
    time.sleep(sleep_time)
    return parameter / 2, sleep_time, parameter * parameter


def procedure(hypothesis, reference, weight):  # just factoring out the
    # hypothesis, reference, weight = param[0], param[1], param[2]
    hypothesis = hypothesis['generated_sentence']
    hypothesis = nltk.word_tokenize(hypothesis)
    # reference = [list(ref) for ref in reference]
    r = calc_bleu(reference, hypothesis, weight)
    return r


def main():
    output1 = list()
    output2 = list()
    output3 = list()
    start = time.time()  # let's see how long this takes

    # we can swap out ProcessPoolExecutor for ThreadPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for bleu_res in executor.map(procedure, json_obj['sentences']):
            # put results into correct output list
            output1.append(out1)
            output2.append(out2)
            output3.append(out3)


class BleuAmazon(Metrics):

    def __init__(self, name, json_file, gram):
        super().__init__()
        self.name = name
        self.json_file = json_file
        self.gram = gram
        self.sample_size = 200

    def get_score(self):
        ngram = self.gram
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        json_senteces = load_json(self.json_file)
        for ind, sentences in enumerate(json_senteces['sentences']):
            if ind > self.sample_size:
                break
            generated, ground_through = sentences['generated_sentence'], sentences['real_starting']
            bleu.append(calc_bleu(ground_through, generated, weight))

        return sum(bleu) / len(bleu)


def calc_bleu(reference, hypothesis, weight):
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)
