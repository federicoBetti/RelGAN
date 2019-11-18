# --------------------------------------------------------------------------------
# Note that different from its original code https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py,
# we do not use "is_first" and "is_fast", because an issue exists otherwise for evaluating self-BLEU over training: Only
# in the first time of evaluation that the reference and hypothesis come from the same “test data” (i.e. the whole set
# of generated sentences). After that, the hypothesis keeps updated but the reference remains unchanged (due to
# “self.is_first=False”), which means hypothesis and reference are not from the same “test data” any more, and thus the
# scores obtained under that implementation is not self-BLEU scores.
# --------------------------------------------------------------------------------
import concurrent
import time
from itertools import repeat

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

from utils.metrics.Metrics import Metrics
from utils.static_file_manage import load_json


class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3, name='SelfBleu', portion=1):
        super().__init__()
        self.name = name
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 200  # SelfBLEU scores remain nearly unchanged for self.sample_size >= 200
        self.portion = portion  # how many posrtions to use in the evaluation, default to use the whole test dataset

    def get_name(self):
        return self.name

    def get_score(self, is_fast=False, ignore=False):
        if ignore:
            return 0

        return self.get_bleu()

    def get_reference(self):
        reference = list()
        json_obj = load_json(self.test_data)
        for i, hypothesis in enumerate(json_obj['sentences']):
            text = nltk.word_tokenize(hypothesis['generated_sentence'])
            reference.append(text)
        len_ref = len(reference)

        return reference[:self.sample_size]

    def get_bleu(self):
        t = time.time()
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        json_obj = load_json(self.test_data)
        # for i, hypothesis in enumerate(tqdm(json_obj['sentences'])):
        #     i = 0
        #     if i >= self.sample_size:
        #         break
        #     hypothesis = hypothesis['generated_sentence']
        #     hypothesis = nltk.word_tokenize(hypothesis)
        #     bleu.append(self.calc_bleu(reference, hypothesis, weight))
        #     i += 1

        index_considered = range(self.sample_size)
        # we can swap out ProcessPoolExecutor for ThreadPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for bleu_res in executor.map(procedure, index_considered, repeat(reference), repeat(weight)):
                bleu.append(bleu_res)

        # print("SelfBleu executed in {}".format(time.time() - t))
        return sum(bleu) / len(bleu)



def calc_bleu(hypothesis, reference, weight):
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)

def procedure(index, reference, weight):                 # just factoring out the
    # hypothesis, reference, weight = param[0], param[1], param[2]
    hypothesis = reference[index]
    other = reference[:index] + reference[index + 1:]
    return calc_bleu(hypothesis, other, weight)

