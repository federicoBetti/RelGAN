import json
from typing import List, Tuple

from numpy import mean
from scipy.stats import entropy

from real.real_gan.real_loader import RealDataTopicLoader
from utils.metrics.Metrics import Metrics


def read_json_file(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
    return data


class KL_divergence(Metrics):
    def __init__(self, data_loader: RealDataTopicLoader, json_file: str, name='doc_embsim'):
        super().__init__()
        self.name = name
        self.data_loader = data_loader
        self.json_file = json_file

    def get_score(self):
        return self.computeKL()

    def computeKL(self):
        real_sentences, generated_sentences = self.get_sentences()
        real_sentences_topic = self.data_loader.get_topic(real_sentences)
        generated_sentences_topic = self.data_loader.get_topic(generated_sentences)

        result = []
        for real, generated in zip(real_sentences_topic, generated_sentences_topic):
            r = entropy(real, generated)
            result.append(r)

        return mean(result)

    def get_sentences(self) -> Tuple[List[str], List[str]]:
        real_sentences, generated_sentences = [], []
        data = read_json_file(self.json_file)
        for el in data['sentences']:
            real_sentences.append(el['real_starting'])
            generated_sentences.append(el['generated_sentence'])

        return real_sentences, generated_sentences
