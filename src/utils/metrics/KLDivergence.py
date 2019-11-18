import json
import time
from abc import abstractmethod
from typing import List, Tuple

from numpy import mean
from scipy.stats import entropy, wasserstein_distance

from real.real_gan.loaders.real_loader import RealDataTopicLoader
from utils.metrics.Metrics import Metrics


def read_json_file(json_file):
    try:
        with open(json_file) as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = None
    return data


class Divergence(Metrics):
    def __init__(self, data_loader: RealDataTopicLoader, json_file: str, name='doc_embsim'):
        super().__init__()
        self.name = name
        self.data_loader = data_loader
        self.json_file = json_file

    def get_score(self):
        return self.computeKL()

    def computeKL(self):
        # print("Inizio KL")
        # t = time.time()
        real_sentences, generated_sentences = self.get_sentences()
        real_sentences_topic = self.data_loader.get_topic(real_sentences)
        generated_sentences_topic = self.data_loader.get_topic(generated_sentences)

        result = []
        for real, generated in zip(real_sentences_topic, generated_sentences_topic):
            r = self.compute_divergence(real, generated)
            result.append(r)
            # earth_result.append(wasserstein_distance(real, generated))
        # print("Fine KL: KL {}, Weist, time: {}".format(mean(result), mean(earth_result), time.time() - t))
        return mean(result)

    def get_sentences(self) -> Tuple[List[str], List[str]]:
        real_sentences, generated_sentences = [], []
        data = read_json_file(self.json_file)
        data = data['sentences'][:400]
        for el in data:
            real_sentences.append(el['real_starting'])
            generated_sentences.append(el['generated_sentence'])

        return real_sentences, generated_sentences

    @abstractmethod
    def compute_divergence(self, real, generated):
        raise NotImplementedError


class KL_divergence(Divergence):
    def __init__(self, data_loader: RealDataTopicLoader, json_file: str, name=''):
        super().__init__(data_loader, json_file, name)

    def compute_divergence(self, real, generated):
        return entropy(real, generated)


class EarthMover(Divergence):
    def __init__(self, data_loader: RealDataTopicLoader, json_file: str, name=''):
        super().__init__(data_loader, json_file, name)

    def compute_divergence(self, real, generated):
        return wasserstein_distance(real, generated)
