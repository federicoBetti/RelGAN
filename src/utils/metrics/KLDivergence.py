from typing import Union, List

from sklearn.metrics import mutual_info_score
from numpy import mean

from real.real_gan.real_loader import RealDataTopicLoader
from utils.metrics.Metrics import Metrics


class KL_divergence(Metrics):

    def __init__(self, data_loader: RealDataTopicLoader, gen_file: str, name='doc_embsim'):
        super().__init__()
        self.name = name
        self.data_loader = data_loader
        self.gen_file = gen_file

    def get_score(self):
        return self.computeKL()

    def computeKL(self):
        real_sentences, generated_sentences = self.get_sentences()
        real_sentences_topic = self.data_loader.get_topic(real_sentences)
        generated_sentences_topic = self.data_loader.get_topic(generated_sentences)
        result = []
        for real, generated in zip(real_sentences_topic, generated_sentences_topic):
            result.append(mutual_info_score(real, generated))

        return mean(result)

    def get_sentences(self) -> Union[List[str], List[str]]:
        pass
