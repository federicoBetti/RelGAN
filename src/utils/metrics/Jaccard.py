from utils.metrics.Metrics import Metrics
from utils.static_file_manage import load_json
import numpy as np


class JaccardSimilarity(Metrics):
    def __init__(self, data_loader, json_file: str, name='doc_embsim'):
        super().__init__()
        self.name = name
        self.all_sentences = data_loader.get_all_sentences()
        self.json_file = json_file

    def get_score(self):
        return self.computeDistanceJaccard()

    def computeDistanceJaccard(self):
        generated_sentences = self.get_sentences()
        jaccard_values = []
        for generated_sentence in generated_sentences:
            values = []
            for real_sent in self.all_sentences:
                values.append(self.distJaccard(generated_sentence, real_sent))

            jaccard_values.append(1-max(values))


        return np.mean(jaccard_values)

    def distJaccard(self, str1, str2):
        str1 = set(str1.split())
        str2 = set(str2.split())
        return float(len(str1 & str2)) / len(str1 | str2)

    def get_sentences(self):
        generated_sentences = []
        data = load_json(self.json_file)
        for el in data['sentences']:
            generated_sentences.append(el['generated_sentence'])

        return generated_sentences
