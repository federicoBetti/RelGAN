import numpy as np

from utils.metrics.Metrics import Metrics


class Nll(Metrics):
    def __init__(self, data_loader, pretrain_loss, x_real, sess, name='Nll'):
        super().__init__()
        self.name = name
        self.data_loader = data_loader
        self.sess = sess
        self.pretrain_loss = pretrain_loss
        self.x_real = x_real

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.nll_loss()

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        batch_num = self.data_loader.num_batch
        for it in range(batch_num):
            batch = self.data_loader.next_batch()
            g_loss = self.sess.run(self.pretrain_loss, {self.x_real: batch})
            nll.append(g_loss)
        return np.mean(nll)


class NllTopic(Nll):
    def __init__(self, data_loader, pretrain_loss, x_real, sess, x_topic, name='NllTopic'):
        super().__init__(data_loader, pretrain_loss, x_real, sess, name)
        self.x_topic = x_topic

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        batch_num = self.data_loader.num_batch
        for it in range(batch_num):
            text_batch, topic_batch = self.data_loader.next_batch(only_text=False)
            g_loss = self.sess.run(self.pretrain_loss, feed_dict={self.x_real: text_batch, self.x_topic: topic_batch})
            nll.append(g_loss)
        return np.mean(nll)


class NllAmazon(Nll):
    def __init__(self, oracle_loader, generator_obj, sess, name):
        super().__init__(oracle_loader, generator_obj.pretrain_loss, None, sess, name)
        self.generator_object = generator_obj

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        batch_num = self.data_loader.num_batch
        n = np.zeros((self.generator_object.batch_size, self.generator_object.seq_len))
        for it in range(batch_num):
            user, product, rating, sentence = self.data_loader.next_batch()
            for ind, el in enumerate(sentence):
                n[ind] = el

            g_loss = self.sess.run(self.pretrain_loss, feed_dict={self.generator_object.x_real: n,
                                                                  self.generator_object.x_user: user,
                                                                  self.generator_object.x_product: product,
                                                                  self.generator_object.x_rating: rating})

            nll.append(g_loss)
        return np.mean(nll)


class NllReview(Nll):
    def __init__(self, oracle_loader, generator_obj, sess, name):
        super().__init__(oracle_loader, generator_obj.pretrain_loss, None, sess, name)
        self.generator_object = generator_obj

    def nll_loss(self):
        return self.generator_object.pretrain_epoch(oracle_loader=self.data_loader, sess=self.sess)
