import sys

from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tqdm import tqdm

from real.real_gan.loaders.custom_reviews_loader import RealDataCustomerReviewsLoader
from utils.models.relational_memory import RelationalMemory
from utils.ops import *
import numpy as np


class ReviewGenerator:
    def __init__(self, x_real, temperature, x_sentiment, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots,
                 head_size,
                 num_heads, hidden_dim, start_token, sentiment_num, **kwargs):
        self.generated_num = None
        self.x_real = x_real
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
        output_memory_size = mem_slots * head_size * num_heads
        self.temperature = temperature
        self.x_sentiment = x_sentiment

        self.g_embeddings = tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                            initializer=create_linear_initializer(vocab_size))
        self.gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
        self.g_output_unit = create_output_unit(output_memory_size, vocab_size)

        # managing of attributes
        self.g_sentiment = linear(input_=tf.one_hot(self.x_sentiment, sentiment_num), output_size=gen_emb_dim,
                                  use_bias=True,
                                  scope="linear_x_sentiment")

        # self_attention_unit = create_self_attention_unit(scope="attribute_self_attention") #todo

        # initial states
        self.init_states = self.gen_mem.initial_state(batch_size)

        # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
        self.gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
        self.gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)
        self.gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                             infer_shape=True)
        self.pretrain_loss = None
        self.generate_recurrence_graph()
        self.generate_pretrain()

    def generate_recurrence_graph(self):
        def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
            mem_o_t, h_t = self.gen_mem(x_t,
                                        h_tm1)  # hidden_memory_tuple, output della memoria che si potrebbe riutilizzare
            mem_o_t, h_t = self.gen_mem(self.g_sentiment, h_t)
            # mem_o_t, h_t = gen_mem(self_attention_unit(), h_t) # todo
            o_t = self.g_output_unit(mem_o_t)  # batch x vocab, logits not prob

            gumbel_t = add_gumbel(o_t)
            next_token = tf.cast(tf.argmax(gumbel_t, axis=1), tf.int32)
            x_onehot_appr = tf.nn.softmax(tf.multiply(gumbel_t, self.temperature, name="gumbel_x_temp"),
                                          name="softmax_gumbel_temp")  # one-hot-like, [batch_size x vocab_size]

            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # embeddings, [batch_size x emb_dim]
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, [batch_size]
            gen_x_onehot_adv = gen_x_onehot_adv.write(i, x_onehot_appr)

            return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv

        _, _, _, gen_o, gen_x, gen_x_onehot_adv = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5: i < self.seq_len,
            body=_gen_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_tokens),
                       # todo si potrebbe pensare di modificare il primo input
                       self.init_states, self.gen_o, self.gen_x, self.gen_x_onehot_adv),
            name="while_adv_recurrence")

        gen_x = gen_x.stack()  # seq_len x batch_size
        self.gen_x = tf.transpose(gen_x, perm=[1, 0], name="gen_x_trans")  # batch_size x seq_len

        gen_o = gen_o.stack()
        self.gen_o = tf.transpose(gen_o, perm=[1, 0], name="gen_o_trans")

        gen_x_onehot_adv = gen_x_onehot_adv.stack()
        self.gen_x_onehot_adv = tf.transpose(gen_x_onehot_adv, perm=[1, 0, 2],
                                             name="gen_x_onehot_adv_trans")  # batch_size x seq_len x vocab_size

    def generate_pretrain(self):
        # ----------- pre-training for generator -----------------
        x_emb = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x_real), perm=[1, 0, 2],
                             name="input_embedding")  # seq_len x batch_size x emb_dim
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False,
                                                     infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len)
        ta_emb_x = ta_emb_x.unstack(x_emb)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            mem_o_t, h_t = self.gen_mem(x_t, h_tm1)
            mem_o_t, h_t = self.gen_mem(self.g_sentiment, h_t)
            o_t = self.g_output_unit(mem_o_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch_size x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_tokens),
                       self.init_states, g_predictions),
            name="while_pretrain")

        g_predictions = tf.transpose(g_predictions.stack(),
                                     perm=[1, 0, 2], name="g_predictions_trans")  # batch_size x seq_length x vocab_size

        # pretraining loss
        with tf.variable_scope("pretrain_loss_computation"):
            self.pretrain_loss = -tf.reduce_sum(
                tf.one_hot(tf.cast(tf.reshape(self.x_real, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
                )
            ) / (self.seq_len * self.batch_size)

    def pretrain_epoch(self, oracle_loader, sess, **kwargs):
        supervised_g_losses = []
        for it in range(oracle_loader.num_batch):
            sentiment, sentence = oracle_loader.next_batch()
            n = np.zeros((self.batch_size, self.seq_len))
            for ind, el in enumerate(sentence):
                n[ind] = el

            try:
                _ = kwargs['g_pretrain_op']
                _, g_loss = sess.run([kwargs['g_pretrain_op'], self.pretrain_loss], feed_dict={self.x_real: n,
                                                                                               self.x_sentiment: sentiment})
            except KeyError:
                g_loss = sess.run(self.pretrain_loss, feed_dict={self.x_real: n,
                                                                 self.x_sentiment: sentiment})

            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)

    def generate_json(self, oracle_loader: RealDataCustomerReviewsLoader, sess, **config):
        generated_samples, input_sentiment = [], []
        sentence_generated_from = []

        max_gen = int(self.generated_num / self.batch_size)  # - 155 # 156
        for ii in range(max_gen):
            sentiment, sentences = oracle_loader.random_batch()
            feed_dict = {self.x_sentiment: sentiment}
            sentence_generated_from.extend(sentences)
            gen_x_res = sess.run([self.gen_x], feed_dict=feed_dict)

            generated_samples.extend([x for a in gen_x_res for x in a])
            input_sentiment.extend(sentiment)

        json_file = {'sentences': []}
        for sent, input_sent in zip(generated_samples, input_sentiment):
            json_file['sentences'].append({
                'generated_sentence': " ".join([
                    oracle_loader.model_index_word_dict[str(el)] for el in sent if
                    el < len(oracle_loader.model_index_word_dict)]),
                'sentiment': input_sent
            })

        return json_file


class ReviewDiscriminator:
    def __init__(self, x_onehot, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn):
        self.sn = sn
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.x_onehot = x_onehot
        self.dis_emb_dim = dis_emb_dim
        self.vocab_size = vocab_size
        self.logits = self.create_graph()

    def create_graph(self):
        # Compute its embedding matrix
        d_embeddings = tf.get_variable('d_emb', shape=[self.vocab_size, self.dis_emb_dim],
                                       initializer=create_linear_initializer(self.vocab_size))
        input_x_re = tf.reshape(self.x_onehot, [-1, self.vocab_size], name="input_reshaping")
        # Multiply each input for its embedding matrix
        emb_x_re = tf.matmul(input_x_re, d_embeddings, name="input_x_embeddings")
        emb_x = tf.reshape(emb_x_re, [self.batch_size, self.seq_len, self.dis_emb_dim],
                           name="reshape_back")  # batch_size x seq_len x dis_emb_dim

        emb_x_expanded = tf.expand_dims(emb_x, 2, name="add_fake_dim_for_conv")  # batch_size x seq_len x 1 x emd_dim
        # convolution
        out = conv2d(emb_x_expanded, self.dis_emb_dim * 2, k_h=2, d_h=2, sn=self.sn, scope='conv1')
        out = tf.nn.relu(out)
        out = conv2d(out, self.dis_emb_dim * 2, k_h=1, d_h=1, sn=self.sn, scope='conv2')
        out = tf.nn.relu(out)

        # self-attention
        out = self_attention(out, self.dis_emb_dim * 2, sn=self.sn)

        # convolution
        out = conv2d(out, self.dis_emb_dim * 4, k_h=2, d_h=2, sn=self.sn, scope='conv3')
        out = tf.nn.relu(out)
        out = conv2d(out, self.dis_emb_dim * 4, k_h=1, d_h=1, sn=self.sn, scope='conv4')
        out = tf.nn.relu(out)

        # fc
        out = tf.contrib.layers.flatten(out, scope="flatten_output_layer")
        logits = linear(out, output_size=1, use_bias=True, sn=self.sn, scope='fc5')
        logits = tf.squeeze(logits, -1)  # batch_size

        return logits
