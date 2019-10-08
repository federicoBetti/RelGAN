import sys
import time

from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tqdm import tqdm
import numpy as np

from utils.models.relational_memory import RelationalMemory
from utils.ops import *


class AmazonGenerator:
    def __init__(self, x_real, temperature, x_user, x_product, x_rating, vocab_size, batch_size, seq_len, gen_emb_dim,
                 mem_slots,
                 head_size,
                 num_heads, hidden_dim, start_token, user_num, product_num, rating_num, **kwargs):
        self.generated_num = None
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.seq_len = seq_len
        self.gen_emb_dim = gen_emb_dim
        self.x_real = x_real
        self.start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
        self.x_user = x_user
        self.x_rating = x_rating
        self.x_product = x_product
        output_memory_size = mem_slots * head_size * num_heads

        self.g_embeddings = tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                            initializer=create_linear_initializer(vocab_size))
        self.gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
        self.g_output_unit = create_output_unit(output_memory_size, vocab_size)

        # managing of attributes
        self.g_user = linear(input_=tf.one_hot(x_user, user_num), output_size=gen_emb_dim, use_bias=True,
                             scope="linear_x_user")
        self.g_product = linear(input_=tf.one_hot(x_product, product_num), output_size=gen_emb_dim, use_bias=True,
                                scope="linear_x_product")
        self.g_rating = linear(input_=tf.one_hot(x_rating, rating_num), output_size=gen_emb_dim, use_bias=True,
                               scope="linear_x_rating")
        self.g_attribute = linear(input_=tf.concat([self.g_user, self.g_product, self.g_rating], axis=1),
                                  output_size=self.gen_emb_dim,
                                  use_bias=True, scope="linear_after_concat")

        # self_attention_unit = create_self_attention_unit(scope="attribute_self_attention") #todo

        # initial states
        self.init_states = self.gen_mem.initial_state(batch_size)
        self.create_recurrence()
        self.create_pretrain()

    def multihead_attention(self, attribute):
        """Perform multi-head attention from 'Attention is All You Need'.

        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.

        Args:
          memory: Memory tensor to perform attention on, with size [B, N, H*V].

        Returns:
          new_memory: New memory tensor.
        """
        key_size = 512
        head_size = 512
        num_heads = 2
        qkv_size = 2 * key_size + head_size
        total_size = qkv_size * num_heads  # Denote as F.
        batch_size = attribute.get_shape().as_list()[0]  # Denote as B
        qkv = linear(attribute, total_size, use_bias=False, scope='lin_qkv')  # [B*N, F]
        qkv = tf.reshape(qkv, [batch_size, -1, total_size])  # [B, N, F]
        qkv = tf.contrib.layers.layer_norm(qkv, trainable=True)  # [B, N, F]

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = tf.reshape(qkv, [batch_size, -1, num_heads, qkv_size])

        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(qkv_transpose, [key_size, key_size, head_size], -1)

        q *= qkv_size ** -0.5
        dot_product = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]
        weights = tf.nn.softmax(dot_product)

        output = tf.matmul(weights, v)  # [B, H, N, V]

        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])

        # [B, N, H, V] -> [B, N, H * V]
        attended_attribute = tf.reshape(output_transpose, [batch_size, -1])
        return attended_attribute

    def create_recurrence(self):
        # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_len, dynamic_size=False, infer_shape=True)
        gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False,
                                                        infer_shape=True)

        def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
            mem_o_t, h_t = self.gen_mem(x_t, h_tm1)  # hidden_memory_tuple
            mem_o_t, h_t = self.gen_mem(self.g_attribute, h_t)
            # mem_o_t, h_t = gen_mem(self_attention_unit(), h_t) # todo
            o_t = self.g_output_unit(mem_o_t)  # batch x vocab, logits not prob

            # print_op = tf.print("o_t shape", o_t.shape, ", o_t: ", o_t[0], output_stream=sys.stderr)

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
                       self.init_states, gen_o, gen_x, gen_x_onehot_adv),
            name="while_adv_recurrence")

        gen_x = gen_x.stack()  # seq_len x batch_size
        self.gen_x = tf.transpose(gen_x, perm=[1, 0], name="gen_x_trans")  # batch_size x seq_len

        gen_o = gen_o.stack()
        self.gen_o = tf.transpose(gen_o, perm=[1, 0], name="gen_o_trans")

        gen_x_onehot_adv = gen_x_onehot_adv.stack()
        self.gen_x_onehot_adv = tf.transpose(gen_x_onehot_adv, perm=[1, 0, 2],
                                             name="gen_x_onehot_adv_trans")  # batch_size x seq_len x vocab_size

    def create_pretrain(self):
        # ----------- pre-training for generator -----------------
        x_emb = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x_real), perm=[1, 0, 2],
                             name="input_embedding")  # seq_len x batch_size x emb_dim
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False,
                                                     infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len)
        ta_emb_x = ta_emb_x.unstack(x_emb)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            mem_o_t, h_t = self.gen_mem(x_t, h_tm1)
            mem_o_t, h_t = self.gen_mem(self.g_attribute, h_t)
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
        oracle_loader.reset_pointer()

        n = np.zeros((self.batch_size, self.seq_len))
        for it in tqdm(range(oracle_loader.num_batch)):
            # t = time.time()
            user, product, rating, sentence = oracle_loader.next_batch()
            # t1 = time.time()
            for ind, el in enumerate(sentence):
                n[ind] = el
            # t2 = time.time()
            _, g_loss = sess.run([kwargs['g_pretrain_op'], self.pretrain_loss], feed_dict={self.x_real: n,
                                                                              self.x_user: user,
                                                                              self.x_product: product,
                                                                              self.x_rating: rating})
            t3 = time.time()
            # print("Loader {}".format(t1 - t))
            # print("n: {}".format(t2 -t1))
            # print("pretrain: {}".format(t3 - t2))
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)

    def generate_samples(self, sess, oracle_loader, **tensors):
        generated_samples = []
        sentence_generated_from = []

        max_gen = int(self.generated_num / self.batch_size)  # - 155 # 156
        for ii in range(max_gen):
            user, product, rating, sentences = oracle_loader.random_batch(dataset=tensors['dataset'])
            feed_dict = {self.x_user: user,
                         self.x_product: product,
                         self.x_rating: rating}
            sentence_generated_from.extend(sentences)
            gen_x_res = sess.run([self.gen_x], feed_dict=feed_dict)

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


class AmazonDiscriminator:
    def __init__(self, x_onehot, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn):
        self.sn = sn
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dis_emb_dim = dis_emb_dim
        self.vocab_size = vocab_size
        self.x_onehot = x_onehot

        # Compute its embedding matrix
        self.logits = self.compute_logits()

    def compute_logits(self):
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

        # todo si potrebbe mettere qua e fare due output, uno che riguarda la frase in generale e uno che riguarda il topic
        return logits


def topic_discriminator(x_onehot, x_topic, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn, discriminator):
    # Compute its embedding matrix
    d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    input_x_re = tf.reshape(x_onehot, [-1, vocab_size], name="input_reshaping")
    # Multiply each input for its embedding matrix
    emb_x_re = tf.matmul(input_x_re, d_embeddings, name="input_x_embeddings")
    emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim],
                       name="reshape_back")  # batch_size x seq_len x dis_emb_dim

    emb_x_expanded = tf.expand_dims(emb_x, 2, name="add_fake_dim_for_conv")  # batch_size x seq_len x 1 x emd_dim
    # convolution
    out = conv2d(emb_x_expanded, dis_emb_dim * 2, k_h=2, d_h=2, sn=sn, scope='conv1')
    out = tf.nn.relu(out)
    out = conv2d(out, dis_emb_dim * 2, k_h=1, d_h=1, sn=sn, scope='conv2')
    out = tf.nn.relu(out)

    # self-attention
    out = self_attention(out, dis_emb_dim * 2, sn=sn)

    # convolution
    out = conv2d(out, dis_emb_dim * 4, k_h=2, d_h=2, sn=sn, scope='conv3')
    out = tf.nn.relu(out)
    out = conv2d(out, dis_emb_dim * 4, k_h=1, d_h=1, sn=sn, scope='conv4')
    out = tf.nn.relu(out)

    # fc
    out = tf.contrib.layers.flatten(out, scope="flatten_output_layer")

    # topic network
    emb_topic = tf.matmul(x_topic, d_embeddings, name="x_topic_embeddings")
    first_topic = linear(emb_topic, output_size=128, use_bias=True, sn=sn, scope='topic_first_linear')

    flatten = tf.concat([out, first_topic], axis=1)

    logits = linear(flatten, output_size=1, use_bias=True, sn=sn, scope='fc_topic')
    logits = tf.squeeze(logits, -1)  # batch_size
    logits = tf.sigmoid(logits)
    return logits


def topic_discriminator_reuse(x_onehot, x_topic, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn,
                              discriminator):
    _, out, d_embeddings = discriminator(x_onehot, True)

    # topic network
    emb_topic = tf.matmul(x_topic, d_embeddings, name="x_topic_embeddings")
    second_topic = linear(emb_topic, output_size=32, use_bias=True, sn=sn, scope='topic_second_linear')

    flatten = tf.concat([out, second_topic], axis=1)

    logits = linear(flatten, output_size=1, use_bias=True, sn=sn, scope='fc_topic')
    logits = tf.sigmoid(logits)
    logits = tf.squeeze(logits, -1)  # batch_size
    return logits
