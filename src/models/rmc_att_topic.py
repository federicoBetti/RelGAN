import random
import sys

from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tqdm import tqdm

from utils.models.relational_memory import RelationalMemory
from utils.ops import *
import numpy as np


def get_sentence_from_index(sent, model_index_word_dict):
    return " ".join([model_index_word_dict[str(el)] for el in sent if el < len(model_index_word_dict)])


class generator:

    def __init__(self, x_real, temperature, x_topic, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots, head_size,
                 num_heads, hidden_dim, start_token, use_lambda=True, **kwargs):
        self.start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
        self.output_memory_size = mem_slots * head_size * num_heads
        self.seq_len = seq_len
        self.x_real = x_real
        self.x_topic = x_topic
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.topic_in_memory = kwargs["TopicInMemory"]
        self.no_topic = kwargs["NoTopic"]

        self.g_embeddings = tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                            initializer=create_linear_initializer(vocab_size))
        self.gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
        self.g_output_unit = create_output_unit(self.output_memory_size, vocab_size)
        self.g_topic_embedding = create_topic_embedding_unit(vocab_size, gen_emb_dim)
        self.g_output_unit_lambda = create_output_unit_lambda(output_size=1, input_size=self.output_memory_size,
                                                              additive_scope="_lambda", min_value=0.01)

        # initial states
        self.init_states = self.gen_mem.initial_state(batch_size)
        self.create_recurrent_adv()
        self.create_pretrain()

    def create_recurrent_adv(self):
        # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_len, dynamic_size=False, infer_shape=True)
        gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False,
                                                        infer_shape=True)
        topicness_values = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False,
                                                        infer_shape=True)
        gen_x_no_lambda = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_len, dynamic_size=False,
                                                       infer_shape=True)

        def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv, lambda_values, gen_x_no_lambda):
            mem_o_t, h_t = self.gen_mem(x_t, h_tm1)  # hidden_memory_tuple
            if self.topic_in_memory and not self.no_topic:
                mem_o_t, h_t = self.gen_mem(self.g_topic_embedding(self.x_topic), h_t)
            o_t = self.g_output_unit(mem_o_t)  # batch x vocab, logits not prob

            if not self.topic_in_memory and not self.kwargs["NoTopic"]:
                topic_vector = self.x_topic
                lambda_param = self.g_output_unit_lambda(mem_o_t)
                next_token_no_lambda = tf.cast(tf.argmax(o_t, axis=1), tf.int32)
                o_t = o_t + lambda_param * topic_vector
            else:
                lambda_param = tf.zeros(self.batch_size)
                next_token_no_lambda = tf.cast(tf.argmax(o_t, axis=1), tf.int32)

            gumbel_t = add_gumbel(o_t)

            next_token = tf.cast(tf.argmax(gumbel_t, axis=1), tf.int32)

            x_onehot_appr = tf.nn.softmax(tf.multiply(gumbel_t, self.temperature, name="gumbel_x_temp"),
                                          name="softmax_gumbel_temp")  # one-hot-like, [batch_size x vocab_size]
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # embeddings, [batch_size x emb_dim]
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, [batch_size]
            gen_x_onehot_adv = gen_x_onehot_adv.write(i, x_onehot_appr)

            lambda_values = lambda_values.write(i, tf.squeeze(lambda_param))
            gen_x_no_lambda = gen_x_no_lambda.write(i, tf.squeeze(next_token_no_lambda))
            return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv, lambda_values, gen_x_no_lambda

        _, _, _, gen_o, gen_x, gen_x_onehot_adv, topicness_values, gen_x_no_lambda = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7: i < self.seq_len,
            body=_gen_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_tokens),
                       self.init_states, gen_o, gen_x, gen_x_onehot_adv, topicness_values, gen_x_no_lambda),
            name="while_adv_recurrence")

        gen_x = gen_x.stack()  # seq_len x batch_size
        self.gen_x = tf.transpose(gen_x, perm=[1, 0], name="gen_x_trans")  # batch_size x seq_len

        gen_o = gen_o.stack()
        self.gen_o = tf.transpose(gen_o, perm=[1, 0], name="gen_o_trans")

        gen_x_onehot_adv = gen_x_onehot_adv.stack()
        self.gen_x_onehot_adv = tf.transpose(gen_x_onehot_adv, perm=[1, 0, 2],
                                             name="gen_x_onehot_adv_trans")  # batch_size x seq_len x vocab_size

        topicness_values = topicness_values.stack()  # seq_len x batch_size
        self.topicness_values = tf.transpose(topicness_values, perm=[1, 0],
                                             name="lambda_values_trans")  # batch_size x seq_len

        gen_x_no_lambda = gen_x_no_lambda.stack()  # seq_len x batch_size
        self.gen_x_no_lambda = tf.transpose(gen_x_no_lambda, perm=[1, 0],
                                            name="gen_x_no_lambda_trans")  # batch_size x seq_len

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
            if self.topic_in_memory and not self.no_topic:
                mem_o_t, h_t = self.gen_mem(self.g_topic_embedding(self.x_topic), h_t)
            o_t = self.g_output_unit(mem_o_t)
            if not self.topic_in_memory and not self.no_topic:
                lambda_param = self.g_output_unit_lambda(mem_o_t)
                o_t = o_t + lambda_param * self.x_topic
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch_size x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_tokens),
                       self.init_states, g_predictions),
            name="while_pretrain")

        self.g_predictions = tf.transpose(g_predictions.stack(),
                                          perm=[1, 0, 2],
                                          name="g_predictions_trans")  # batch_size x seq_length x vocab_size

        # pretraining loss
        with tf.variable_scope("pretrain_loss_computation"):
            self.pretrain_loss = -tf.reduce_sum(
                tf.one_hot(tf.cast(tf.reshape(self.x_real, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
                )
            ) / (self.seq_len * self.batch_size)

    def pretrain_epoch(self, sess, oracle_loader, **kwargs):
        supervised_g_losses = []
        oracle_loader.reset_pointer()

        for it in tqdm(range(oracle_loader.num_batch)):
            text_batch, topic_batch = oracle_loader.next_batch(only_text=False)
            _, g_loss = sess.run([kwargs['g_pretrain_op'], self.pretrain_loss],
                                 feed_dict={self.x_real: text_batch, self.x_topic: topic_batch})
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)

    def generate_samples_topic(self, sess, generated_num, oracle_loader):
        generated_samples = []
        generated_samples_lambda = []
        sentence_generated_from = []
        generated_samples_no_lambda_words = []

        max_gen = int(generated_num / self.batch_size)  # - 155 # 156
        for ii in range(max_gen):
            if self.no_topic:
                gen_x_res = sess.run(self.gen_x)

            else:
                text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
                feed = {self.x_topic: topic_batch}
                sentence_generated_from.extend(text_batch)
                if self.topic_in_memory:
                    gen_x_res = sess.run(self.gen_x, feed_dict=feed)
                else:
                    gen_x_res, lambda_values_res, gen_x_no_lambda_res = sess.run(
                        [self.gen_x, self.topicness_values, self.gen_x_no_lambda],
                        feed_dict=feed)
                    generated_samples_lambda.extend(lambda_values_res)
                    generated_samples_no_lambda_words.extend(gen_x_no_lambda_res)

            generated_samples.extend(gen_x_res)

        codes = ""
        codes_with_lambda = ""
        json_file = {'sentences': []}
        if self.no_topic:
            for sent in generated_samples:
                json_file['sentences'].append({
                    'generated_sentence': get_sentence_from_index(sent, oracle_loader.model_index_word_dict)
                })
        else:
            if self.topic_in_memory:
                for sent, start_sentence in zip(generated_samples, sentence_generated_from):
                    json_file['sentences'].append({
                        'real_starting': get_sentence_from_index(start_sentence, oracle_loader.model_index_word_dict),
                        'generated_sentence': get_sentence_from_index(sent, oracle_loader.model_index_word_dict)
                    })
            else:
                for sent, lambda_value_sent, no_lambda_words, start_sentence in zip(generated_samples,
                                                                                    generated_samples_lambda,
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
                        'real_starting': get_sentence_from_index(start_sentence, oracle_loader.model_index_word_dict),
                        'generated_sentence': get_sentence_from_index(sent, oracle_loader.model_index_word_dict)
                    })

        return json_file

    def get_sentences(self, json_object):
        sentences = json_object['sentences']
        sent_number = 10
        sent = random.sample(sentences, sent_number)
        all_sentences = []
        for s in sent:
            if self.no_topic:
                all_sentences.append("{}".format(s['generated_sentence']))
            else:
                if self.topic_in_memory:
                    all_sentences.append("{} --- {}".format(str(s['generated_sentence']), s['real_starting']))
                else:
                    word_with_no_lambda = []
                    for letter in sent['generated']:
                        generated_word, real_word = letter['word_text'], letter['no_lambda_word']
                        if generated_word:
                            word_with_no_lambda.append(
                                "{} ({}, {})".format(generated_word, letter['lambda'], real_word))
                    word_with_no_lambda = " ".join(word_with_no_lambda)
                    all_sentences.append("{} ---- {} ---- {}".format(s['generated_sentence'], word_with_no_lambda, s['real_starting']))
        return all_sentences


class discriminator:
    def __init__(self, x_onehot, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn):
        self.x_onehot = x_onehot
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
        logits = linear(out, output_size=1, use_bias=True, sn=sn, scope='fc5')
        self.logits = tf.squeeze(logits, -1)  # batch_size


class topic_discriminator:
    def __init__(self, x_onehot, x_topic, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn, discriminator):
        self.x_onehot = x_onehot
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
        self.logits = tf.sigmoid(logits)


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
