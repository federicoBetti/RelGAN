import sys

from tensorflow.python.ops import tensor_array_ops, control_flow_ops

from utils.models.relational_memory import RelationalMemory
from utils.ops import *


def generator(x_real, temperature, x_user, x_product, x_rating, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots,
              head_size,
              num_heads, hidden_dim, start_token, user_num, product_num, rating_num, **kwargs):
    start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
    output_memory_size = mem_slots * head_size * num_heads

    g_embeddings = tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
    g_output_unit = create_output_unit(output_memory_size, vocab_size)

    # managing of attributes
    g_user = linear(input_=tf.one_hot(x_user, user_num), output_size=gen_emb_dim, use_bias=True, scope="linear_x_user")
    g_product = linear(input_=tf.one_hot(x_product, product_num), output_size=gen_emb_dim, use_bias=True,
                       scope="linear_x_product")
    g_rating = linear(input_=tf.one_hot(x_rating, rating_num), output_size=gen_emb_dim, use_bias=True,
                      scope="linear_x_rating")
    g_attribute = linear(input_=tf.concat([g_user, g_product, g_rating], axis=1), output_size=gen_emb_dim,
                         use_bias=True, scope="linear_after_concat")

    # self_attention_unit = create_self_attention_unit(scope="attribute_self_attention") #todo

    # initial states
    init_states = gen_mem.initial_state(batch_size)

    # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
    gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                    infer_shape=True)
    topicness_values = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                    infer_shape=True)
    gen_x_no_lambda = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False,
                                                   infer_shape=True)

    def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)  # hidden_memory_tuple, output della memoria che si potrebbe riutilizzare
        mem_o_t, h_t = gen_mem(g_attribute, h_t)
        # mem_o_t, h_t = gen_mem(self_attention_unit(), h_t) # todo
        o_t = g_output_unit(mem_o_t)  # batch x vocab, logits not prob

        # print_op = tf.print("o_t shape", o_t.shape, ", o_t: ", o_t[0], output_stream=sys.stderr)

        gumbel_t = add_gumbel(o_t)
        next_token = tf.cast(tf.argmax(gumbel_t, axis=1), tf.int32)
        x_onehot_appr = tf.nn.softmax(tf.multiply(gumbel_t, temperature, name="gumbel_x_temp"),
                                      name="softmax_gumbel_temp")  # one-hot-like, [batch_size x vocab_size]

        x_tp1 = tf.nn.embedding_lookup(g_embeddings, next_token)  # embeddings, [batch_size x emb_dim]
        gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, vocab_size, 1.0, 0.0),
                                                         tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
        gen_x = gen_x.write(i, next_token)  # indices, [batch_size]
        gen_x_onehot_adv = gen_x_onehot_adv.write(i, x_onehot_appr)

        lambda_param = tf.zeros(batch_size)
        next_token_no_lambda = tf.cast(tf.argmax(o_t, axis=1), tf.int32)
        return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv

    _, _, _, gen_o, gen_x, gen_x_onehot_adv = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3, _4, _5: i < seq_len,
        body=_gen_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                   # todo si potrebbe pensare di modificare il primo input
                   init_states, gen_o, gen_x, gen_x_onehot_adv),
        name="while_adv_recurrence")

    gen_x = gen_x.stack()  # seq_len x batch_size
    gen_x = tf.transpose(gen_x, perm=[1, 0], name="gen_x_trans")  # batch_size x seq_len

    gen_o = gen_o.stack()
    gen_o = tf.transpose(gen_o, perm=[1, 0], name="gen_o_trans")

    gen_x_onehot_adv = gen_x_onehot_adv.stack()
    gen_x_onehot_adv = tf.transpose(gen_x_onehot_adv, perm=[1, 0, 2],
                                    name="gen_x_onehot_adv_trans")  # batch_size x seq_len x vocab_size

    topicness_values = topicness_values.stack()  # seq_len x batch_size
    topicness_values = tf.transpose(topicness_values, perm=[1, 0], name="lambda_values_trans")  # batch_size x seq_len

    gen_x_no_lambda = gen_x_no_lambda.stack()  # seq_len x batch_size
    gen_x_no_lambda = tf.transpose(gen_x_no_lambda, perm=[1, 0], name="gen_x_no_lambda_trans")  # batch_size x seq_len

    # ----------- pre-training for generator -----------------
    x_emb = tf.transpose(tf.nn.embedding_lookup(g_embeddings, x_real), perm=[1, 0, 2],
                         name="input_embedding")  # seq_len x batch_size x emb_dim
    g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)

    ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len)
    ta_emb_x = ta_emb_x.unstack(x_emb)

    def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)
        mem_o_t, h_t = gen_mem(g_attribute, h_t)
        o_t = g_output_unit(mem_o_t)
        g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch_size x vocab_size
        x_tp1 = ta_emb_x.read(i)
        return i + 1, x_tp1, h_t, g_predictions

    _, _, _, g_predictions = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3: i < seq_len,
        body=_pretrain_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                   init_states, g_predictions),
        name="while_pretrain")

    g_predictions = tf.transpose(g_predictions.stack(),
                                 perm=[1, 0, 2], name="g_predictions_trans")  # batch_size x seq_length x vocab_size

    # pretraining loss
    with tf.variable_scope("pretrain_loss_computation"):
        pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.cast(tf.reshape(x_real, [-1]), tf.int32), vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(g_predictions, [-1, vocab_size]), 1e-20, 1.0)
            )
        ) / (seq_len * batch_size)

    return gen_x_onehot_adv, gen_x, pretrain_loss, gen_o


def discriminator(x_onehot, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn):
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
