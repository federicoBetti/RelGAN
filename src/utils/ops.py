import math
import tensorflow as tf
from tensorflow.compat.v1 import get_variable


def hw_flatten(x):
    """
    Remove the useless third dimension \n
    :param x: input tensor
    :return: input tensor with the third dim removed
    """
    return tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2], x.shape[-1]])


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def create_linear_initializer(input_size, dtype=tf.float32):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1 / math.sqrt(input_size * 1.0)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def create_bias_initializer(dtype=tf.float32):
    """Returns a default initializer for the biases of a linear/AddBias module."""
    return tf.zeros_initializer(dtype=dtype)


def linear(input_, output_size, use_bias=False, sn=False, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: Variable Scope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        W = get_variable("Matrix", shape=[output_size, input_size],
                         initializer=create_linear_initializer(input_size, input_.dtype),
                         dtype=input_.dtype)
        if sn:
            W = spectral_norm(W)
        output_ = tf.matmul(input_, tf.transpose(W))
        if use_bias:
            bias_term = get_variable("Bias", [output_size],
                                     initializer=create_bias_initializer(input_.dtype),
                                     dtype=input_.dtype)
            output_ += bias_term

    return output_


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def mlp(input_, output_sizes, act_func=tf.nn.relu, use_bias=True):
    '''
    Constructs a MLP module
    :param input_:
    :param output_sizes: An iterable of output dimensionalities
    :param act_func: activation function
    :param use_bias: whether use bias term for linear mapping
    :return: the output of the MLP module
    '''
    net = input_
    num_layers = len(output_sizes)
    for layer_id in range(num_layers):
        net = linear(net, output_sizes[layer_id], use_bias=use_bias, scope='linear_{}'.format(layer_id))
        if layer_id != num_layers - 1:
            net = act_func(net)
    return net


def conv2d(input_, out_nums, k_h=2, k_w=1, d_h=2, d_w=1, stddev=None, sn=False, padding='SAME', scope=None):
    """
    This function create a Conv2D layer using Glorot initialization.
    Usually width params are set to 1 since you want to do to Conv only between the same sentence \n
    :param input_: input tensor (batch_size x seq_len x 1 x channels_dim)
    :param out_nums: output channel dim
    :param k_h: filter height
    :param k_w: filter width (usually 1)
    :param d_h: stride height
    :param d_w: stride width (usually 1)
    :param stddev: usually None, so that the Glorot initialization is done
    :param sn:
    :param padding: padding type for Pooling procedure
    :param scope: scope name of the Conv2D layer
    :return: the Conv2D layer to added in the graph
    """
    in_nums = input_.get_shape().as_list()[-1]
    # Glorot initialization
    if stddev is None:
        stddev = math.sqrt(2. / (k_h * k_w * in_nums))
    with tf.variable_scope(scope or "Conv2d"):
        W = get_variable("Matrix", shape=[k_h, k_w, in_nums, out_nums],
                         initializer=tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            W = spectral_norm(W)
        b = get_variable("Bias", shape=[out_nums], initializer=tf.zeros_initializer)
        conv = tf.nn.conv2d(input_, filter=W, strides=[1, d_h, d_w, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)

    return conv


def self_attention(x, ch, sn=False, scope="conv_self_attention"):
    """
    'self-attention for GAN' \n
    :param x: input tensor (batch_size x sentence_len_remained x 1 x channels)
    :param ch: output channels dim
    :param sn:
    :param scope: scope name for the layer, useful for tensorboard
    :return:
    """
    with tf.variable_scope(scope):
        f = conv2d(x, ch // 8, k_h=1, d_h=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        g = conv2d(x, ch // 8, k_h=1, d_h=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = conv2d(x, ch, k_h=1, d_h=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True, name="self_attention_key_x_query")  # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1, name="self_attention_softmax")  # attention map

        o = tf.matmul(beta, hw_flatten(h), name="self_attention_prob_x_values")  # [bs, N, C]
        gamma = get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, [-1] + x.get_shape().as_list()[1:])  # [bs, h, w, C]
        x = tf.add(gamma * o, x, name="add_self_attention_input")

    return x


def spectral_norm(w, iteration=1):
    """spectral normalization for GANs"""
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def create_output_unit(output_size, vocab_size):
    # output_size = self.gen_mem.output_size.as_list()[0]
    Wo = get_variable('Wo', shape=[output_size, vocab_size], initializer=create_linear_initializer(output_size))
    bo = get_variable('bo', shape=[vocab_size], initializer=create_bias_initializer())

    def unit(hidden_mem_o):
        with tf.variable_scope("output_unit"):
            logits = tf.matmul(hidden_mem_o, Wo) + bo
            # logits = tf.nn.softmax(logits)  # so that they are positive todo fare qualcosa qua nel caso
        return logits

    return unit


def create_topic_embedding_unit(input_size, output_size):
    # output_size = self.gen_mem.output_size.as_list()[0]
    Wo = get_variable('W_topic_embedding', shape=[input_size, output_size],
                      initializer=create_linear_initializer(input_size))
    bo = get_variable('b_topic_embedding', shape=[output_size], initializer=create_bias_initializer())

    def unit(hidden_mem_o):
        with tf.variable_scope("output_unit_topic_embedding", reuse=tf.AUTO_REUSE):
            logits = tf.matmul(hidden_mem_o, Wo) + bo
            logits = tf.squeeze(attend_over_vector(tf.expand_dims(logits, 1)))
        return logits

    return unit


def create_output_unit_lambda(output_size, input_size, additive_scope="_lambda", min_value=0.01):
    """
    create a one-layer MLP with sigmoid in the end
    :param output_size: if lambda is a scalar it is one
    :param vocab_size: input size
    :param additive_scope:
    :return:
    """
    Wo = get_variable('W' + additive_scope, shape=[input_size, output_size],
                      initializer=create_linear_initializer(input_size))
    bo = get_variable('b' + additive_scope, shape=[output_size], initializer=create_bias_initializer())

    def unit(hidden_mem_o):
        with tf.variable_scope("output_unit" + additive_scope):
            logits = tf.matmul(hidden_mem_o, Wo) + bo
            # logits = tf.sigmoid(logits)
        return logits

    return unit


def multihead_attention(memory):
    """Perform multi-head attention from 'Attention is All You Need'.

    Implementation of the attention mechanism from
    https://arxiv.org/abs/1706.03762.

    Args:
      memory: Memory tensor to perform attention on, with size [B, N, H*V].

    Returns:
      new_memory: New memory tensor.
    """
    with tf.variable_scope("multihead_attention_vector"):
        head_size = 32
        key_size = head_size
        num_heads = 1
        mem_size = memory.shape[-1]
        qkv_size = 2 * key_size + head_size
        total_size = qkv_size * num_heads  # Denote as F.
        batch_size = memory.get_shape().as_list()[0]  # Denote as B
        memory_flattened = tf.reshape(memory, [-1, mem_size])  # [B * N, H * V]
        qkv = linear(memory_flattened, total_size, use_bias=False, scope='lin_qkv')  # [B*N, F]
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
        new_memory = tf.reshape(output_transpose, [batch_size, -1, mem_size])
        return new_memory


def attend_over_vector(vector):
    """Perform multiheaded attention over `memory`.

    Args:
      vector: input vector [batch_size, 1, dim] 

    Returns:
      The attended-over vector.
    """

    # Memoria 'modificata'
    mem_size = vector.shape[-1]
    attended_vector = multihead_attention(vector)  # [B, N, H * V]
    return attended_vector

    # Add a skip connection to the multiheaded attention's input.
    vector = tf.contrib.layers.layer_norm(vector + attended_vector, trainable=True)  # [B, N, H * V]

    # Add a mlp map
    batch_size = vector.get_shape().as_list()[0]

    memory_mlp = tf.reshape(vector, [-1, mem_size])  # [B * N, H * V]
    memory_mlp = mlp(memory_mlp, [mem_size] * 1)  # [B * N, H * V]
    memory_mlp = tf.reshape(memory_mlp, [batch_size, -1, mem_size])

    # Add a skip connection to the memory_mlp's input.
    vector = tf.contrib.layers.layer_norm(vector + memory_mlp, trainable=True)  # [B, N, H * V]

    return vector


def create_1D_self_attention_unit(scope, output_dim):
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

    def unit(query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


def add_gumbel(o_t, eps=1e-10):
    """
    Sample from Gumbel(0, 1)
    After some research and tests I discovered that it introdcues a noise with Final mean: 0.5862 and std: 1.29265
    This is a quite high std for that mean, and the added noise is high if compared to a softmax between 5k values
    that has mean of 0.0002!
    """
    with tf.variable_scope("Gumbel_softmax"):
        u = tf.random.uniform(tf.shape(o_t), minval=0, maxval=1, dtype=tf.float32)
        g_t = -tf.log(-tf.log(u + eps) + eps)
        gumbel_t = tf.add(o_t, g_t)
    return gumbel_t


def add_gumbel_cond(o_t, next_token_onehot, eps=1e-10):
    """draw reparameterization z of categorical variable b from p(z|b)."""

    def truncated_gumbel(gumbel, truncation):
        return -tf.log(eps + tf.exp(-gumbel) + tf.exp(-truncation))

    v = tf.random.uniform(tf.shape(o_t), minval=0, maxval=1, dtype=tf.float32)

    print("shape of v: {}".format(v.get_shape().as_list()))
    print("shape of next_token_onehot: {}".format(next_token_onehot.get_shape().as_list()))

    gumbel = -tf.log(-tf.log(v + eps) + eps, name="gumbel")
    topgumbels = gumbel + tf.reduce_logsumexp(o_t, axis=-1, keep_dims=True)
    topgumbel = tf.reduce_sum(next_token_onehot * topgumbels, axis=-1, keep_dims=True)

    truncgumbel = truncated_gumbel(gumbel + o_t, topgumbel)
    return (1. - next_token_onehot) * truncgumbel + next_token_onehot * topgumbels


def gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config):
    """compute the gradiet penalty for the WGAN-GP loss"""
    alpha = tf.random.uniform(shape=[config['batch_size'], 1, 1], minval=0., maxval=1.)
    interpolated = alpha * x_real_onehot + (1. - alpha) * x_fake_onehot_appr

    logit = discriminator(x_onehot=interpolated)

    grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
    grad_norm = tf.norm(tf.layers.flatten(grad), axis=1)  # l2 norm

    GP = config['reg_param'] * tf.reduce_mean(tf.square(grad_norm - 1.))

    return GP
