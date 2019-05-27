import os

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from path_resolution import resources_path
from real.real_gan.real_topic_train_utils import get_accuracy, get_train_ops, get_metric_summary_op, get_losses
from utils.text_process import code_to_text
from utils.utils import CustomSummary
# from translate import Translator
from googletrans import Translator
import numpy as np


def get_sentences(path):
    sentences = []
    with open(path, 'r') as outfile:
        for line in outfile:
            sentences.append(line)

    # translator = Translator()
    # sentences = [translator.translate(s, dest='en') for s in sentences]

    return sentences


def fix_size(x_topic, batch_size):
    print(x_topic)
    print(x_topic.shape)
    x_topic_shape = x_topic.shape
    if x_topic_shape[0] > batch_size:
        return x_topic[:batch_size, :]
    else:
        fake_ret = np.zeros((batch_size, x_topic_shape[1]))
        fake_ret[:x_topic_shape[0], :] = x_topic
        return fake_ret


def inference_main(generator, discriminator, topic_discriminator, oracle_loader, config, args, model_path, input_path):
    tf.reset_default_graph()
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    dataset = config['dataset']
    npre_epochs = config['npre_epochs']
    n_topic_pre_epochs = config['n_topic_pre_epochs']
    nadv_steps = config['nadv_steps']
    temper = config['temperature']
    adapt = config['adapt']

    # changed to match resources path
    data_dir = resources_path(config['data_dir'])
    log_dir = resources_path(config['log_dir'])
    sample_dir = resources_path(config['sample_dir'])

    # filename
    oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(dataset))
    gen_file = os.path.join(sample_dir, 'generator.txt')
    gen_text_file = os.path.join(sample_dir, 'generator_text.txt')
    gen_text_file_print = os.path.join(sample_dir, 'gen_text_file_print.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-rmcgan.csv')
    data_file = os.path.join(data_dir, '{}.txt'.format(dataset))
    if dataset == 'image_coco':
        test_file = os.path.join(data_dir, 'testdata/test_coco.txt')
    elif dataset == 'emnlp_news':
        test_file = os.path.join(data_dir, 'testdata/test_emnlp.txt')
    else:
        raise NotImplementedError('Unknown dataset!')

    # create necessary directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # placeholder definitions
    x_real = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_real")  # tokens of oracle sequences
    x_topic = tf.placeholder(tf.float32, [batch_size, oracle_loader.vocab_size + 1],
                             name="x_topic")  # todo stessa cosa del +1
    x_topic_random = tf.placeholder(tf.float32, [batch_size, oracle_loader.vocab_size + 1], name="x_topic_random")

    temperature = tf.Variable(1., trainable=False, name='temperature')

    x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    x_fake_onehot_appr, x_fake, g_pretrain_loss, gen_o, lambda_values_returned = generator(x_real=x_real,
                                                                                           temperature=temperature,
                                                                                           x_topic=x_topic)
    d_out_real = discriminator(x_onehot=x_real_onehot, with_out=False)
    d_out_fake = discriminator(x_onehot=x_fake_onehot_appr, with_out=False)
    d_topic_out_real_pos = topic_discriminator(x_onehot=x_real_onehot, x_topic=x_topic)
    d_topic_out_real_neg = topic_discriminator(x_onehot=x_real_onehot, x_topic=x_topic_random)
    d_topic_out_fake = topic_discriminator(x_onehot=x_fake_onehot_appr, x_topic=x_topic)

    # GAN / Divergence type
    log_pg, g_loss, d_loss, d_loss_real, d_loss_fake, d_topic_loss_real_pos, \
    d_topic_loss_real_neg, d_topic_loss_fake, g_sentence_loss, g_topic_loss = get_losses(
        d_out_real, d_out_fake, x_real_onehot, x_fake_onehot_appr,
        d_topic_out_real_pos, d_topic_out_real_neg, d_topic_out_fake,
        gen_o, discriminator, config)
    d_topic_loss = d_topic_loss_real_pos + d_topic_loss_real_neg  # only from real data for pretrain
    d_topic_accuracy = get_accuracy(d_topic_out_real_pos, d_topic_out_real_neg)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Train ops
    g_pretrain_op, g_train_op, d_train_op, d_topic_pretrain_op = get_train_ops(config, g_pretrain_loss, g_loss, d_loss,
                                                                               d_topic_loss,
                                                                               log_pg, temperature, global_step)

    # Record wall clock time
    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    temp_var = tf.placeholder(tf.float32)
    update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
        tf.summary.scalar('adv_loss/discriminator/classic/d_loss_real', d_loss_real),
        tf.summary.scalar('adv_loss/discriminator/classic/d_loss_fake', d_loss_fake),
        tf.summary.scalar('adv_loss/discriminator/topic_discriminator/d_topic_loss_real_pos', d_topic_loss_real_pos),
        # tf.summary.scalar('adv_loss/discriminator/topic_discriminator/d_topic_loss_real_neg', d_topic_loss_real_neg),
        tf.summary.scalar('adv_loss/discriminator/topic_discriminator/d_topic_loss_fake', d_topic_loss_fake),
        tf.summary.scalar('adv_loss/discriminator/total', d_loss),
        tf.summary.scalar('adv_loss/generator/g_sentence_loss', g_sentence_loss),
        tf.summary.scalar('adv_loss/generator/g_topic_loss', g_topic_loss),
        tf.summary.scalar('adv_loss/generator/total_g_loss', g_loss),
        tf.summary.scalar('adv_loss/log_pg', log_pg),
        tf.summary.scalar('adv_loss/Wall_clock_time', Wall_clock_time),
        tf.summary.scalar('adv_loss/temperature', temperature),
    ]
    loss_summary_op = tf.summary.merge(loss_summaries)

    # Metric Summaries
    metrics_pl, metric_summary_op = get_metric_summary_op(config)

    # Summaries
    gen_pretrain_loss_summary = CustomSummary(name='pretrain_loss', scope='generator')
    gen_sentences_summary = CustomSummary(name='generated_sentences', scope='generator',
                                          summary_type=tf.summary.text, item_type=tf.string)
    topic_discr_pretrain_summary = CustomSummary(name='pretrain_loss', scope='topic_discriminator')
    topic_discr_accuracy_summary = CustomSummary(name='pretrain_accuracy', scope='topic_discriminator')
    run_information = CustomSummary(name='run_information', scope='info',
                                    summary_type=tf.summary.text, item_type=tf.string)
    custom_summaries = [gen_pretrain_loss_summary, gen_sentences_summary, topic_discr_pretrain_summary,
                        topic_discr_accuracy_summary, run_information]

    sentences = get_sentences(input_path)
    # Add ops to save and restore all the variables.
    sent_number = len(sentences)
    topic_sentences = oracle_loader.get_topic(sentences)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                model_path,
            )

            x_topic = graph.get_tensor_by_name('x_topic:0')
            gen_x = graph.get_tensor_by_name("generator/gen_x_trans:0")

            topic_sentences = fix_size(topic_sentences, batch_size)
            res = sess.run(gen_x, feed_dict={x_topic: topic_sentences})

    print("FINITO!!")
    for index in range(sent_number):
        print(code_to_text(codes=[res[index]], dictionary=oracle_loader.model_index_word_dict))


    #
    # saver = tf.train.Saver()
    #
    # with tf.Session() as sess:
    #     # Restore variables from disk.
    #     saver.restore(sess, model_path)
    #     print("Model restored.")
