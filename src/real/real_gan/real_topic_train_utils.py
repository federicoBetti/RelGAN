import json

import tensorflow as tf
from utils.metrics.Bleu import Bleu
from utils.metrics.DocEmbSim import DocEmbSim
from utils.metrics.KLDivergence import KL_divergence
from utils.metrics.Nll import Nll, NllTopic
from utils.metrics.SelfBleu import SelfBleu
from utils.ops import gradient_penalty
import numpy as np

EPS = 1e-10


# A function to get different GAN losses
# def get_losses(d_out_real, d_out_fake, x_real_onehot, x_fake_onehot_appr, d_topic_out_real_pos, d_topic_out_real_neg,
#                d_topic_out_fake, gen_o, discriminator, config):
#     """
#     :param d_out_real: output del discriminatore ricevuto in input una frase reale
#     :param d_out_fake: output del discriminatore ricevuto in input l'output del generatore
#     :param x_real_onehot: input reale in one-hot
#     :param x_fake_onehot_appr: frasi generate dal generatore in one hot
#     :param d_topic_out_real_pos: output del topic discriminator ricevedno in input la frase reale e il suo topic
#     :param d_topic_out_real_neg: output del topic discriminator ricevedno in input la frase reale e un topic sbagliato
#     :param d_topic_out_fake: output del topic discriminator ricevendo in input la frase generata
#     :param gen_o: distribuzione di probabilità sulle parole della frase generata dal generatore
#     :param discriminator: discriminator
#     :param config: args passed as input
#     :return:
#     """
#     batch_size = config['batch_size']
#     gan_type = config['gan_type']  # select the gan loss type
#
#     if gan_type == 'standard':  # the non-satuating GAN loss
#         with tf.variable_scope("standard_GAN_loss"):
#             d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                 logits=d_out_real, labels=tf.ones_like(d_out_real)
#             ), name="d_loss_real")
#             d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                 logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
#             ), name="d_loss_fake")
#             d_loss = d_loss_real + d_loss_fake
#
#             if d_topic_out_real_neg is not None:
#                 d_topic_loss_real_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                     logits=d_topic_out_real_pos, labels=tf.ones_like(d_topic_out_real_neg)
#                 ), name="d_topic_loss_real_pos")
#
#                 d_topic_loss_real_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                     logits=d_topic_out_real_neg, labels=tf.zeros_like(d_topic_out_real_pos)
#                 ), name="d_topic_loss_real_neg")
#
#                 d_topic_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                     logits=d_topic_out_fake, labels=tf.zeros_like(d_topic_out_fake)
#                 ), name="d_topic_loss_fake")
#
#                 d_loss += (d_topic_loss_real_pos + d_topic_loss_fake) / 10
#             else:
#                 d_topic_loss_real_pos = None
#                 d_topic_loss_real_neg = None
#                 d_topic_loss_fake = None
#
#             g_sentence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                 logits=d_out_fake, labels=tf.ones_like(d_out_fake)
#             ), name="g_sentence_loss")
#             g_loss = g_sentence_loss
#
#             if d_topic_out_fake is not None:
#                 g_topic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#                     logits=d_topic_out_fake, labels=tf.ones_like(d_topic_out_fake)
#                 ), name="g_topic_loss")
#
#                 g_loss = g_loss + (g_topic_loss / 10)
#             else:
#                 g_topic_loss = None
#
#             log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)
#
#             return log_pg, g_loss, d_loss, d_loss_real, d_loss_fake, d_topic_loss_real_pos, d_topic_loss_real_neg, d_topic_loss_fake, g_sentence_loss, g_topic_loss
#
#     elif gan_type == 'JS':  # the vanilla GAN loss
#         d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=d_out_real, labels=tf.ones_like(d_out_real)
#         ))
#         d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
#         ))
#         d_loss = d_loss_real + d_loss_fake
#
#         g_loss = -d_loss_fake
#
#     elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
#         d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=d_out_real, labels=tf.ones_like(d_out_real)
#         ))
#         d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
#         ))
#         d_loss = d_loss_real + d_loss_fake
#
#         g_loss = tf.reduce_mean(-d_out_fake)
#
#     elif gan_type == 'hinge':  # the hinge loss
#         d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
#         d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
#         d_loss = d_loss_real + d_loss_fake
#
#         g_loss = -tf.reduce_mean(d_out_fake)
#
#     elif gan_type == 'tv':  # the total variation distance
#         d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
#         g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))
#
#     elif gan_type == 'wgan-gp':  # WGAN-GP
#         d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
#         GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
#         d_loss += GP
#
#         g_loss = -tf.reduce_mean(d_out_fake)
#
#     elif gan_type == 'LS':  # LS-GAN
#         d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, 1.0))
#         d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
#         d_loss = d_loss_real + d_loss_fake
#
#         g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, 1.0))
#
#     elif gan_type == 'RSGAN':  # relativistic standard GAN
#         d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
#         ))
#         g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
#         ))
#
#     else:
#         raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)
#
#     log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)
#
#     return log_pg, g_loss, d_loss

# A function to get different GAN losses
def get_losses(generator, d_real, d_fake, d_topic_real_pos, d_topic_real_neg, d_topic_fake, config):
    EPS = 10e-5
    batch_size = config['batch_size']
    gan_type = config['gan_type']  # select the gan loss type
    losses = {}

    if gan_type == 'standard':  # the non-satuating GAN loss
        with tf.variable_scope("standard_GAN_loss"):
            losses['d_loss_real'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_real.logits, labels=tf.ones_like(d_real.logits)
            ), name="d_loss_real")
            losses['d_loss_fake'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_fake.logits, labels=tf.zeros_like(d_fake.logits)
            ), name="d_loss_fake")
            losses['d_loss'] = losses['d_loss_real'] + losses['d_loss_fake']

            if d_topic_real_neg is not None:
                losses['d_topic_loss_real_pos'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_topic_real_pos.logits, labels=tf.ones_like(d_topic_real_pos.logits)
                ), name="d_topic_loss_real_pos")

                losses['d_topic_loss_real_neg'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_topic_real_neg.logits, labels=tf.zeros_like(d_topic_real_neg.logits)
                ), name="d_topic_loss_real_neg")

                losses['d_topic_loss_fake'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_topic_fake.logits, labels=tf.zeros_like(d_topic_fake.logits)
                ), name="d_topic_loss_fake")

                losses['d_loss'] += (losses['d_topic_loss_real_pos'] + losses['d_topic_loss_fake']) / 10
            else:
                losses['d_topic_loss_real_pos'] = None
                losses['d_topic_loss_real_neg'] = None
                losses['d_topic_loss_fake'] = None

            losses['g_sentence_loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_fake.logits, labels=tf.ones_like(d_fake.logits)
            ), name="g_sentence_loss")
            losses['g_loss'] = losses['g_sentence_loss']

            if d_topic_fake is not None:
                losses['g_topic_loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_topic_fake, labels=tf.ones_like(d_topic_fake.logits)
                ), name="g_topic_loss")

                losses['g_loss'] = losses['g_loss'] + (losses['g_topic_loss'] / 10)
            else:
                losses['g_topic_loss'] = None

            losses['log_pg'] = tf.reduce_mean(tf.log(generator.gen_o + EPS))  # [1], measures the log p_g(x)

            return losses

# A function to calculate the gradients and get training operations
def get_train_ops(config, g_pretrain_loss, g_loss, d_loss, d_topic_loss,
                  log_pg, temperature, global_step):
    '''
    :param config:
    :param g_pretrain_loss: final loss del generatore in pretrain
    :param g_loss: final loss del generatore in adv
    :param d_loss: final loss of discriminator (summing both discriminators)
    :param d_topic_loss: loss of the topic discriminator considering only real data
    :param log_pg:
    :param temperature:
    :param global_step:
    :return: ritorna i tre tensori che se 'runnati' fanno un epoca di train rispettivamente per gen_pretrain, gen_adv e discr_adv
    '''
    optimizer_name = config['optimizer']
    nadv_steps = config['nadv_steps']
    d_lr = config['d_lr']
    d_topic_lr = config['d_lr']
    gpre_lr = config['gpre_lr']
    gadv_lr = config['gadv_lr']

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars_pos = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_positive')
    d_vars_neg = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_negative')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    d_topic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='topic_discriminator')

    # train together both discriminators
    d_vars = d_vars + d_topic_vars + d_vars_neg + d_vars_pos

    grad_clip = 5.0  # keep the same with the previous setting

    # generator pre-training
    pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999, name="gen_pretrain_adam")
    pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars, name="gradients_g_pretrain"),
                                              grad_clip, name="g_pretrain_clipping")  # gradient clipping
    g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

    # decide if using the weight decaying
    if config['decay']:
        d_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        d_topic_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        gadv_lr = tf.train.exponential_decay(gadv_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)

    # Adam optimizer
    if optimizer_name == 'adam':
        d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999, name="discriminator_adam")
        d_topic_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999, name="topic_discriminator_adam")
        g_optimizer = tf.train.AdamOptimizer(gadv_lr, beta1=0.9, beta2=0.999, name="generator_adam")
        temp_optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.9, beta2=0.999)

    # RMSProp optimizer
    elif optimizer_name == 'rmsprop':
        d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        d_topic_optimizer = tf.train.RMSPropOptimizer(d_topic_lr)
        g_optimizer = tf.train.RMSPropOptimizer(gadv_lr)
        temp_optimizer = tf.train.RMSPropOptimizer(1e-2)

    else:
        raise NotImplementedError

    # gradient clipping
    g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars, name="gradients_g_adv"), grad_clip,
                                        name="g_adv_clipping")
    g_train_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))

    # gradient clipping
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss, d_vars, name="gradients_d_adv"), grad_clip,
                                        name="d_adv_clipping")
    d_train_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))

    # gradient clipping
    if d_topic_loss is not None:
        d_topic_grads, _ = tf.clip_by_global_norm(
            tf.gradients(d_topic_loss, d_topic_vars, name="gradients_d_topic_adv"),
            grad_clip, name="d_topic_adv_clipping")
        d_topic_pretrain_op = d_topic_optimizer.apply_gradients(zip(d_topic_grads, d_topic_vars))
    else:
        d_topic_pretrain_op = None

    return g_pretrain_op, g_train_op, d_train_op, d_topic_pretrain_op


# A function to get various evaluation metrics
def get_metrics(config, oracle_loader, test_file, gen_file, g_pretrain_loss, x_real, x_topic, sess, json_file):
    # set up evaluation metric
    metrics = []
    if config['nll_gen']:
        nll_gen = NllTopic(oracle_loader, g_pretrain_loss, x_real, sess, name='nll_gen', x_topic=x_topic)
        metrics.append(nll_gen)
    if config['doc_embsim']:
        doc_embsim = DocEmbSim(test_file, gen_file, config['vocab_size'], name='doc_embsim')
        metrics.append(doc_embsim)
    if config['bleu']:
        for i in [2, 4]: #range(2, 6):
            bleu = Bleu(test_text=json_file, real_text=test_file, gram=i, name='bleu' + str(i))
            metrics.append(bleu)
    if config['selfbleu']:
        for i in range(2, 6):
            selfbleu = SelfBleu(test_text=gen_file, gram=i, name='selfbleu' + str(i))
            metrics.append(selfbleu)
    if config['KL']:
        KL_div = KL_divergence(oracle_loader, json_file, name='KL_divergence')
        metrics.append(KL_div)

    return metrics


# A function to get the summary for each metric
def get_metric_summary_op(config):
    metrics_pl = []
    metrics_sum = []

    if config['nll_gen']:
        nll_gen = tf.placeholder(tf.float32)
        metrics_pl.append(nll_gen)
        metrics_sum.append(tf.summary.scalar('metrics/nll_gen', nll_gen))

    if config['doc_embsim']:
        doc_embsim = tf.placeholder(tf.float32)
        metrics_pl.append(doc_embsim)
        metrics_sum.append(tf.summary.scalar('metrics/doc_embsim', doc_embsim))

    if config['bleu']:
        for i in [2, 4]: #range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='bleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/bleu{}'.format(i), temp_pl))

    if config['bleu_amazon']:
        for i in range(2, 5):
            temp_pl = tf.placeholder(tf.float32, name='BleuAmazon_{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/BleuAmazon_{}'.format(i), temp_pl))

    if config['bleu_amazon_validation']:
        for i in range(2, 3):
            temp_pl = tf.placeholder(tf.float32, name='BleuAmazon_validation_{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/BleuAmazon_validation_{}'.format(i), temp_pl))

    if config['selfbleu']:
        for i in range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='selfbleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/selfbleu{}'.format(i), temp_pl))

    if config['KL']:
        KL_placeholder = tf.placeholder(tf.float32)
        metrics_pl.append(KL_placeholder)
        metrics_sum.append(tf.summary.scalar('metrics/KL_topic_divergence', KL_placeholder))

    if config['jaccard_similarity']:
        Jaccard_placeholder = tf.placeholder(tf.float32)
        metrics_pl.append(Jaccard_placeholder)
        metrics_sum.append(tf.summary.scalar('metrics/Jaccard_Similarity', Jaccard_placeholder))

    if config['jaccard_diversity']:
        Jaccard_placeholder = tf.placeholder(tf.float32)
        metrics_pl.append(Jaccard_placeholder)
        metrics_sum.append(tf.summary.scalar('metrics/Jaccard_Diversity', Jaccard_placeholder))

    metric_summary_op = tf.summary.merge(metrics_sum)
    return metrics_pl, metric_summary_op


# A function to set up different temperature control policies
def get_fixed_temperature(temper, i, N, adapt):
    if adapt == 'no':
        temper_var_np = temper  # no increase
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np


def get_accuracy(d_topic_out_real_pos, d_topic_out_real_neg):
    """
    Compute accuracy of the topic discriminator \n
    :param d_topic_out_real_pos: output from the topic discriminator with correct topic
    :param d_topic_out_real_neg: output from the topic discriminator with wrong topic
    :return: summary tensor of the accuracy
    """
    d_topic_out_real_pos = tf.expand_dims(d_topic_out_real_pos, 1)
    d_topic_out_real_neg = tf.expand_dims(d_topic_out_real_neg, 1)
    correct_answer = tf.squeeze(
        tf.concat([tf.ones_like(d_topic_out_real_pos), tf.zeros_like(d_topic_out_real_neg)], axis=0))
    predictions = tf.squeeze(
        tf.concat([d_topic_out_real_pos, d_topic_out_real_neg], axis=0))
    # acc, acc_op = tf.metrics.accuracy(labels=correct_answer, predictions=predictions)
    predicted_class = tf.greater(predictions, 0.5)
    correct = tf.equal(predicted_class, tf.equal(correct_answer, 1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    return accuracy
