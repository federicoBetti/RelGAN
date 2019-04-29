# In this file I will create the training for the model with topics
import datetime
import random

from tqdm import tqdm

from models import rmc_att_topic
from path_resolution import resources_path
from real.real_gan.real_loader import RealDataTopicLoader
from utils.metrics.Bleu import Bleu
from utils.metrics.DocEmbSim import DocEmbSim
from utils.metrics.Nll import Nll
from utils.metrics.SelfBleu import SelfBleu
from utils.ops import gradient_penalty
from utils.utils import *

EPS = 1e-10

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("Available GPUs: {}".format(get_available_gpus()))


# A function to initiate the graph and train the networks
def real_topic_train(generator: rmc_att_topic.generator, discriminator: rmc_att_topic.discriminator,
                     topic_discriminator: rmc_att_topic.topic_discriminator, oracle_loader: RealDataTopicLoader,
                     config):
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

    # todo check that here it uses vocab size from params that is 5000 instead of 4682 that is the output of text_preprocessing
    x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    x_fake_onehot_appr, x_fake, g_pretrain_loss, gen_o = generator(x_real=x_real, temperature=temperature,
                                                                   x_topic=x_topic)
    d_out_real = discriminator(x_onehot=x_real_onehot)
    d_out_fake = discriminator(x_onehot=x_fake_onehot_appr)
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
        tf.summary.scalar('adv_loss/d_loss_real', d_loss_real),
        tf.summary.scalar('adv_loss/d_loss_fake', d_loss_fake),
        tf.summary.scalar('adv_loss/d_topic_loss_real_pos', d_topic_loss_real_pos),
        tf.summary.scalar('adv_loss/d_topic_loss_real_neg', d_topic_loss_real_neg),
        tf.summary.scalar('adv_loss/d_topic_loss_fake', d_topic_loss_fake),
        tf.summary.scalar('adv_loss/discriminator', d_loss),
        tf.summary.scalar('adv_loss/g_sentence_loss', g_sentence_loss),
        tf.summary.scalar('adv_loss/g_topic_loss', g_topic_loss),
        tf.summary.scalar('adv_loss/g_loss', g_loss),
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

    custom_summaries = [gen_pretrain_loss_summary, gen_sentences_summary, topic_discr_pretrain_summary,
                        topic_discr_accuracy_summary]

    # To save the trained model
    saver = tf.train.Saver()

    # ------------- initial the graph --------------
    with init_sess() as sess:
        log = open(csv_file, 'w')
        # file_suffix = "date: {}, normal RelGAN, pretrain epochs: {}, adv epochs: {}".format(datetime.datetime.now(),
        #                                                                                     npre_epochs, nadv_steps)
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'),
                                           sess.graph)  # , filename_suffix=file_suffix)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)

        # generate oracle data and create batches
        # todo se le parole hanno poco senso potrebbe essere perchè qua ho la corrispondenza indice-parola sbagliata
        # nel codice di prima lo ricomputava qua, invece ora lo faccio io prendendolo da prima
        index_word_dict = oracle_loader.model_index_word_dict
        oracle_loader.create_batches(oracle_file)

        metrics = get_metrics(config, oracle_loader, test_file, gen_text_file, g_pretrain_loss, x_real, sess)

        print('Start pre-training...')
        progress = tqdm(range(npre_epochs))
        for epoch in progress:
            # pre-training
            g_pretrain_loss_np = pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, oracle_loader)
            gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)

            # Test
            ntest_pre = 10
            if np.mod(epoch, ntest_pre) == 0:
                # generate fake data and create batches
                gen_save_file = os.path.join(sample_dir, 'pre_samples_{:05d}.txt'.format(epoch))
                codes, sentence_generated_from = generate_samples_topic(sess, x_fake, batch_size, num_sentences,
                                                                        oracle_loader=oracle_loader, x_topic=x_topic)
                gen_real_test_file_not_file(codes, sentence_generated_from, gen_save_file, index_word_dict)
                gen_real_test_file_not_file(codes, sentence_generated_from, gen_text_file, index_word_dict)

                # take sentences from saved files
                sent = take_sentences_topic(gen_text_file)
                sent = random.sample(sent, 5)  # pick just one sentence
                gen_sentences_summary.write_summary(sent, epoch)

                # write summaries
                print("Computing Metrics and writing summaries", end=" ")
                t = time.time()
                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, epoch)
                print("in {} seconds".format(time.time() - t))

                msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    msg += ', ' + name + ': %.4f' % score
                progress.set_description(msg)
                log.write(msg)
                log.write('\n')

        print('Start Topic Discriminator pre-training...')
        progress = tqdm(range(n_topic_pre_epochs))
        for epoch in progress:
            # pre-training and write loss
            d_topic_pretrain_loss, accuracy_mean = pre_train_discriminator(sess, d_topic_pretrain_op, d_topic_loss,
                                                                           d_topic_accuracy, x_real, x_topic,
                                                                           x_topic_random, oracle_loader,
                                                                           d_topic_out_real_pos, d_topic_out_real_neg)
            topic_discr_pretrain_summary.write_summary(d_topic_pretrain_loss, epoch)
            topic_discr_accuracy_summary.write_summary(accuracy_mean, epoch)
            progress.set_description('topic_loss: %4.4f, accuracy: %4.4f' % (d_topic_pretrain_loss, accuracy_mean))

        print('Start adversarial training...')
        progress = tqdm(range(nadv_steps))
        for adv_epoch in progress:
            niter = sess.run(global_step)

            t0 = time.time()
            # Adversarial training
            for _ in range(config['gsteps']):
                text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
                sess.run(g_train_op, feed_dict={x_real: text_batch, x_topic: topic_batch, })
            for _ in range(config['dsteps']):
                # normal + topic discriminator together
                text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
                sess.run(d_train_op, feed_dict={x_real: text_batch, x_topic: topic_batch,
                                                x_topic_random: oracle_loader.random_topic()})

            t1 = time.time()
            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            # temperature
            temp_var_np = get_fixed_temperature(temper, niter, nadv_steps, adapt)
            sess.run(update_temperature_op, feed_dict={temp_var: temp_var_np})

            text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
            feed = {x_real: text_batch, x_topic: topic_batch, x_topic_random: oracle_loader.random_topic()}
            g_loss_np, d_loss_np, loss_summary_str = sess.run([g_loss, d_loss, loss_summary_op], feed_dict=feed)
            sum_writer.add_summary(loss_summary_str, niter)

            sess.run(global_step_op)

            progress.set_description('g_loss: %4.4f, d_loss: %4.4f' % (g_loss_np, d_loss_np))

            # Test
            # print("N_iter: {}, test every {} epochs".format(niter, config['ntest']))
            if np.mod(adv_epoch, 1000) == 0:
                # generate fake data and create batches
                gen_save_file = os.path.join(sample_dir, 'adv_samples_{:05d}.txt'.format(niter))
                codes, sentence_generated_from = generate_samples_topic(sess, x_fake, batch_size, num_sentences,
                                                                        oracle_loader=oracle_loader, x_topic=x_topic)
                gen_real_test_file_not_file(codes, sentence_generated_from, gen_save_file, index_word_dict)
                gen_real_test_file_not_file(codes, sentence_generated_from, gen_text_file, index_word_dict)

                # take sentences from saved files
                sent = take_sentences_topic(gen_text_file)
                sent = random.sample(sent, 5)  # pick just one sentence
                gen_sentences_summary.write_summary(sent, adv_epoch)

                # write summaries
                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, niter + config['npre_epochs'])

                msg = 'adv_step: ' + str(niter)
                metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    msg += ', ' + name + ': %.4f' % score
                print(msg)
                log.write(msg)
                log.write('\n')

        model_dir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = resources_path(os.path.join("trained_models", model_dir))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"))
        print("Model saved in path: %s" % save_path)


# A function to get different GAN losses
def get_losses(d_out_real, d_out_fake, x_real_onehot, x_fake_onehot_appr, d_topic_out_real_pos, d_topic_out_real_neg,
               d_topic_out_fake, gen_o, discriminator, config):
    """
    :param d_out_real: output del discriminatore ricevuto in input una frase reale
    :param d_out_fake: output del discriminatore ricevuto in input l'output del generatore
    :param x_real_onehot: input reale in one-hot
    :param x_fake_onehot_appr: frasi generate dal generatore in one hot
    :param d_topic_out_real_pos:
    :param d_topic_out_real_neg:
    :param d_topic_out_fake:
    :param gen_o: distribuzione di probabilità sulle parole della frase generata dal generatore
    :param discriminator: discriminator
    :param config: args passed as input
    :return:
    """
    batch_size = config['batch_size']
    gan_type = config['gan_type']  # select the gan loss type

    if gan_type == 'standard':  # the non-satuating GAN loss
        with tf.variable_scope("standard_GAN_loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_real, labels=tf.ones_like(d_out_real)
            ), name="d_loss_real")
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
            ), name="d_loss_fake")

            d_topic_loss_real_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_topic_out_real_pos, labels=tf.ones_like(d_topic_out_real_neg)
            ), name="d_topic_loss_real_pos")

            d_topic_loss_real_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_topic_out_real_neg, labels=tf.zeros_like(d_topic_out_real_pos)
            ), name="d_topic_loss_real_neg")

            d_topic_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_topic_out_fake, labels=tf.zeros_like(d_topic_out_fake)
            ), name="d_topic_loss_fake")

            d_loss = d_loss_real + d_loss_fake + d_topic_loss_real_pos + d_topic_loss_fake

            g_sentence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.ones_like(d_out_fake)
            ), name="g_sentence_loss")

            g_topic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_topic_out_fake, labels=tf.ones_like(d_topic_out_fake)
            ), name="g_topic_loss")

            g_loss = g_sentence_loss + g_topic_loss

            log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)

            return log_pg, g_loss, d_loss, d_loss_real, d_loss_fake, d_topic_loss_real_pos, d_topic_loss_real_neg, d_topic_loss_fake, g_sentence_loss, g_topic_loss

    elif gan_type == 'JS':  # the vanilla GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(-d_out_fake)

    elif gan_type == 'hinge':  # the hinge loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'tv':  # the total variation distance
        d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
        g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))

    elif gan_type == 'wgan-gp':  # WGAN-GP
        d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
        GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
        d_loss += GP

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'LS':  # LS-GAN
        d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, 1.0))
        d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, 1.0))

    elif gan_type == 'RSGAN':  # relativistic standard GAN
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
        ))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
        ))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)

    return log_pg, g_loss, d_loss


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
    gpre_lr = config['gpre_lr']
    gadv_lr = config['gadv_lr']

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    d_topic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='topic_discriminator')

    # train together both discriminators
    # d_vars = d_vars + d_topic_vars

    grad_clip = 5.0  # keep the same with the previous setting

    # generator pre-training
    pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999, name="gen_pretrain_adam")
    pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars, name="gradients_g_pretrain"),
                                              grad_clip, name="g_pretrain_clipping")  # gradient clipping
    g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

    # decide if using the weight decaying
    if config['decay']:
        d_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
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
        g_optimizer = tf.train.RMSPropOptimizer(gadv_lr)
        temp_optimizer = tf.train.RMSPropOptimizer(1e-2)

    else:
        raise NotImplementedError

    # gradient clipping
    g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars, name="gradients_g_adv"), grad_clip,
                                        name="g_adv_clipping")
    g_train_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))

    print('len of g_grads without None: {}'.format(len([i for i in g_grads if i is not None])))
    print('len of g_grads: {}'.format(len(g_grads)))

    # gradient clipping
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(d_loss, d_vars, name="gradients_d_adv"), grad_clip,
                                        name="d_adv_clipping")
    d_train_op = d_optimizer.apply_gradients(zip(d_grads, d_vars))

    # gradient clipping
    d_topic_grads, _ = tf.clip_by_global_norm(tf.gradients(d_topic_loss, d_topic_vars, name="gradients_d_topic_adv"),
                                              grad_clip, name="d_topic_adv_clipping")
    d_topic_pretrain_op = d_optimizer.apply_gradients(zip(d_topic_grads, d_topic_vars))

    return g_pretrain_op, g_train_op, d_train_op, d_topic_pretrain_op


# A function to get various evaluation metrics
def get_metrics(config, oracle_loader, test_file, gen_file, g_pretrain_loss, x_real, sess):
    # set up evaluation metric
    metrics = []
    if config['nll_gen']:
        nll_gen = Nll(oracle_loader, g_pretrain_loss, x_real, sess, name='nll_gen')
        metrics.append(nll_gen)
    if config['doc_embsim']:
        doc_embsim = DocEmbSim(test_file, gen_file, config['vocab_size'], name='doc_embsim')
        metrics.append(doc_embsim)
    if config['bleu']:
        for i in range(2, 6):
            bleu = Bleu(test_text=gen_file, real_text=test_file, gram=i, name='bleu' + str(i))
            metrics.append(bleu)
    if config['selfbleu']:
        for i in range(2, 6):
            selfbleu = SelfBleu(test_text=gen_file, gram=i, name='selfbleu' + str(i))
            metrics.append(selfbleu)

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
        for i in range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='bleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/bleu{}'.format(i), temp_pl))

    if config['selfbleu']:
        for i in range(2, 6):
            temp_pl = tf.placeholder(tf.float32, name='selfbleu{}'.format(i))
            metrics_pl.append(temp_pl)
            metrics_sum.append(tf.summary.scalar('metrics/selfbleu{}'.format(i), temp_pl))

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
