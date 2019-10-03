# In this file I will create the training for the model with topics
import datetime
import gc

from tensorflow.python.client import device_lib
from tensorflow.python.saved_model.simple_save import simple_save
from tqdm import tqdm

from models import rmc_att_topic
from path_resolution import resources_path
from real.real_gan.loaders.amazon_loader import RealDataAmazonLoader
from real.real_gan.real_topic_train_utils import get_train_ops, \
    get_metric_summary_op, get_fixed_temperature, EPS
from utils.metrics.Bleu import BleuAmazon
from utils.metrics.KLDivergence import KL_divergence
from utils.metrics.Nll import NllAmazon
from utils.utils import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("Available GPUs: {}".format(get_available_gpus()))


# A function to initiate the graph and train the networks
def amazon_attribute_train(generator: rmc_att_topic.generator, discriminator: rmc_att_topic.discriminator,
                           oracle_loader: RealDataAmazonLoader,
                           config, args):
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocabulary_size']
    seq_len = config['seq_len']
    dataset = config['dataset']
    npre_epochs = config['npre_epochs']
    n_topic_pre_epochs = config['n_topic_pre_epochs']
    nadv_steps = config['nadv_steps']
    temper = config['temperature']
    adapt = config['adapt']

    # changed to match resources path
    data_dir = resources_path(config['data_dir'], "Amazon_Attribute")
    log_dir = resources_path(config['log_dir'])
    sample_dir = resources_path(config['sample_dir'])

    # filename
    oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(dataset))
    gen_file = os.path.join(sample_dir, 'generator.txt')
    gen_text_file = os.path.join(sample_dir, 'generator_text.txt')
    gen_text_file_print = os.path.join(sample_dir, 'gen_text_file_print.txt')
    json_file = os.path.join(sample_dir, 'json_file.txt')
    json_file_validation = os.path.join(sample_dir, 'json_file_validation.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-rmcgan.csv')
    data_file = os.path.join(data_dir, '{}.txt'.format(dataset))

    test_file = os.path.join(data_dir, 'test.csv')

    # create necessary directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # placeholder definitions
    x_real = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_real")  # tokens of oracle sequences
    x_user = tf.placeholder(tf.int32, [batch_size], name="x_user")
    x_product = tf.placeholder(tf.int32, [batch_size], name="x_product")
    x_rating = tf.placeholder(tf.int32, [batch_size], name="x_rating")

    temperature = tf.Variable(1., trainable=False, name='temperature')

    x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    generator_obj = generator(x_real=x_real,
                              temperature=temperature,
                              x_user=x_user, x_product=x_product,
                              x_rating=x_rating)
    discriminator_real = discriminator(x_onehot=x_real_onehot)  # , with_out=False)
    discriminator_fake = discriminator(x_onehot=generator_obj.gen_x_onehot_adv)  # , with_out=False)

    # GAN / Divergence type
    log_pg, g_loss, d_loss = get_losses(generator_obj, discriminator_real, discriminator_fake, config)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Train ops
    g_pretrain_op, g_train_op, d_train_op, d_topic_pretrain_op = get_train_ops(config, generator_obj.pretrain_loss,
                                                                               g_loss, d_loss,
                                                                               None,
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
        tf.summary.scalar('adv_loss/discriminator/total', d_loss),
        tf.summary.scalar('adv_loss/generator/total_g_loss', g_loss),
        tf.summary.scalar('adv_loss/log_pg', log_pg),
        tf.summary.scalar('adv_loss/Wall_clock_time', Wall_clock_time),
        tf.summary.scalar('adv_loss/temperature', temperature),
    ]
    loss_summary_op = tf.summary.merge(loss_summaries)

    # Metric Summaries
    config['bleu_amazon'] = True
    config['bleu_amazon_validation'] = True
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

    # To save the trained model
    saver = tf.train.Saver()
    # ------------- initial the graph --------------
    with init_sess() as sess:
        variables_dict = get_parameters_division()

        log = open(csv_file, 'w')
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'), sess.graph)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)
        run_information.write_summary(str(args), 0)
        print("Information stored in the summary!")

        metrics = get_metrics(config, oracle_loader, sess, json_file, json_file_validation, generator_obj)

        gc.collect()
        # Check if there is a pretrained generator saved
        model_dir = "PretrainGenerator"
        model_path = resources_path(os.path.join("checkpoint_folder", model_dir))
        try:
            new_saver = tf.train.import_meta_graph(os.path.join(model_path, "model.ckpt.meta"))
            new_saver.restore(sess, os.path.join(model_path, "model.ckpt"))
            print("Used saved model for generator pretrain")
        except OSError:
            print('Start pre-training...')
            # pre-training
            # Pre-train the generator using MLE for one epoch

            progress = tqdm(range(npre_epochs))
            for epoch in progress:
                g_pretrain_loss_np = generator_obj.pretrain_epoch(oracle_loader, sess, g_pretrain_op=g_pretrain_op)
                gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)

                # Test
                ntest_pre = 40
                if np.mod(epoch, ntest_pre) == 0:
                    generator_obj.generated_num = 200
                    json_object = generator_obj.generate_samples(sess, oracle_loader, dataset="train")
                    write_json(json_file, json_object)
                    json_object = generator_obj.generate_samples(sess, oracle_loader, dataset="validation")
                    write_json(json_file_validation, json_object)

                    # write summaries
                    scores = [metric.get_score() for metric in metrics]
                    metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                    sum_writer.add_summary(metrics_summary_str, epoch)

                    msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                    metric_names = [metric.get_name() for metric in metrics]
                    for (name, score) in zip(metric_names, scores):
                        msg += ', ' + name + ': %.4f' % score
                    tqdm.write(msg)
                    log.write(msg)
                    log.write('\n')

                    gc.collect()

        print('Start adversarial training...')
        progress = tqdm(range(nadv_steps))
        for adv_epoch in progress:
            gc.collect()
            niter = sess.run(global_step)

            t0 = time.time()
            # Adversarial training
            for _ in range(config['gsteps']):
                user, product, rating, sentence = oracle_loader.random_batch(dataset="train")
                feed_dict = {generator_obj.x_user: user,
                             generator_obj.x_product: product,
                             generator_obj.x_rating: rating}
                sess.run(g_train_op, feed_dict=feed_dict)
            for _ in range(config['dsteps']):
                user, product, rating, sentence = oracle_loader.random_batch(dataset="train")
                n = np.zeros((batch_size, seq_len))
                for ind, el in enumerate(sentence):
                    n[ind] = el
                feed_dict = {generator_obj.x_user: user,
                             generator_obj.x_product: product,
                             generator_obj.x_rating: rating,
                             x_real: n}
                sess.run(d_train_op, feed_dict=feed_dict)

            t1 = time.time()
            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            # temperature
            temp_var_np = get_fixed_temperature(temper, niter, nadv_steps, adapt)
            sess.run(update_temperature_op, feed_dict={temp_var: temp_var_np})

            user, product, rating, sentence = oracle_loader.random_batch(dataset="train")
            n = np.zeros((batch_size, seq_len))
            for ind, el in enumerate(sentence):
                n[ind] = el
            feed_dict = {generator_obj.x_user: user,
                         generator_obj.x_product: product,
                         generator_obj.x_rating: rating,
                         x_real: n}
            g_loss_np, d_loss_np, loss_summary_str = sess.run([g_loss, d_loss, loss_summary_op], feed_dict=feed_dict)
            sum_writer.add_summary(loss_summary_str, niter)

            sess.run(global_step_op)

            progress.set_description('g_loss: %4.4f, d_loss: %4.4f' % (g_loss_np, d_loss_np))

            # Test
            if np.mod(adv_epoch, 500) == 0 or adv_epoch == nadv_steps - 1:
                generator_obj.generated_num = generator_obj.batch_size * 10
                json_object = generator_obj.generate_samples(sess, oracle_loader, dataset="train")
                write_json(json_file, json_object)
                json_object = generator_obj.generate_samples(sess, oracle_loader, dataset="validation")
                write_json(json_file_validation, json_object)

                # write summaries
                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, adv_epoch + npre_epochs)
                # tqdm.write("in {} seconds".format(time.time() - t))

                msg = 'pre_gen_epoch:' + str(adv_epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    msg += ', ' + name + ': %.4f' % score
                tqdm.write(msg)
                log.write(msg)
                log.write('\n')

                gc.collect()

        sum_writer.close()

        save_model = True
        if save_model:
            model_dir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = os.path.join(resources_path("trained_models"), model_dir)
            simple_save(sess,
                        model_path,
                        inputs={"x_user": x_user,
                                "x_rating": x_rating,
                                "x_product": x_product},
                        outputs={"gen_x": generator_obj.gen_x})
            # save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"))
            print("Model saved in path: %s" % model_path)


def topic_discriminator_pretrain(n_topic_pre_epochs, sess, d_topic_pretrain_op, d_topic_loss,
                                 d_topic_accuracy, x_real, x_topic,
                                 x_topic_random, oracle_loader,
                                 d_topic_out_real_pos, d_topic_out_real_neg, topic_discr_pretrain_summary,
                                 topic_discr_accuracy_summary):
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


# A function to get different GAN losses
def get_losses(generator, discriminator_real, discriminator_fake, config):
    '''
    :param d_out_real: output del discriminatore ricevuto in input una frase reale
    :param d_out_fake: output del discriminatore ricevuto in input l'output del generatore
    :param x_real_onehot: input reale in one-hot
    :param x_fake_onehot_appr: frasi generate dal generatore in one hot
    :param gen_o: distribuzione di probabilità sulle parole della frase generata dal generatore
    :param discriminator: discriminator
    :param config: args passed as input
    :return:
    '''
    batch_size = config['batch_size']
    gan_type = config['gan_type']  # select the gan loss type

    d_out_real = discriminator_real.logits
    d_out_fake = discriminator_fake.logits
    if gan_type == 'standard':  # the non-satuating GAN loss
        with tf.variable_scope("standard_GAN_loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_real, labels=tf.ones_like(d_out_real)
            ), name="d_loss_real")
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
            ), name="d_loss_fake")
            d_loss = d_loss_real + d_loss_fake
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.ones_like(d_out_fake)
            ), name="g_loss")

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

    # elif gan_type == 'wgan-gp':  # WGAN-GP
    #     d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
    #     GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
    #     d_loss += GP
    #
    #     g_loss = -tf.reduce_mean(d_out_fake)

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

    log_pg = tf.reduce_mean(tf.log(generator.gen_o + EPS))  # [1], measures the log p_g(x)

    return log_pg, g_loss, d_loss


def get_metrics(config, oracle_loader, sess, json_file, json_file_validation, generator_obj):
    # set up evaluation metric
    metrics = []
    if config['nll_gen']:
        nll_gen = NllAmazon(oracle_loader, generator_obj, sess, name='nll_gen')
        metrics.append(nll_gen)
    if config['bleu_amazon']:
        for i in range(2, 5):
            bleu = BleuAmazon("BleuAmazon_{}".format(i), json_file=json_file, gram=i)
            metrics.append(bleu)
    if config['bleu_amazon_validation']:
        for i in range(2, 3):
            bleu = BleuAmazon("BleuAmazon_validation_{}".format(i), json_file=json_file_validation, gram=i)
            metrics.append(bleu)
    if config['KL']:
        KL_div = KL_divergence(oracle_loader, json_file, name='KL_divergence')
        metrics.append(KL_div)

    return metrics
