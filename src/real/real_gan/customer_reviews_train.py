# In this file I will create the training for the model with topics
import datetime
import gc
import random

from tensorflow.python.client import device_lib
from tensorflow.python.saved_model.simple_save import simple_save
from tqdm import tqdm

from models import customer_reviews
from models.customer_reviews import ReviewGenerator, ReviewDiscriminator
from path_resolution import resources_path
from real.real_gan.loaders.custom_reviews_loader import RealDataCustomerReviewsLoader
from real.real_gan.real_topic_train_utils import get_train_ops, \
    get_metric_summary_op, get_fixed_temperature, create_json_file
from utils.metrics.Bleu import BleuAmazon
from utils.metrics.Jaccard import JaccardSimilarity, JaccardDiversity
from utils.metrics.KLDivergence import KL_divergence
from utils.metrics.Nll import NllTopic, NllReview
from utils.utils import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("Available GPUs: {}".format(get_available_gpus()))


# A function to initiate the graph and train the networks

def customer_reviews_train(generator: ReviewGenerator, discriminator_positive: ReviewDiscriminator,
                           discriminator_negative: ReviewDiscriminator,
                           oracle_loader: RealDataCustomerReviewsLoader, config, args):
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocabulary_size']
    seq_len = config['seq_len']
    dataset = config['dataset']
    npre_epochs = config['npre_epochs']
    nadv_steps = config['nadv_steps']
    temper = config['temperature']
    adapt = config['adapt']

    # changed to match resources path
    data_dir = resources_path(config['data_dir'], "Amazon_Attribute")
    log_dir = resources_path(config['log_dir'])
    sample_dir = resources_path(config['sample_dir'])

    # filename
    json_file = os.path.join(sample_dir, 'json_file.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-rmcgan.csv')

    # create necessary directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # placeholder definitions
    x_real = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_real")
    x_pos = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_pos")
    x_neg = tf.placeholder(tf.int32, [batch_size, seq_len], name="x_neg")
    x_sentiment = tf.placeholder(tf.int32, [batch_size], name="x_sentiment")

    temperature = tf.Variable(1., trainable=False, name='temperature')

    x_real_pos_onehot = tf.one_hot(x_pos, vocab_size)  # batch_size x seq_len x vocab_size
    x_real_neg_onehot = tf.one_hot(x_neg, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_pos_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    generator_obj = generator(x_real=x_real, temperature=temperature, x_sentiment=x_sentiment)
    # discriminator for positive sentences
    discriminator_positive_real_pos = discriminator_positive(x_onehot=x_real_pos_onehot)
    discriminator_positive_real_neg = discriminator_positive(x_onehot=x_real_neg_onehot)
    discriminator_positive_fake = discriminator_positive(x_onehot=generator_obj.gen_x_onehot_adv)
    # discriminator for negative sentences
    discriminator_negative_real_pos = discriminator_negative(x_onehot=x_real_pos_onehot)
    discriminator_negative_real_neg = discriminator_negative(x_onehot=x_real_neg_onehot)
    discriminator_negative_fake = discriminator_negative(x_onehot=generator_obj.gen_x_onehot_adv)

    # GAN / Divergence type

    log_pg, g_loss, d_loss = get_losses(generator_obj,
                                        discriminator_positive_real_pos,
                                        discriminator_positive_real_neg,
                                        discriminator_positive_fake,
                                        discriminator_negative_real_pos,
                                        discriminator_negative_real_neg,
                                        discriminator_negative_fake)

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
    metrics_pl, metric_summary_op = get_metric_summary_op(config)

    # Summaries
    gen_pretrain_loss_summary = CustomSummary(name='pretrain_loss', scope='generator')
    gen_sentences_summary = CustomSummary(name='generated_sentences', scope='generator',
                                          summary_type=tf.summary.text, item_type=tf.string)
    run_information = CustomSummary(name='run_information', scope='info',
                                    summary_type=tf.summary.text, item_type=tf.string)
    custom_summaries = [gen_pretrain_loss_summary, gen_sentences_summary, run_information]

    # To save the trained model
    # ------------- initial the graph --------------
    with init_sess() as sess:

        # count parameters

        log = open(csv_file, 'w')
        summary_dir = os.path.join(log_dir, 'summary', str(time.time()))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        sum_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)

        run_information.write_summary(str(args), 0)
        print("Information stored in the summary!")

        def get_metrics():
            # set up evaluation metric
            metrics = []
            if config['nll_gen']:
                nll_gen = NllReview(oracle_loader, generator_obj, sess, name='nll_gen_review')
                metrics.append(nll_gen)
            if config['KL']:
                KL_div = KL_divergence(oracle_loader, json_file, name='KL_divergence')
                metrics.append(KL_div)
            if config['jaccard_similarity']:
                Jaccard_Sim = JaccardSimilarity(oracle_loader, json_file, name='jaccard_similarity')
                metrics.append(Jaccard_Sim)
            if config['jaccard_diversity']:
                Jaccard_Sim = JaccardDiversity(oracle_loader, json_file, name='jaccard_diversity')
                metrics.append(Jaccard_Sim)

            return metrics

        metrics = get_metrics()
        generator_obj.generated_num = num_sentences

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
                oracle_loader.reset_pointer()
                g_pretrain_loss_np = generator_obj.pretrain_epoch(oracle_loader, sess, g_pretrain_op=g_pretrain_op)
                gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)
                msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                progress.set_description(msg)

                # Test
                ntest_pre = 20
                if np.mod(epoch, ntest_pre) == 0 or epoch == npre_epochs - 1:
                    json_object = generator_obj.generate_json(oracle_loader, sess)
                    write_json(json_file, json_object)

                    # take sentences from saved files
                    sent = take_sentences_json(json_object)
                    gen_sentences_summary.write_summary(sent, epoch)

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

        gc.collect()

        print('Start adversarial training...')
        progress = tqdm(range(nadv_steps))
        for adv_epoch in progress:
            gc.collect()
            niter = sess.run(global_step)

            t0 = time.time()
            # Adversarial training
            for _ in range(config['gsteps']):
                sentiment, sentence = oracle_loader.random_batch()
                n = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
                for ind, el in enumerate(sentence):
                    n[ind] = el
                sess.run(g_pretrain_op, feed_dict={generator_obj.x_real: n,
                                                   generator_obj.x_sentiment: sentiment})
            for _ in range(config['dsteps']):
                sentiment, sentence, pos, neg = oracle_loader.get_positive_negative_batch()
                n1 = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
                n2 = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
                n3 = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
                for ind, (s, p, n) in enumerate(zip(sentence, pos, neg)):
                    n1[ind] = s
                    n2[ind] = p[0]
                    n3[ind] = n[0]
                feed_dict = {
                    x_real: n1,
                    x_pos: n2,
                    x_neg: n3,
                    x_sentiment: sentiment
                }
                sess.run(d_train_op, feed_dict=feed_dict)

            t1 = time.time()
            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            # temperature
            temp_var_np = get_fixed_temperature(temper, niter, nadv_steps, adapt)
            sess.run(update_temperature_op, feed_dict={temp_var: temp_var_np})

            sentiment, sentence, pos, neg = oracle_loader.get_positive_negative_batch()
            n1 = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
            n2 = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
            n3 = np.zeros((generator_obj.batch_size, generator_obj.seq_len))
            for ind, (s, p, n) in enumerate(zip(sentence, pos, neg)):
                n1[ind] = s
                n2[ind] = p[0]
                n3[ind] = n[0]
            feed_dict = {
                x_real: n1,
                x_pos: n2,
                x_neg: n3,
                x_sentiment: sentiment
            }
            g_loss_np, d_loss_np, loss_summary_str = sess.run([g_loss, d_loss, loss_summary_op], feed_dict=feed_dict)
            sum_writer.add_summary(loss_summary_str, niter)

            sess.run(global_step_op)

            progress.set_description('g_loss: %4.4f, d_loss: %4.4f' % (g_loss_np, d_loss_np))

            # Test
            # print("N_iter: {}, test every {} epochs".format(niter, config['ntest']))
            if np.mod(adv_epoch, 20) == 0 or adv_epoch == nadv_steps - 1:
                json_object = generator_obj.generate_json(oracle_loader, sess)
                write_json(json_file, json_object)

                # take sentences from saved files
                sent = take_sentences_json(json_object)
                print(sent[:5])
                gen_sentences_summary.write_summary(sent, niter + config['npre_epochs'])

                # write summaries
                scores = [metric.get_score() for metric in metrics]
                metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                sum_writer.add_summary(metrics_summary_str, niter + config['npre_epochs'])

                msg = 'adv_step: ' + str(niter)
                metric_names = [metric.get_name() for metric in metrics]
                for (name, score) in zip(metric_names, scores):
                    msg += ', ' + name + ': %.4f' % score
                tqdm.write(msg)
                log.write(msg)
                log.write('\n')

                gc.collect()

        sum_writer.close()

        save_model = False
        if save_model:
            model_dir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = os.path.join(resources_path("trained_models"), model_dir)
            simple_save(sess,
                        model_path,
                        inputs={"x_topic": x_topic},
                        outputs={"gen_x": x_fake})
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


def get_losses(generator_obj,
               discriminator_positive_real_pos,
               discriminator_positive_real_neg,
               discriminator_positive_fake,
               discriminator_negative_real_pos,
               discriminator_negative_real_neg,
               discriminator_negative_fake):
    EPS = 1e-10
    with tf.variable_scope("standard_GAN_loss"):
        d_loss_pos_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_positive_real_pos.logits,
            labels=tf.ones_like(discriminator_positive_real_pos.logits)
        ), name="d_loss_pos_pos")
        d_loss_pos_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_positive_real_neg.logits,
            labels=tf.zeros_like(discriminator_positive_real_neg.logits)
        ), name="d_loss_pos_neg")
        d_loss_pos_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_positive_fake.logits, labels=tf.zeros_like(discriminator_positive_fake.logits)
        ), name="d_loss_pos_fake")
        d_loss_pos = d_loss_pos_pos + d_loss_pos_neg + d_loss_pos_fake

        d_loss_neg_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_negative_real_neg.logits,
            labels=tf.ones_like(discriminator_negative_real_neg.logits)
        ), name="d_loss_neg_neg")
        d_loss_neg_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_negative_real_pos.logits,
            labels=tf.zeros_like(discriminator_negative_real_pos.logits)
        ), name="d_loss_neg_pos")
        d_loss_neg_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_negative_fake.logits, labels=tf.zeros_like(discriminator_negative_fake.logits)
        ), name="d_loss_neg_fake")
        d_loss_neg = d_loss_neg_neg + d_loss_neg_pos + d_loss_neg_fake
        d_loss = d_loss_neg + d_loss_pos

        g_sentence_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_positive_fake.logits, labels=tf.ones_like(discriminator_positive_fake.logits)
        ), name="g_sentence_loss_pos")
        g_sentence_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_negative_fake.logits, labels=tf.ones_like(discriminator_negative_fake.logits)
        ), name="g_sentence_loss_neg")
        g_loss = g_sentence_loss_pos + g_sentence_loss_neg

        log_pg = tf.reduce_mean(tf.log(generator_obj.gen_o + EPS))  # [1], measures the log p_g(x)

        return log_pg, g_loss, d_loss
