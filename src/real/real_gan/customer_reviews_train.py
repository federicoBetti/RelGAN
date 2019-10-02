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
from utils.metrics.Nll import NllTopic
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
    generator.generated_num = num_sentences
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
    saver = tf.train.Saver()
    # ------------- initial the graph --------------
    with init_sess() as sess:

        # count parameters
        variables_dict = get_parameters_division()

        log = open(csv_file, 'w')
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'), sess.graph)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)

        run_information.write_summary(str(args), 0)
        print("Information stored in the summary!")

        def get_metrics():
            # set up evaluation metric
            metrics = []
            if config['nll_gen']:
                nll_gen = NllTopic(oracle_loader, generator_obj.pretrain_loss, x_real, sess, name='nll_gen', x_topic=x_topic)
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
            if config['KL']:
                KL_div = KL_divergence(oracle_loader, json_file, name='KL_divergence')
                metrics.append(KL_div)

            return metrics
        metrics = get_metrics(config, oracle_loader, test_file, gen_text_file, g_pretrain_loss, x_real, None, sess,
                              json_file)
        metrics = []
        metrics.append(BleuAmazon("BleuAmazon_2", json_file=json_file, gram=2))
        metrics.append(BleuAmazon("BleuAmazon_3", json_file=json_file, gram=3))
        metrics.append(BleuAmazon("BleuAmazon_4", json_file=json_file, gram=4))
        metrics.append(BleuAmazon("BleuAmazon_validation_2", json_file=json_file_validation, gram=2))

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

                # Test
                ntest_pre = 40
                if np.mod(epoch, ntest_pre) == 0:
                    json_object = generator_obj.generate_json(oracle_loader, sess)
                    write_json(json_file, json_object)

                    # write summaries
                    scores = [metric.get_score() for metric in metrics]
                    metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                    sum_writer.add_summary(metrics_summary_str, epoch)

                    msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                    metric_names = [metric.get_name() for metric in metrics]
                    for (name, score) in zip(metric_names, scores):
                        msg += ', ' + name + ': %.4f' % score
                    progress.set_description(msg)
                    log.write(msg)
                    log.write('\n')

                    gc.collect()

        gc.collect()
        # Check if there is a pretrained generator and a topic discriminator saved
        model_dir = "PretrainGeneratorAndTopicDiscriminator"
        model_path = resources_path(os.path.join("checkpoint_folder", model_dir))
        try:
            new_saver = tf.train.import_meta_graph(os.path.join(model_path, "model.ckpt.meta"))
            new_saver.restore(sess, os.path.join(model_path, "model.ckpt"))
            print("Used saved model for topic discriminator and pretrained generator")
        except OSError:
            print('Start Topic Discriminator pre-training...')
            topic_discriminator_pretrain(n_topic_pre_epochs, sess, d_topic_pretrain_op, d_topic_loss,
                                         d_topic_accuracy, x_real, x_topic,
                                         x_topic_random, oracle_loader,
                                         d_topic_out_real_pos, d_topic_out_real_neg, topic_discr_pretrain_summary,
                                         topic_discr_accuracy_summary)

            # if not os.path.exists(model_path):
            #     os.makedirs(model_path)
            # save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"))
            # print("Up to Topic Discriminator Pretrain saved in path: %s" % save_path)

        print('Start adversarial training...')
        progress = tqdm(range(nadv_steps))
        for adv_epoch in progress:
            gc.collect()
            niter = sess.run(global_step)

            t0 = time.time()
            # Adversarial training
            for _ in range(config['gsteps']):
                text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
                sess.run(g_train_op, feed_dict={x_real: text_batch, x_topic: topic_batch})
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
            if np.mod(adv_epoch, 500) == 0 or adv_epoch == nadv_steps - 1:
                # generate fake data and create batches
                gen_save_file = os.path.join(sample_dir, 'adv_samples_{:05d}.txt'.format(niter))
                codes_with_lambda, sentence_generated_from, codes, json_object = generate_samples_topic(sess, x_fake,
                                                                                                        batch_size,
                                                                                                        num_sentences,
                                                                                                        lambda_values=lambda_values_returned,
                                                                                                        oracle_loader=oracle_loader,
                                                                                                        gen_x_no_lambda=gen_x_no_lambda,
                                                                                                        x_topic=x_topic)
                create_json_file(json_object, json_file)
                # gen_real_test_file_not_file(codes, sentence_generated_from, gen_save_file, index_word_dict)
                gen_real_test_file_not_file(codes, sentence_generated_from, gen_text_file, index_word_dict, json_object)
                gen_real_test_file_not_file(codes_with_lambda, sentence_generated_from, gen_text_file_print,
                                            index_word_dict, json_object, generator_sentences=True)

                # take sentences from saved files
                sent = take_sentences_topic(gen_text_file_print)
                if adv_epoch < 3500:
                    sent_number = 8
                else:
                    sent_number = 20
                sent = random.sample(sent, sent_number)
                gen_sentences_summary.write_summary(sent, adv_epoch)

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

        sum_writer.close()

        save_model = True
        if save_model:
            model_dir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = os.path.join(resources_path("trained_models"), model_dir)
            simple_save(sess,
                        model_path,
                        inputs={"x_topic": x_topic},
                        outputs={"gen_x": x_fake})
            # save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"))
            print("Model saved in path: %s" % model_path)


def generator_pretrain_amazon(npre_epochs, sess, g_pretrain_op, g_pretrain_loss, x_real, oracle_loader,
                              gen_pretrain_loss_summary, sample_dir, x_fake, batch_size, num_sentences, gen_text_file,
                              gen_sentences_summary, metrics, metric_summary_op, metrics_pl, sum_writer, log,
                              lambda_values, gen_text_file_print, gen_x_no_lambda, json_file):
    progress = tqdm(range(npre_epochs))
    for epoch in progress:
        # pre-training
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        oracle_loader.reset_pointer()

        for it in range(oracle_loader.num_batch):
            user, product, rating, sentence = oracle_loader.next_batch()
            _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: sentence})

        supervised_g_losses.append(g_loss)

        g_pretrain_loss_np = np.mean(supervised_g_losses)
        gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)

        # Test
        ntest_pre = 40
        if np.mod(epoch, ntest_pre) == 0:
            # generate fake data and create batches
            # tqdm.write("Epoch: {}; Computing Metrics and writing summaries".format(epoch), end=" ")
            t = time.time()
            codes_with_lambda, sentence_generated_from, codes, json_object = generate_samples_topic(sess, x_fake,
                                                                                                    batch_size,
                                                                                                    num_sentences,
                                                                                                    lambda_values=lambda_values,
                                                                                                    oracle_loader=oracle_loader,
                                                                                                    gen_x_no_lambda=gen_x_no_lambda,
                                                                                                    x_topic=x_topic)
            create_json_file(json_object, json_file)
            # gen_real_test_file_not_file(codes, sentence_generated_from, gen_save_file, index_word_dict)
            gen_real_test_file_not_file(codes, sentence_generated_from, gen_text_file, index_word_dict, json_object)
            gen_real_test_file_not_file(codes_with_lambda, sentence_generated_from, gen_text_file_print,
                                        index_word_dict, json_object, True)

            # take sentences from saved files
            sent = take_sentences_topic(gen_text_file_print)
            sent = random.sample(sent, 5)
            gen_sentences_summary.write_summary(sent, epoch)

            # write summaries
            scores = [metric.get_score() for metric in metrics]
            metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
            sum_writer.add_summary(metrics_summary_str, epoch)
            # tqdm.write("in {} seconds".format(time.time() - t))

            msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
            metric_names = [metric.get_name() for metric in metrics]
            for (name, score) in zip(metric_names, scores):
                msg += ', ' + name + ': %.4f' % score
            progress.set_description(msg)
            log.write(msg)
            log.write('\n')

            gc.collect()


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
