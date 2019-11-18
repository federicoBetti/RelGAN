# In this file I will create the training for the model with topics
import datetime
import gc

from tensorflow.python.client import device_lib
from tensorflow.python.saved_model.simple_save import simple_save

from path_resolution import resources_path
from real.real_gan.loaders.real_loader import RealDataTopicLoader
from real.real_gan.real_topic_train_utils import get_accuracy, get_train_ops, \
    get_metric_summary_op, get_metrics, get_fixed_temperature
from utils.utils import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("Available GPUs: {}".format(get_available_gpus()))


# A function to initiate the graph and train the networks
# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def real_topic_train(generator_obj, discriminator_obj, topic_discriminator_obj, oracle_loader: RealDataTopicLoader,
                     config, args):
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
    json_file = os.path.join(sample_dir, 'json_file.txt')
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
    generator = generator_obj(x_real=x_real, temperature=temperature, x_topic=x_topic)
    d_real = discriminator_obj(x_onehot=x_real_onehot)
    d_fake = discriminator_obj(x_onehot=generator.gen_x_onehot_adv)
    if not args.no_topic:
        d_topic_real_pos = topic_discriminator_obj(x_onehot=x_real_onehot, x_topic=x_topic)
        d_topic_real_neg = topic_discriminator_obj(x_onehot=x_real_onehot, x_topic=x_topic_random)
        d_topic_fake = topic_discriminator_obj(x_onehot=generator.gen_x_onehot_adv, x_topic=x_topic)
    else:
        d_topic_real_pos = None
        d_topic_real_neg = None
        d_topic_fake = None

    # GAN / Divergence type
    losses = get_losses(generator, d_real, d_fake, d_topic_real_pos, d_topic_real_neg, d_topic_fake, config)
    if not args.no_topic:
        d_topic_loss = losses['d_topic_loss_real_pos'] + losses[
            'd_topic_loss_real_neg']  # only from real data for pretrain
        d_topic_accuracy = get_accuracy(d_topic_real_pos.logits, d_topic_real_neg.logits)
    else:
        d_topic_loss = None
        d_topic_accuracy = None

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Train ops
    g_pretrain_op, g_train_op, d_train_op, d_topic_pretrain_op = get_train_ops(config, generator.pretrain_loss,
                                                                               losses['g_loss'], losses['d_loss'],
                                                                               d_topic_loss,
                                                                               losses['log_pg'], temperature,
                                                                               global_step)
    generator.g_pretrain_op = g_pretrain_op

    # Record wall clock time
    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    temp_var = tf.placeholder(tf.float32)
    update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
        tf.summary.scalar('adv_loss/discriminator/classic/d_loss_real', losses['d_loss_real']),
        tf.summary.scalar('adv_loss/discriminator/classic/d_loss_fake', losses['d_loss_fake']),
        tf.summary.scalar('adv_loss/discriminator/total', losses['d_loss']),
        tf.summary.scalar('adv_loss/generator/g_sentence_loss', losses['g_sentence_loss']),
        tf.summary.scalar('adv_loss/generator/total_g_loss', losses['g_loss']),
        tf.summary.scalar('adv_loss/log_pg', losses['log_pg']),
        tf.summary.scalar('adv_loss/Wall_clock_time', Wall_clock_time),
        tf.summary.scalar('adv_loss/temperature', temperature),
    ]
    if not args.no_topic:
        loss_summaries += [
            tf.summary.scalar('adv_loss/discriminator/topic_discriminator/d_topic_loss_real_pos',
                              losses['d_topic_loss_real_pos']),
            tf.summary.scalar('adv_loss/discriminator/topic_discriminator/d_topic_loss_fake',
                              losses['d_topic_loss_fake']),
            tf.summary.scalar('adv_loss/generator/g_topic_loss', losses['g_topic_loss'])]

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

    # To save the trained model
    saver = tf.compat.v1.train.Saver()

    # ------------- initial the graph --------------
    with init_sess() as sess:
        variables_dict = get_parameters_division()

        print("Total paramter number: {}".format(
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        log = open(csv_file, 'w')

        now = datetime.datetime.now()
        additional_text = now.strftime("%Y-%m-%d_%H-%M") + "_" + args.summary_name
        summary_dir = os.path.join(log_dir, 'summary', additional_text)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        sum_writer = tf.compat.v1.summary.FileWriter(os.path.join(summary_dir), sess.graph)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)

        run_information.write_summary(str(args), 0)
        print("Information stored in the summary!")

        # generate oracle data and create batches
        oracle_loader.create_batches(oracle_file)

        metrics = get_metrics(config, oracle_loader, test_file, gen_text_file, generator.pretrain_loss, x_real,
                              x_topic, sess,
                              json_file)

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
            progress = tqdm(range(npre_epochs), dynamic_ncols=True)
            for epoch in progress:
                # pre-training
                g_pretrain_loss_np = generator.pretrain_epoch(sess, oracle_loader)
                gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)

                # Test
                ntest_pre = 30
                if np.mod(epoch, ntest_pre) == 0:
                    json_object = generator.generate_samples_topic(sess, oracle_loader, num_sentences)
                    write_json(json_file, json_object)

                    # take sentences from saved files
                    sent = generator.get_sentences(json_object)
                    gen_sentences_summary.write_summary(sent, epoch)

                    # write summaries
                    t = time.time()
                    scores = [metric.get_score() for metric in metrics]
                    metrics_summary_str = sess.run(metric_summary_op, feed_dict=dict(zip(metrics_pl, scores)))
                    sum_writer.add_summary(metrics_summary_str, epoch)

                    msg = 'pre_gen_epoch:' + str(epoch) + ', g_pre_loss: %.4f' % g_pretrain_loss_np
                    metric_names = [metric.get_name() for metric in metrics]
                    for (name, score) in zip(metric_names, scores):
                        score = score * 1e5 if 'Earth' in name else score
                        msg += ', ' + name + ': %.4f' % score
                    progress.set_description(msg + " in {:.2f} sec".format(time.time() - t))
                    log.write(msg)
                    log.write('\n')

                    gc.collect()

        if not args.no_topic:
            gc.collect()
            print('Start Topic Discriminator pre-training...')
            progress = tqdm(range(n_topic_pre_epochs))
            for epoch in progress:
                # pre-training and write loss
                # Pre-train the generator using MLE for one epoch
                supervised_g_losses = []
                supervised_accuracy = []
                oracle_loader.reset_pointer()

                for it in range(oracle_loader.num_batch):
                    text_batch, topic_batch = oracle_loader.next_batch(only_text=False)
                    _, topic_loss, accuracy = sess.run([d_topic_pretrain_op, d_topic_loss, d_topic_accuracy],
                                                       feed_dict={x_real: text_batch, x_topic: topic_batch,
                                                                  x_topic_random: oracle_loader.random_topic()})
                    supervised_g_losses.append(topic_loss)
                    supervised_accuracy.append(accuracy)

                d_topic_pretrain_loss = np.mean(supervised_g_losses)
                accuracy_mean = np.mean(supervised_accuracy)
                topic_discr_pretrain_summary.write_summary(d_topic_pretrain_loss, epoch)
                topic_discr_accuracy_summary.write_summary(accuracy_mean, epoch)
                progress.set_description('topic_loss: %4.4f, accuracy: %4.4f' % (d_topic_pretrain_loss, accuracy_mean))

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
            g_loss_np, d_loss_np, loss_summary_str = sess.run([losses['g_loss'], losses['d_loss'], loss_summary_op],
                                                              feed_dict=feed)
            sum_writer.add_summary(loss_summary_str, niter)

            sess.run(global_step_op)

            progress.set_description('g_loss: %4.4f, d_loss: %4.4f' % (g_loss_np, d_loss_np))

            # Test
            if np.mod(adv_epoch, 400) == 0 or adv_epoch == nadv_steps - 1:
                json_object = generator.generate_samples_topic(sess, oracle_loader, num_sentences)
                write_json(json_file, json_object)

                # take sentences from saved files
                sent = generator.get_sentences(json_object)
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

        save_model = False
        if save_model:
            model_dir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = os.path.join(resources_path("trained_models"), model_dir)
            simple_save(sess,
                        model_path,
                        inputs={"x_topic": x_topic},
                        outputs={"gen_x": generator.gen_x})
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
# noinspection PyUnusedLocal
def get_losses(generator, d_real, d_fake, d_topic_real_pos, d_topic_real_neg, d_topic_fake, config):
    EPS = 10e-5
    batch_size = config['batch_size']
    gan_type = config['gan_type']  # select the gan loss type
    topic_loss_weight = config['topic_loss_weight']
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

                losses['d_loss'] += (topic_loss_weight * (losses['d_topic_loss_real_pos'] + losses['d_topic_loss_fake']))
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
                    logits=d_topic_fake.logits, labels=tf.ones_like(d_topic_fake.logits)
                ), name="g_topic_loss")

                losses['g_loss'] = losses['g_loss'] + (topic_loss_weight * losses['g_topic_loss'])
            else:
                losses['g_topic_loss'] = None

            losses['log_pg'] = tf.reduce_mean(tf.log(generator.gen_o + EPS))  # [1], measures the log p_g(x)

            return losses
