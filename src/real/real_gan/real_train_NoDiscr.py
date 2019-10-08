# In this file I will create the training for the model with topics

import gc

from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import placeholder

from models import rmc_att_topic
from path_resolution import resources_path
from real.real_gan.loaders.real_loader import RealDataTopicLoader
from real.real_gan.real_topic_train_utils import get_metric_summary_op
from utils.metrics.Bleu import Bleu
from utils.metrics.KLDivergence import KL_divergence
from utils.metrics.Nll import Nll
from utils.metrics.SelfBleu import SelfBleu
from utils.utils import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("Available GPUs: {}".format(get_available_gpus()))


# A function to initiate the graph and train the networks
def real_train_NoDiscr(generator_obj: rmc_att_topic.generator, oracle_loader, config, args):
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    dataset = config['dataset']
    npre_epochs = config['npre_epochs']
    # noinspection PyUnusedLocal
    n_topic_pre_epochs = config['n_topic_pre_epochs']
    # noinspection PyUnusedLocal
    nadv_steps = config['nadv_steps']
    # noinspection PyUnusedLocal
    temper = config['temperature']
    # noinspection PyUnusedLocal
    adapt = config['adapt']

    # changed to match resources path
    data_dir = resources_path(config['data_dir'])
    log_dir = resources_path(config['log_dir'])
    sample_dir = resources_path(config['sample_dir'])

    # filename
    oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(dataset))
    # noinspection PyUnusedLocal
    gen_file = os.path.join(sample_dir, 'generator.txt')
    gen_text_file = os.path.join(sample_dir, 'generator_text.txt')
    # noinspection PyUnusedLocal
    gen_text_file_print = os.path.join(sample_dir, 'gen_text_file_print.txt')
    json_file = os.path.join(sample_dir, 'json_file.txt')
    csv_file = os.path.join(log_dir, 'experiment-log-rmcgan.csv')
    # noinspection PyUnusedLocal
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
    x_real = placeholder(tf.int32, [batch_size, seq_len], name="x_real")  # tokens of oracle sequences

    temperature = tf.Variable(1., trainable=False, name='temperature')

    x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    generator = generator_obj(x_real=x_real, temperature=temperature)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    # noinspection PyUnusedLocal
    global_step_op = global_step.assign_add(1)

    # A function to calculate the gradients and get training operations
    def get_train_ops(config, g_pretrain_loss):
        gpre_lr = config['gpre_lr']

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        grad_clip = 5.0  # keep the same with the previous setting

        # generator pre-training
        pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999, name="gen_pretrain_adam")
        pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars, name="gradients_g_pretrain"),
                                                  grad_clip, name="g_pretrain_clipping")  # gradient clipping
        g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

        return g_pretrain_op

    # Train ops
    # noinspection PyUnusedLocal
    g_pretrain_op = get_train_ops(config, generator.pretrain_loss)

    # Record wall clock time
    time_diff = placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    # noinspection PyUnusedLocal
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    temp_var = placeholder(tf.float32)
    # noinspection PyUnusedLocal
    update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
        tf.summary.scalar('adv_loss/Wall_clock_time', Wall_clock_time),
        tf.summary.scalar('adv_loss/temperature', temperature),
    ]
    # noinspection PyUnusedLocal
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
    # noinspection PyUnusedLocal
    saver = tf.train.Saver()

    # ------------- initial the graph --------------
    with init_sess() as sess:
        variables_dict = {}
        for v in tf.trainable_variables():
            name_scope = v.name.split('/')
            d = variables_dict
            params_number = np.prod(v.get_shape().as_list())
            for name in name_scope:
                d[name] = d.get(name, {})
                d = d[name]
                d['total_param'] = d.get('total_param', 0) + params_number

        print("Total paramter number: {}".format(
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        log = open(csv_file, 'w')
        # file_suffix = "date: {}, normal RelGAN, pretrain epochs: {}, adv epochs: {}".format(datetime.datetime.now(),
        #                                                                                     npre_epochs, nadv_steps)
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'),
                                           sess.graph)  # , filename_suffix=file_suffix)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)

        run_information.write_summary(str(args), 0)
        print("Information stored in the summary!")
        # generate oracle data and create batches
        # todo se le parole hanno poco senso potrebbe essere perchè qua ho la corrispondenza indice-parola sbagliata
        # noinspection PyUnusedLocal
        oracle_loader.create_batches(oracle_file)

        metrics = get_metrics(config, oracle_loader, test_file, gen_text_file, generator.pretrain_loss, x_real, sess,
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

        progress = tqdm(range(npre_epochs))
        for epoch in progress:
            # pre-training
            g_pretrain_loss_np = generator.pre_train_epoch(sess, g_pretrain_op, oracle_loader)
            gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)
            progress.set_description("Pretrain_loss: {}".format(g_pretrain_loss_np))
            # Test
            ntest_pre = 40
            if np.mod(epoch, ntest_pre) == 0:
                json_object = generator.generate_sentences(sess, batch_size, num_sentences, oracle_loader=oracle_loader)

                with open(gen_text_file, 'w') as outfile:
                    i = 0
                    for sent in json_object['sentences']:
                        if i < 200:
                            outfile.write(sent['generated_sentence'] + "\n")
                        else:
                            break

                # take sentences from saved files
                sent = take_sentences_json(json_object, first_elem='generated_sentence', second_elem=None)
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
                tqdm.write(msg)
                log.write(msg)
                log.write('\n')

                gc.collect()

            gc.collect()
            sum_writer.close()


def get_metrics(config, oracle_loader, test_file, gen_file, g_pretrain_loss, x_real, sess, json_file):
    # set up evaluation metric
    metrics = []
    if config['nll_gen']:
        nll_gen = Nll(oracle_loader, g_pretrain_loss, x_real, sess, name='nll_gen')
        metrics.append(nll_gen)
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
