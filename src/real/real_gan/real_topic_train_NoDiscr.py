# In this file I will create the training for the model with topics
import datetime
import gc

from tensorflow.compat.v1 import placeholder
from tensorflow.python.client import device_lib
from tensorflow.python.saved_model.simple_save import simple_save

from models.rmc_att_topic import get_sentence_from_index
from path_resolution import resources_path
from real.real_gan.loaders.real_loader import RealDataTopicLoader
from real.real_gan.real_topic_train_utils import get_metric_summary_op, get_metrics
from utils.utils import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("Available GPUs: {}".format(get_available_gpus()))


# A function to initiate the graph and train the networks
def real_topic_train_NoDiscr(generator, oracle_loader: RealDataTopicLoader, config, args):
    batch_size = config['batch_size']
    num_sentences = config['num_sentences']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    dataset = config['dataset']
    npre_epochs = config['npre_epochs']
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
    x_real = placeholder(tf.int32, [batch_size, seq_len], name="x_real")  # tokens of oracle sequences
    x_topic = placeholder(tf.float32, [batch_size, oracle_loader.vocab_size + 1],
                          name="x_topic")  # todo stessa cosa del +1
    x_topic_random = placeholder(tf.float32, [batch_size, oracle_loader.vocab_size + 1], name="x_topic_random")

    temperature = tf.Variable(1., trainable=False, name='temperature')

    x_real_onehot = tf.one_hot(x_real, vocab_size)  # batch_size x seq_len x vocab_size
    assert x_real_onehot.get_shape().as_list() == [batch_size, seq_len, vocab_size]

    # generator and discriminator outputs
    x_fake_onehot_appr, x_fake, g_pretrain_loss, gen_o, \
    lambda_values_returned, gen_x_no_lambda = generator(x_real=x_real,
                                                        temperature=temperature,
                                                        x_topic=x_topic)

    # Global step
    global_step = tf.Variable(0, trainable=False)
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
    g_pretrain_op = get_train_ops(config, g_pretrain_loss)

    # Record wall clock time
    time_diff = placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Temperature placeholder
    temp_var = placeholder(tf.float32)
    update_temperature_op = temperature.assign(temp_var)

    # Loss summaries
    loss_summaries = [
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

    # ------------- initial the graph --------------
    with init_sess() as sess:
        variables_dict = get_parameters_division()

        log = open(csv_file, 'w')
        sum_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'), sess.graph)
        for custom_summary in custom_summaries:
            custom_summary.set_file_writer(sum_writer, sess)

        run_information.write_summary(str(args), 0)
        print("Information stored in the summary!")

        oracle_loader.create_batches(oracle_file)

        metrics = get_metrics(config, oracle_loader, test_file, gen_text_file, g_pretrain_loss, x_real, x_topic, sess,
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
            g_pretrain_loss_np = pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, oracle_loader, x_topic)
            gen_pretrain_loss_summary.write_summary(g_pretrain_loss_np, epoch)
            progress.set_description("Pretrain_loss: {}".format(g_pretrain_loss_np))

            # Test
            ntest_pre = 40
            if np.mod(epoch, ntest_pre) == 0:
                json_object = generate_sentences(sess, x_fake, batch_size, num_sentences, oracle_loader=oracle_loader,
                                                 x_topic=x_topic)
                write_json(json_file, json_object)

                with open(gen_text_file, 'w') as outfile:
                    i = 0
                    for sent in json_object['sentences']:
                        if i < 200:
                            outfile.write(sent['generated_sentence'] + "\n")
                        else:
                            break

                # take sentences from saved files
                sent = take_sentences_json(json_object, first_elem='generated_sentence', second_elem='real_sentence')
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


def generate_sentences(sess, x_fake, batch_size, num_sentences, oracle_loader, x_topic):
    generated_samples, input_sentiment = [], []
    sentence_generated_from = []

    max_gen = int(num_sentences / batch_size)  # - 155 # 156
    for ii in range(max_gen):
        text_batch, topic_batch = oracle_loader.random_batch(only_text=False)
        feed = {x_topic: topic_batch}
        sentence_generated_from.extend(text_batch)
        gen_x_res = sess.run(x_fake, feed_dict=feed)

        generated_samples.extend(gen_x_res)

    json_file = {'sentences': []}
    for sent, input_sent in zip(generated_samples, sentence_generated_from):
        json_file['sentences'].append({
            'generated_sentence': get_sentence_from_index(sent, oracle_loader.model_index_word_dict),
            'real_sentence': get_sentence_from_index(input_sent, oracle_loader.model_index_word_dict)
        })
    return json_file
