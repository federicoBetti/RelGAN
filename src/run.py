import argparse
import os
import random
from os.path import join

import models
from oracle.oracle_gan.oracle_loader import OracleDataLoader
from oracle.oracle_gan.oracle_train import oracle_train
from path_resolution import resources_path
from real.real_gan.loaders.real_loader import RealDataTopicLoader, RealDataLoader
from utils.models.OracleLstm import OracleLstm
from utils.static_file_manage import load_json
from utils.text_process import text_precess
from utils.utils import pp, str2bool

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# con la cpu ho un epoca di pretrain intorno a 4.5 sec
# con la GPU circa 2.3 secondi, la metà

parser = argparse.ArgumentParser(description='Train and run a RmcGAN')
# Topic?
parser.add_argument('--topic', default=False, action='store_true', help='If to use topic models or not')
parser.add_argument('--topic_number', default=9, type=int, help='How many topic to use in the LDA')
parser.add_argument('--topic-in-memory', type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Activate topic-in-memory mode.")
parser.add_argument('--no-topic', type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Use condition on topic.")
parser.add_argument('--LSTM', default=False, action='store_true', help='If LSTM need to be used')
parser.add_argument('--summary_name', default='', type=str, help='Name of the summary')
parser.add_argument('--topic_loss_weight', default=1e-1, type=float, help='Weight of the topic loss')

# Architecture
parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator')
parser.add_argument('--g-architecture', default='rmc_att', type=str, help='Architecture for generator')
parser.add_argument('--d-architecture', default='rmc_att', type=str, help='Architecture for discriminator')
parser.add_argument('--topic-architecture', default='standard', type=str, help='Architecture for topic discriminator')
parser.add_argument('--gan-type', default='standard', type=str, help='Which type of GAN to use')
parser.add_argument('--hidden-dim', default=32, type=int, help='only used for OrcaleLstm and lstm_vanilla (generator)')
parser.add_argument('--sn', default=False, action='store_true', help='if using spectral norm')

# Training
parser.add_argument('--gsteps', default='1', type=int, help='How many training steps to use for generator')
parser.add_argument('--dsteps', default='5', type=int, help='How many training steps to use for discriminator')
parser.add_argument('--n-topic-pre-epochs', default=1, type=int,
                    help='Number of steps to run pre-training for the topic discriminator')
parser.add_argument('--npre-epochs', default=1, type=int, help='Number of steps to run pre-training')
parser.add_argument('--nadv-steps', default=1, type=int, help='Number of steps to run adversarial training')
parser.add_argument('--ntest', default=50, type=int, help='How often to run tests')
parser.add_argument('--d-lr', default=1e-4, type=float, help='Learning rate for the discriminator')
parser.add_argument('--gpre-lr', default=1e-2, type=float, help='Learning rate for the generator in pre-training')
parser.add_argument('--gadv-lr', default=1e-4, type=float, help='Learning rate for the generator in adv-training')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
parser.add_argument('--log-dir', default=os.path.join('.', 'oracle', 'logs'), type=str,
                    help='Where to store log and checkpoint files')
parser.add_argument('--sample-dir', default=os.path.join('.', 'oracle', 'samples'), type=str,
                    help='Where to put samples during training')
parser.add_argument('--optimizer', default='adam', type=str, help='training method')
parser.add_argument('--decay', default=False, action='store_true', help='if decaying learning rate')
parser.add_argument('--adapt', default='exp', type=str,
                    help='temperature control policy: [no, lin, exp, log, sigmoid, quad, sqrt]')
parser.add_argument('--seed', default=123, type=int, help='for reproducing the results')
parser.add_argument('--temperature', default=1000, type=float, help='the largest temperature')

# evaluation
parser.add_argument('--nll-oracle', default=False, action='store_true', help='if using nll-oracle metric')
parser.add_argument('--nll-gen', default=False, action='store_true', help='if using nll-gen metric')
parser.add_argument('--bleu', default=False, action='store_true', help='if using bleu metric, [2,3,4,5]')
parser.add_argument('--selfbleu', default=False, action='store_true', help='if using selfbleu metric, [2,3,4,5]')
parser.add_argument('--doc-embsim', default=False, action='store_true', help='if using DocEmbSim metric')
parser.add_argument('--KL', default=False, action='store_true', help='if using KL divergence metric')
parser.add_argument('--earth_mover', default=False, action='store_true', help='if using earth mover distance metric')
parser.add_argument('--jaccard-similarity', default=False, action='store_true', help='if using Jaccard metric')
parser.add_argument('--jaccard-diversity', default=False, action='store_true', help='if using Jaccard diversity metric')
parser.add_argument('--bleu-amazon', default=False, action='store_true', help='if using bleu on amazon dataset')
parser.add_argument('--bleu-amazon-validation', default=False, action='store_true', help='if using bleu on amazon '
                                                                                         'dataset validation set')

# relational memory
parser.add_argument('--mem-slots', default=1, type=int, help="memory size")
parser.add_argument('--head-size', default=512, type=int, help="head size or memory size")
parser.add_argument('--num-heads', default=1, type=int, help="number of heads")

# Data
parser.add_argument('--dataset', default='image_coco', type=str, help='[oracle, image_coco, emnlp_news]')
parser.add_argument('--vocab-size', default=5000, type=int, help="vocabulary size")
parser.add_argument('--start-token', default=0, type=int, help="start token for a sentence")
parser.add_argument('--seq-len', default=20, type=int, help="sequence length: [20, 40]")
parser.add_argument('--num-sentences', default=100, type=int, help="number of total sentences")
parser.add_argument('--gen-emb-dim', default=32, type=int, help="generator embedding dimension")
parser.add_argument('--dis-emb-dim', default=64, type=int, help="TOTAL discriminator embedding dimension")
parser.add_argument('--num-rep', default=64, type=int, help="number of discriminator embedded representations")
parser.add_argument('--data-dir', default=os.path.join('.', 'data'), type=str, help='Where data data is stored')


# Files
parser.add_argument('--json-file', default='', type=str, help='suffix to the json file name')

def create_subsample_data_file(data_file, train_size=10000):
    lda_file = data_file
    print("Start of create subsample")
    new_data_file = lda_file[:-4] + '_train.txt'
    if os.path.isfile(lda_file) and os.path.isfile(new_data_file):
        return new_data_file, lda_file

    sentences = []
    with open(lda_file) as f:
        for line in f:
            sentences.append(line.rstrip())
    final_sentences = random.sample(sentences, train_size)
    del sentences

    with open(new_data_file, 'w') as f:
        for item in final_sentences:
            f.write("%s\n" % item)

    print("Files written")
    return new_data_file, lda_file


def main():
    args = parser.parse_args()
    pp.pprint(vars(args))
    config = vars(args)

    # train with different datasets
    if args.dataset == 'oracle':
        oracle_model = OracleLstm(num_vocabulary=args.vocab_size, batch_size=args.batch_size, emb_dim=args.gen_emb_dim,
                                  hidden_dim=args.hidden_dim, sequence_length=args.seq_len,
                                  start_token=args.start_token)
        oracle_loader = OracleDataLoader(args.batch_size, args.seq_len)
        gen_loader = OracleDataLoader(args.batch_size, args.seq_len)

        generator = models.get_generator(args.g_architecture, vocab_size=args.vocab_size, batch_size=args.batch_size,
                                         seq_len=args.seq_len, gen_emb_dim=args.gen_emb_dim, mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                                         start_token=args.start_token)
        discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size, seq_len=args.seq_len,
                                                 vocab_size=args.vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)
        oracle_train(generator, discriminator, oracle_model, oracle_loader, gen_loader, config)

    elif args.dataset in ['image_coco', 'emnlp_news']:
        # custom dataset selected
        data_file = resources_path(args.data_dir, '{}.txt'.format(args.dataset))
        sample_dir = resources_path(config['sample_dir'])
        oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(args.dataset))

        data_dir = resources_path(config['data_dir'])
        if args.dataset == 'image_coco':
            test_file = os.path.join(data_dir, 'testdata/test_coco.txt')
        elif args.dataset == 'emnlp_news':
            test_file = os.path.join(data_dir, 'testdata/test_emnlp.txt')
        else:
            raise NotImplementedError('Unknown dataset!')

        if args.dataset == 'emnlp_news':
            data_file, lda_file = create_subsample_data_file(data_file)
        else:
            lda_file = data_file

        seq_len, vocab_size, word_index_dict, index_word_dict = text_precess(data_file, test_file,
                                                                             oracle_file=oracle_file)
        config['seq_len'] = seq_len
        config['vocab_size'] = vocab_size
        print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))

        config['topic_loss_weight'] = args.topic_loss_weight

        if config['LSTM']:
            if config['topic']:
                topic_number = config['topic_number']
                oracle_loader = RealDataTopicLoader(args.batch_size, args.seq_len)
                oracle_loader.set_dataset(args.dataset)
                oracle_loader.set_files(data_file, lda_file)
                oracle_loader.topic_num = topic_number
                oracle_loader.set_dictionaries(word_index_dict, index_word_dict)

                generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                                 seq_len=seq_len, gen_emb_dim=args.gen_emb_dim,
                                                 mem_slots=args.mem_slots,
                                                 head_size=args.head_size, num_heads=args.num_heads,
                                                 hidden_dim=args.hidden_dim,
                                                 start_token=args.start_token, TopicInMemory=args.topic_in_memory,
                                                 NoTopic=args.no_topic)

                from real.real_gan.real_topic_train_NoDiscr import real_topic_train_NoDiscr
                real_topic_train_NoDiscr(generator, oracle_loader, config, args)
            else:
                generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                                 seq_len=seq_len, gen_emb_dim=args.gen_emb_dim,
                                                 mem_slots=args.mem_slots,
                                                 head_size=args.head_size, num_heads=args.num_heads,
                                                 hidden_dim=args.hidden_dim,
                                                 start_token=args.start_token)

                oracle_loader = RealDataLoader(args.batch_size, args.seq_len)
                oracle_loader.set_dictionaries(word_index_dict, index_word_dict)
                oracle_loader.set_dataset(args.dataset)
                oracle_loader.set_files(data_file, lda_file)
                oracle_loader.topic_num = config['topic_number']

                from real.real_gan.real_train_NoDiscr import real_train_NoDiscr
                real_train_NoDiscr(generator, oracle_loader, config, args)
        else:
            if config['topic']:
                topic_number = config['topic_number']
                oracle_loader = RealDataTopicLoader(args.batch_size, args.seq_len)
                oracle_loader.set_dataset(args.dataset)
                oracle_loader.set_files(data_file, lda_file)
                oracle_loader.topic_num = topic_number
                oracle_loader.set_dictionaries(word_index_dict, index_word_dict)

                generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                                 seq_len=seq_len, gen_emb_dim=args.gen_emb_dim,
                                                 mem_slots=args.mem_slots,
                                                 head_size=args.head_size, num_heads=args.num_heads,
                                                 hidden_dim=args.hidden_dim,
                                                 start_token=args.start_token, TopicInMemory=args.topic_in_memory,
                                                 NoTopic=args.no_topic)

                discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size,
                                                         seq_len=seq_len,
                                                         vocab_size=vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                         num_rep=args.num_rep, sn=args.sn)

                if not args.no_topic:
                    topic_discriminator = models.get_topic_discriminator(args.topic_architecture,
                                                                         batch_size=args.batch_size,
                                                                         seq_len=seq_len, vocab_size=vocab_size,
                                                                         dis_emb_dim=args.dis_emb_dim, num_rep=args.num_rep,
                                                                         sn=args.sn, discriminator=discriminator)
                else:
                    topic_discriminator = None
                from real.real_gan.real_topic_train import real_topic_train
                real_topic_train(generator, discriminator, topic_discriminator, oracle_loader, config, args)
            else:

                generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                                 seq_len=seq_len, gen_emb_dim=args.gen_emb_dim,
                                                 mem_slots=args.mem_slots,
                                                 head_size=args.head_size, num_heads=args.num_heads,
                                                 hidden_dim=args.hidden_dim,
                                                 start_token=args.start_token)

                discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size,
                                                         seq_len=seq_len,
                                                         vocab_size=vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                         num_rep=args.num_rep, sn=args.sn)

                oracle_loader = RealDataLoader(args.batch_size, args.seq_len)

                from real.real_gan.real_train import real_train
                real_train(generator, discriminator, oracle_loader, config, args)

    elif args.dataset in ['Amazon_Attribute']:
        # custom dataset selected
        data_dir = resources_path(config['data_dir'], "Amazon_Attribute")
        sample_dir = resources_path(config['sample_dir'])
        oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(args.dataset))
        train_file = os.path.join(data_dir, 'train.csv')
        dev_file = os.path.join(data_dir, 'dev.csv')
        test_file = os.path.join(data_dir, 'test.csv')

        # create_tokens_files(data_files=[train_file, dev_file, test_file])
        config_file = load_json(os.path.join(data_dir, 'config.json'))
        config = {**config, **config_file}  # merge dictionaries

        from real.real_gan.loaders.amazon_loader import RealDataAmazonLoader
        oracle_loader = RealDataAmazonLoader(args.batch_size, args.seq_len)
        oracle_loader.create_batches(data_file=[train_file, dev_file, test_file])
        oracle_loader.model_index_word_dict = load_json(join(data_dir, 'index_word_dict.json'))
        oracle_loader.model_word_index_dict = load_json(join(data_dir, 'word_index_dict.json'))

        generator = models.get_generator("amazon_attribute", vocab_size=config['vocabulary_size'],
                                         batch_size=args.batch_size,
                                         seq_len=config['seq_len'], gen_emb_dim=args.gen_emb_dim,
                                         mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads,
                                         hidden_dim=args.hidden_dim,
                                         start_token=args.start_token, user_num=config['user_num'],
                                         product_num=config['product_num'],
                                         rating_num=5)

        discriminator = models.get_discriminator("amazon_attribute", batch_size=args.batch_size,
                                                 seq_len=config['seq_len'],
                                                 vocab_size=config['vocabulary_size'],
                                                 dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)

        from real.real_gan.amazon_attribute_train import amazon_attribute_train
        amazon_attribute_train(generator, discriminator, oracle_loader, config, args)
    elif args.dataset in ['CustomerReviews', 'imdb']:
        from real.real_gan.loaders.custom_reviews_loader import RealDataCustomerReviewsLoader
        from real.real_gan.customer_reviews_train import customer_reviews_train
        # custom dataset selected
        if args.dataset == 'CustomerReviews':
            data_dir = resources_path(config['data_dir'], "MovieReviews", "cr")
        elif args.dataset == 'imdb':
            data_dir = resources_path(config['data_dir'], "MovieReviews", 'movie', 'sstb')
        else:
            raise ValueError
        sample_dir = resources_path(config['sample_dir'])
        oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(args.dataset))
        train_file = os.path.join(data_dir, 'train.csv')

        # create_tokens_files(data_files=[train_file, dev_file, test_file])
        config_file = load_json(os.path.join(data_dir, 'config.json'))
        config = {**config, **config_file}  # merge dictionaries

        oracle_loader = RealDataCustomerReviewsLoader(args.batch_size, args.seq_len)
        oracle_loader.create_batches(data_file=[train_file])
        oracle_loader.model_index_word_dict = load_json(join(data_dir, 'index_word_dict.json'))
        oracle_loader.model_word_index_dict = load_json(join(data_dir, 'word_index_dict.json'))

        generator = models.get_generator("CustomerReviews", vocab_size=config['vocabulary_size'],
                                         batch_size=args.batch_size, start_token=args.start_token,
                                         seq_len=config['seq_len'], gen_emb_dim=args.gen_emb_dim,
                                         mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads,
                                         hidden_dim=args.hidden_dim,
                                         sentiment_num=config['sentiment_num'])

        discriminator_positive = models.get_discriminator("CustomerReviews", scope="discriminator_positive",
                                                          batch_size=args.batch_size,
                                                          seq_len=config['seq_len'],
                                                          vocab_size=config['vocabulary_size'],
                                                          dis_emb_dim=args.dis_emb_dim,
                                                          num_rep=args.num_rep, sn=args.sn)

        discriminator_negative = models.get_discriminator("CustomerReviews", scope="discriminator_negative",
                                                          batch_size=args.batch_size,
                                                          seq_len=config['seq_len'],
                                                          vocab_size=config['vocabulary_size'],
                                                          dis_emb_dim=args.dis_emb_dim,
                                                          num_rep=args.num_rep, sn=args.sn)

        customer_reviews_train(generator, discriminator_positive, discriminator_negative, oracle_loader, config, args)
    else:
        raise NotImplementedError('{}: unknown dataset!'.format(args.dataset))

    print("RUN FINISHED")
    return


if __name__ == '__main__':
    main()
