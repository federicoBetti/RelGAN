# Load the last model and run inference (generate sentences) using topic extracted from the file taken as input
import argparse
import os

from path_resolution import resources_path
from real.real_gan.loaders.real_loader import RealDataTopicLoader
from run import create_subsample_data_file
from utils.inference_utils import inference_main
from utils.text_process import text_precess
from utils.utils import pp

parser = argparse.ArgumentParser(description='Use the model trained with input')

# Model File
parser.add_argument('--model-name', default="last_model", type=str,
                    help='Name of the model folder in the trained models folder')
parser.add_argument('--input-name', default='input.txt', type=str,
                    help='Name of the file from which the starting sentences should be taken')
parser.add_argument('--topic', default=True, action='store_true', help='If to use topic models or not')
parser.add_argument('--topic-number', default=9, type=int, help='How many topic to use in the LDA')

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
parser.add_argument('--n-topic-pre-epochs', default=300, type=int,
                    help='Number of steps to run pre-training for the topic discriminator')
parser.add_argument('--npre-epochs', default=150, type=int, help='Number of steps to run pre-training')
parser.add_argument('--nadv-steps', default=5000, type=int, help='Number of steps to run adversarial training')
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

# relational memory
parser.add_argument('--mem-slots', default=1, type=int, help="memory size")
parser.add_argument('--head-size', default=512, type=int, help="head size or memory size")
parser.add_argument('--num-heads', default=2, type=int, help="number of heads")

# Data
parser.add_argument('--dataset', default='emnlp_news', type=str, help='[oracle, image_coco, emnlp_news]')
parser.add_argument('--vocab-size', default=5000, type=int, help="vocabulary size")
parser.add_argument('--start-token', default=0, type=int, help="start token for a sentence")
parser.add_argument('--seq-len', default=20, type=int, help="sequence length: [20, 40]")
parser.add_argument('--num-sentences', default=100, type=int, help="number of total sentences")
parser.add_argument('--gen-emb-dim', default=32, type=int, help="generator embedding dimension")
parser.add_argument('--dis-emb-dim', default=64, type=int, help="TOTAL discriminator embedding dimension")
parser.add_argument('--num-rep', default=64, type=int, help="number of discriminator embedded representations")
parser.add_argument('--data-dir', default=os.path.join('.', 'data'), type=str, help='Where data data is stored')


def main():
    args = parser.parse_args()
    pp.pprint(vars(args))
    config = vars(args)
    model_path = resources_path(os.path.join('trained_models', config['model_name']))
    input_path = resources_path(os.path.join('inference_data', config['input_name']))

    data_file = resources_path(args.data_dir, '{}.txt'.format(args.dataset))
    sample_dir = resources_path(config['sample_dir'])
    oracle_file = os.path.join(sample_dir, 'oracle_{}.txt'.format(args.dataset))

    if args.dataset == 'emnlp_news' :
        data_file, lda_file = create_subsample_data_file(data_file)
    else:
        lda_file = data_file

    seq_len, vocab_size, word_index_dict, index_word_dict = text_precess(data_file, oracle_file=oracle_file)
    print(index_word_dict)
    config['seq_len'] = seq_len
    config['vocab_size'] = vocab_size
    print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))

    if config['topic']:
        topic_number = config['topic_number']
        oracle_loader = RealDataTopicLoader(args.batch_size, args.seq_len)
        oracle_loader.set_dataset(args.dataset)
        oracle_loader.topic_num = topic_number
        oracle_loader.set_dictionaries(word_index_dict, index_word_dict)
        oracle_loader.get_LDA(word_index_dict, index_word_dict, data_file)
        print(oracle_loader.model_index_word_dict)
        inference_main(oracle_loader, config, model_path, input_path)


if __name__ == '__main__':
    main()
