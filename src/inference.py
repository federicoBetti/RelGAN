# Load the last model and run inference (generate sentences) using topic extracted from the file taken as input
import argparse
import os

import tensorflow as tf
from path_resolution import resources_path

parser = argparse.ArgumentParser(description='Use the model trained with input')

# Model File
parser.add_argument('--model-name', default=None, type=str,
                    help='Name of the model folder in the trained models folder')
parser.add_argument('--input-name', default='input.txt', type=str,
                    help='Name of the file from which the starting sentences should be taken')


def main(model_path, input_path):
    tf.reset_default_graph()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_path)
        print("Model restored.")


if __name__ == '__main__':
    args = parser.parse_args()
    config = vars(args)
    model_path = resources_path(os.path.join('trained_models', config['model_name']))
    input_path = resources_path(os.path.join('inference_data', config['input_name']))
    main(model_path, input_path)
