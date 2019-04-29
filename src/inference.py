# Load the last model and run inference (generate sentences) using topic extracted from the file taken as input
import argparse

import tensorflow as tf
from path_resolution import resources_path

parser = argparse.ArgumentParser(description='Use the model trained with input')

# Model File
parser.add_argument('--model-name', default=None, type=str, help='Name of the model folder in the trained models folder')
parser.add_argument('--input-name', default='input.txt', type=str, help='Name of the file from which the starting sentences should be taken')

def main():
    tf.reset_default_graph()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")


if __name__ == '__main__':
    main()
