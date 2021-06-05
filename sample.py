import argparse
import os

from six import text_type

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import time

from Model import Model
from OneStep import OneStep


parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('save_dir', type=str, default='save',
                    help='checkpoints and configurations directory')
parser.add_argument('-n', type=int, default=500,
                    help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default=u':',
                    help='prime text')
parser.add_argument('--temperature', type=float, default=1,
                    help='sampling temperature')

args = parser.parse_args()



def sample(args):
    latest = tf.train.latest_checkpoint(args.save_dir)
    with open(os.path.join(args.save_dir, "config.json"), "r") as config_file:
        config = json.load(config_file)

    # String Lookup layer, which assign to every char an id
    ids_from_chars = preprocessing.StringLookup(vocabulary=config['vocab'], mask_token=None)
    # String lookup layer with invert = true, so from id it return to char given the vocabulary
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    model = Model(vocab_size=len(ids_from_chars.get_vocabulary()),
                  embedding_dim=config["embedding_dim"],
                  rnn_units=config["rnn_units"])

    model.load_weights(latest).expect_partial()
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars, temperature=args.temperature)

    start = time.time()
    states = None
    next_char = tf.constant([args.prime])
    result = [next_char]

    for n in range(args.n):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)


if __name__ == '__main__':
    sample(args)