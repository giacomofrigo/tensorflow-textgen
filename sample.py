import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import time

from Model import Model
from OneStep import OneStep


def sample():
    SAVE_DIR = "save-second"
    PRIME = "ROMEO:"
    N_CHAR = 500
    TEMPERATURE = 1

    latest = tf.train.latest_checkpoint(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, "config.json"), "r") as config_file:
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
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars, temperature=TEMPERATURE)

    start = time.time()
    states = None
    next_char = tf.constant([PRIME])
    result = [next_char]

    for n in range(N_CHAR):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)


if __name__ == '__main__':
    sample()