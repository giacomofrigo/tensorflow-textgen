import json
import os

from EpochsStatsCallback import EpochsStatsCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import time

from Model import Model


def train():
    PATH_TO_FILE = "data/shakespeare/input.txt"
    SEQ_LENGTH = 100
    EMBEDDING_DIM = 256
    RNN_UNITS = 256
    CHECKPOINT_DIR = "save"
    EPOCHS = 2

    os.system("rm -rf save")
    os.system("mkdir save")



    # Read, then decode for py2 compat.
    text = open(PATH_TO_FILE, 'rb').read().decode(encoding='utf-8')

    # The unique characters in the file
    vocab = sorted(set(text))

    ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab), mask_token=None)

    # EXPORT MODEL CONFIG
    settings = {
        'seq_length': SEQ_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'rnn_units': RNN_UNITS,
        'vocab': ids_from_chars.get_vocabulary()
    }

    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w") as settings_file:
        print("exporting config file", os.path.join(CHECKPOINT_DIR, "config.json"))
        json.dump(settings, settings_file)



    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    examples_per_epoch = len(text) // (SEQ_LENGTH + 1)

    sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (
        dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    model = Model(
        # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=EMBEDDING_DIM,
        rnn_units=RNN_UNITS)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    epochs_stats_callback = EpochsStatsCallback(CHECKPOINT_DIR, settings)

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback, epochs_stats_callback])

    model.summary()

if __name__ == '__main__':
    train()