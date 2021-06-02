import argparse
import json
import os

from EpochsStatsCallback import EpochsStatsCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


from Model import Model




parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument("input_file", type=str, default="data/shakespeare/input.txt")
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models and model configuration')
parser.add_argument('--log_dir', type=str, default='{save_dir}/logs',
                    help='directory to store tensorflow logs, that can be used for tensorboard')
parser.add_argument('--validation_split', type=int, default=0.1,
                    help='dimension of the validation set (in batches)')
# Model params
parser.add_argument('--embedding_dim', type=int, default=256,
                    help='Dimension of the embedding layer which is the input layer. '
                         'A trainable lookup table that will map each character-ID to a vector with args.embedding_dim dimensions')
parser.add_argument('--rnn_units', type=int, default=256,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
# Optimization
parser.add_argument('--seq_length', type=int, default=100,
                    help='RNN sequence length. Number of timesteps to unroll for.')
parser.add_argument('--batch_size', type=int, default=64,
                    help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='probability of keeping weights in the hidden layer')
args = parser.parse_args()



def train(args):

    if os.path.isdir(args.save_dir):
        os.system("rm -rf {}".format(args.save_dir))
    os.system("mkdir {}".format(args.save_dir))

    args.log_dir = args.save_dir + "/logs"

    # Read, then decode for py2 compat.
    assert os.path.isfile(args.input_file), "no file {} was found".format(args.input_file)
    text = open(args.input_file, 'rb').read().decode(encoding='utf-8')

    # The unique characters in the file
    vocab = sorted(set(text))

    ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab), mask_token=None)

    # EXPORT MODEL CONFIG
    settings = {
        'seq_length': args.seq_length,
        'embedding_dim': args.embedding_dim,
        'rnn_units': args.rnn_units,
        'vocab': ids_from_chars.get_vocabulary()
    }

    with open(os.path.join(args.save_dir, "config.json"), "w") as settings_file:
        print("exporting config file", os.path.join(args.save_dir, "config.json"))
        json.dump(settings, settings_file)



    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    examples_per_epoch = len(text) // (args.seq_length + 1)

    sequences = ids_dataset.batch(args.seq_length + 1, drop_remainder=True)

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

    # split dataset into validation and train
    # get dataset length

    dataset_length = tf.data.experimental.cardinality(dataset).numpy()
    validation_size = int(args.validation_split * dataset_length)

    assert validation_size > 1, "dataset too small"

    train_dataset = dataset.take(dataset_length-validation_size)
    validation_dataset = dataset.skip(dataset_length-validation_size)


    model = Model(
        # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        n_layers=args.num_layers,
        dropout=args.dropout)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(args.save_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    epochs_stats_callback = EpochsStatsCallback(args.save_dir, settings)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir, histogram_freq=1)

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=args.num_epochs, callbacks=[checkpoint_callback, epochs_stats_callback, tensorboard_callback])

    model.summary()

if __name__ == '__main__':
    train(args)