import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


from OneStep import OneStep


class EpochsStatsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, config, n_char = 500, temperature = 1):
        self.save_dir = save_dir
        self.config = config
        self.result_to_save = {}
        self.n_char = n_char
        self.temperature = temperature

    def on_epoch_end(self, epoch, logs={}):
        # String Lookup layer, which assign to every char an id
        ids_from_chars = preprocessing.StringLookup(vocabulary=self.config['vocab'], mask_token=None)
        # String lookup layer with invert = true, so from id it return to char given the vocabulary
        chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        one_step_model = OneStep(self.model, chars_from_ids, ids_from_chars, temperature=1)

        states = None
        next_char = tf.constant(['ROMEO:'])
        result = [next_char]

        for n in range(500):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        self.result_to_save[epoch] = ({
            "loss": logs["loss"],
            "accuracy": logs["accuracy"],
            "text": result[0].numpy().decode('utf-8')
        })


    def on_train_end(self, logs=None):
        with open(os.path.join(self.save_dir, "preds_evolution.json"), "w") as evolution_file:
            json.dump(self.result_to_save, evolution_file)

