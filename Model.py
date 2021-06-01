import tensorflow as tf

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units, n_layers = 2):
    super().__init__(self)
    self.n_layers=n_layers

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.rnn_layers = [tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True) for layer in range (n_layers)]

    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)

    if states is None:
      states = [self.rnn_layers[layer].get_initial_state(x) for layer in range(self.n_layers)]

    new_states=[None for layer in range(self.n_layers)]

    for layer_index, layer in enumerate(self.rnn_layers):
      x, new_state = layer(x, initial_state=states[layer_index], training=training)
      new_states[layer_index] = new_state

    x = self.dense(x, training=training)

    if return_state:
      return x, new_states
    else:
      return x