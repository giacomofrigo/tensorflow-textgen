import tensorflow as tf

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.gru_2 = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)


    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = [self.gru.get_initial_state(x), self.gru_2.get_initial_state(x)]

    new_states=[None, None]
    x, new_states[0] = self.gru(x, initial_state=states[0], training=training)
    x, new_states[1] = self.gru(x, initial_state=states[1], training=training)

    x = self.dense(x, training=training)

    if return_state:
      return x, new_states
    else:
      return x