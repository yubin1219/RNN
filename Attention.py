import random
import tensorflow as tf
from konlpy.tag import Okt

EPOCHS = 200
NUM_WORDS = 2000

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

    def call(self, x, training=False, mask=None):
        x = self.emb(x)
        H, h, c = self.lstm(x)
        return H, h, c
      
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.att = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(NUM_WORDS, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x, s0, c0, H = inputs
        x = self.emb(x)
        S, h, c = self.lstm(x, initial_state=[s0, c0])
        
        S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)
        A = self.att([S_, H])
        y = tf.concat([S, A], axis=-1)
        
        return self.dense(y), h, c
      
 class Seq2seq(tf.keras.Model):
    def __init__(self, sos, eos):
        super(Seq2seq, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.sos = sos
        self.eos = eos

    def call(self, inputs, training=False, mask=None):
        if training is True:
            x, y = inputs
            H, h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c, H))
            return y
        else:
            x = inputs
            H, h, c = self.enc(x)
            
            y = tf.convert_to_tensor(self.sos)
            y = tf.reshape(y, (1, 1))

            seq = tf.TensorArray(tf.int32, 64)

            for idx in tf.range(64):
                y, h, c = self.dec([y, h, c, H])
                y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
                y = tf.reshape(y, (1, 1))
                seq = seq.write(idx, y)

                if y == self.eos:
                    break

            return tf.reshape(seq.stack(), (1, 64))
