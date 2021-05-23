import tensorflow as tf
from tensorflow.keras import layers
import random
from konlpy.tag import Okt

EPOCHS=200
NUM_WORDS=2000

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder,self).__init__()
        self.emb=layers.Embedding(NUM_WORDS,64)
        self.lstm=layers.LSTM(512,return_state=True)
        
    def call(self,x,training=False, mask=None):
        x=self.emb(x)
        _,h,c=self.lstm(x)
        
        return h,c
      
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder,self).__init__()
        self.emb=layers.Embedding(NUM_WORDS,64)
        self.lstm=layers.LSTM(512, return_sequence=True, return_state=True)
        self.dense=layers.Dense(NUM_WORDS,activation='softmax')
    def call(self,inputs,training=False, mask=None):
        x,h,c=inputs
        x=self.emb(x)
        x,h,c=self.lstm(x,initial_state=[h,c])
        
        return self.dense(x),h,c
         
 class Seq2seq(tf.keras.Model):
    def __init__(self,sos,eos):
        super(Seq2seq,self).__init__()
        self.enc=Encoder()
        self.dec=Decoder()
        self.sos=sos
        self.eos=eos
        
    def call(self,inputs,training=False,mask=None):
        if training is True:
            x,y=inputs
            h,c=self.enc(x)
            y,_,_=self.dec((y,h,c))
            
            return y
        else:
            x=inputs
            h,c=self.enc(x)
            y=tf.convert_to_tensor(self.sos)
            y=tf.reshape(y,(1,1))
            
            seq=tf.TensorArray(tf.int32,64)
            
            for idx in tf.range(64):
                y,h,c=self.dec([y,h,c])
                y=tf.cast(tf.argmax(y,axis=-1),dtype=tf.int32)
                y=tf.reshape(y,(1,1))
                seq=seq.write(idx,y)
                
                if y==self.eos:
                    break
                    
            return tf.reshape(seq.stack(),(1,64))
