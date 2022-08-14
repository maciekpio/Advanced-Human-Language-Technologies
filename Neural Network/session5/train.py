#! /usr/bin/python3

import code
import sys
from contextlib import redirect_stdout

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras.optimizers import RMSprop

from dataset import *
from codemaps import *

def build_network(codes) :

   # sizes
   n_words = codes.get_n_words()
   n_sufs = codes.get_n_sufs()
   n_prefs = codes.get_n_prefs()
   n_labels = codes.get_n_labels()  
   max_len = codes.maxlen
   n_lem = codes.get_n_lem()
   n_lab1 = codes.get_n_lab1()
   n_lc = codes.get_n_lc()

   inptW = Input(shape=(max_len,)) # word input layer & embeddings
   embW = Embedding(input_dim=n_words, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptW)  
   
   inptS = Input(shape=(max_len,))  # suf input layer & embeddings
   embS = Embedding(input_dim=n_sufs, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptS) 
                    
   inptP = Input(shape=(max_len,))  # pref input layer & embeddings
   embP = Embedding(input_dim=n_prefs, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptP)

   inptL = Input(shape=(max_len,))  # lemma input layer & embeddings
   embL = Embedding(input_dim=n_lem, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptL)

   inptLab1 = Input(shape=(max_len,))  # specific features input layer & embeddings
   embLab1 = Embedding(input_dim=n_lab1, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptLab1)

   inptLC = Input(shape=(max_len,))  # lowercase input layer & embeddings
   embLC = Embedding(input_dim=n_lc, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptLC)


   dropW = Dropout(0.1)(embW)
   dropS = Dropout(0.1)(embS)
   dropP = Dropout(0.1)(embP)
   dropL = Dropout(0.1)(embL)
   dropLab1 = Dropout(0.1)(embLab1)
   dropLC = Dropout(0.1)(embLC)
   drops = concatenate([dropW, dropS,dropP,dropL,dropLab1,dropLC])

   # biLSTM   
   bilstm = Bidirectional(LSTM(units=32, return_sequences=True,
                               recurrent_dropout=0.1))(drops) 
   # output softmax layer
   out = TimeDistributed(Dense(n_labels, activation="softmax"))(bilstm)

   # build and compile model
   model = Model([inptW,inptS,inptP,inptL,inptLab1,inptLC], out)
   model.compile(optimizer=RMSprop(lr=0.001, epsilon=None, decay=0.0),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
   
   return model
   


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 100
suf_len = 5
pref_len= 4
codes  = Codemaps(traindata, max_len, suf_len, pref_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr) :
   model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr) :
   model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv,Yv), verbose=1)

# save model and indexs
model.save(modelname)
codes.save(modelname)