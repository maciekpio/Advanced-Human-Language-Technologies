#! /usr/bin/python3

import sys
import random
from contextlib import redirect_stdout
from tensorflow.keras import regularizers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPool1D, Reshape, Concatenate, Flatten, Bidirectional, LSTM
from tensorflow.keras.layers import concatenate, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from dataset import *
from codemaps import *
import tensorflow as tf
import numpy as np

def build_network(idx) :
   # get sizes of words, lc words and PoS tags
   n_words = codes.get_n_words()
   n_labels = codes.get_n_labels()
   n_lc = codes.get_n_lc_words()
   n_pos = codes.get_n_pos()
   # define maximum length of used words
   max_len = codes.maxlen

   # we adapted a suggested GloVe model to get initial embedding matrix
   # from https://github.com/suriak/sentence-classification-cnn/blob/master/sentence%20classifier.py
   def load_glove():
      """
      Load pre-trained glove vectors which is downloaded from http://nlp.stanford.edu/data/glove.6B.zip and saved in
      EMBED_DIR.
      :return: A dictionary with key as word and value as vectors.
      """
      embeddings_index = {}
      try:
         f = open('/content/glove.6B.100d.txt', encoding='utf-8')
      except FileNotFoundError:
         print("GloVe vectors missing. You can download from http://nlp.stanford.edu/data/glove.6B.zip")
         sys.exit()
      for line in f:
         values = line.rstrip().rsplit(' ')
         word = values[0]
         coefs = np.asarray(values[1:], dtype='float32')
         embeddings_index[word] = coefs
      f.close()
      print("\tNumber of Tokens from GloVe: %s" % len(embeddings_index))
      return embeddings_index

   def glove_embedding_matrix(embeddings_index):
        """
        Creates an embedding matrix for all the words(vocab) in the training data. An embedding matrix has shape
        (vocab * EMBEDDING_DIM), where each row represents a word in the training data and its corresponding vector
        in GloVe.
        Any word that is not present in GloVe but present in training data is represented by a random vector bounded
        between -0.25 and + 0.25
        :param embeddings_index: The dictionary of words and its corresponding vector representation.
        :return: A matrix of shape (vocab * EMBEDDING_DIM)
        """
        words_not_found = []
        vocab = len(codes.word_index)# + 1
        # embedding_matrix = np.zeros((vocab, self.EMBEDDING_DIM))
        embedding_matrix = np.random.uniform(-0.25, 0.25, size=(vocab, 100))  # 0.25 is chosen so
        # the unknown vectors have (approximately) same variance as pre-trained ones
        print("codes:")
        print(codes)
        for word, i in codes.word_index.items():
            if i >= vocab:
                continue
            embedding_vector = embeddings_index.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)
        # print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("\tShape of embedding matrix: %s" % str(embedding_matrix.shape))
        print("\tNo. of words not found in GloVe: ", len(words_not_found))
        return embedding_matrix

   # load downloaded GloVe
   emb_ind = load_glove()
   # Get the embedding matrix with matching size
   embedding_matrix = glove_embedding_matrix(emb_ind)
   
   # set the dimension for Embedding layers
   out_dim = 100

   # Define the Input-layers, which are afterwards concatenated
   # Only lc words and PoS tags helped to improve the model
   inptW = Input(shape=(max_len,))
   embW = Embedding(input_dim=n_words, output_dim=out_dim,
      weights=[embedding_matrix],
      input_length=max_len, mask_zero=False,trainable=True)(inptW)
   inptLC = Input(shape=(max_len,))
   embLC = Embedding(input_dim=n_lc, output_dim=out_dim,
      input_length=max_len, mask_zero=False,trainable=True)(inptLC)
   inptPos = Input(shape=(max_len,))
   embPos = Embedding(input_dim=n_lemmas, output_dim=out_dim,
      input_length=max_len, mask_zero=False,trainable=True)(inptPos)

   embs = concatenate([embW,embLC,embPos])

   # Define the CNN and MaxPooling layers with different filter sizes
   # As in https://towardsdatascience.com/cnn-sentiment-analysis-1d16b7c5a0e7
   filter_sizes = [2,3,4,5,10]
   convs = []
   for filter_size in filter_sizes:
    l_conv = Conv1D(filters=100, 
                    kernel_size=filter_size, 
                    activation='relu')(embs)
    l_pool = MaxPool1D()(l_conv)
    convs.append(l_pool)
   l_merge = Concatenate(axis=1)(convs)
  
   # Define the LSTM layer
   # Recurrent dropout set to 0.0, as otherwise GPU can't be utilized properly
   lstm = Bidirectional(LSTM(units=50, return_sequences=False,
    dropout=0.3,recurrent_dropout=0.0))(l_merge)
    
   # add a dropout layer
   x = Dropout(0.2)(lstm)
   
   # Define the output layer 
   out = Dense(n_labels, activation='softmax')(x)
   # Define the model with matching input and output
   model = Model([inptW,inptLC,inptPos], out)
   # Use a dedicated optimizer to test parameters of learning rate
   adam = Adam(learning_rate=0.001)
   # compile the model and return it
   model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
   return model

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
trainfile = sys.argv[1]
validationfile = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = 150
codes = Codemaps(traindata, max_len)

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
   model.fit(Xt, Yt, batch_size=32, epochs=12, validation_data=(Xv,Yv), verbose=1)
   
# save model and indexs
model.save(modelname)
codes.save(modelname)