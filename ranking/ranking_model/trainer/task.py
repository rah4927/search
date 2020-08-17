#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:35:16 2019

@author: rah4927
"""

import keras
import numpy as np
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Reshape, Concatenate, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import tensorflow as tf 
import argparse 
from tensorflow.python.lib.io import file_io
import pickle 

def load_obj(name):
    with file_io.FileIO(name,'rb') as f:
             return pickle.load(f)


def model(query_embedding_matrix, product_embedding_matrix):
        
    q_em = Embedding(input_dim=query_embedding_matrix.shape[0],
                                  output_dim=query_embedding_matrix.shape[1], 
                                  input_length=1,
                                  weights=[query_embedding_matrix], 
                                  trainable=False)
    
    d_em = Embedding(input_dim=product_embedding_matrix.shape[0],
                                  output_dim=product_embedding_matrix.shape[1], 
                                  input_length=1,
                                  weights=[product_embedding_matrix], 
                                  trainable=False)
    
    q = Input((1,), name='Q')
    d = Input((1,), name='D')
    
    x = Concatenate(name = 'concat-2')([q_em(q), d_em(d)])
    x = Dense(1000, activation='relu', name = 'dense-N')(x)
    N = Model([q, d], Dense(1)(x), name = 'N')
    
    q = Input((1,), name = 'Q')
    d1 = Input((1,), name = 'D1')
    d2 = Input((1,), name = 'D2')
    
    x = Concatenate(name = 'concat-1')([N([q, d1]), N([q, d2])])
    x = Dense(1, trainable=False, weights=[np.array([[1], [-1]])], use_bias=False, name = 'dense-1')(x)
    x = Activation('sigmoid', name = 'activation-M')(x)
    x = Flatten()(x)
    M = Model([q, d1, d2], x, name = 'M')
    return M 
    
    


def main(job_dir, query_embeddings, product_embeddings, X_train_data, X_test_data, y_train_data, y_test_data):
    
    with tf.device('/device:GPU:0'):
        
        # query embedding matrix 
        q = np.asarray(load_obj(query_embeddings))
        query_embedding_matrix = np.array(q)
        
        # product embeddings 
        p = np.asarray(load_obj(product_embeddings))
        product_embedding_matrix = np.array(p)
        
        data = [query_embedding_matrix, product_embedding_matrix]
        
        M = model(*data)
        
        X_train = load_obj(X_train_data)
        X_test = load_obj(X_test_data)
        y_train = load_obj(y_train_data)
        y_test = load_obj(y_test_data)
                
        M.compile(optimizer=Adam(clipnorm=0.5), loss='binary_crossentropy', metrics =['accuracy'])
        
        M.fit(X_train, y_train, batch_size=10000, epochs=1, validation_data=[X_test, y_test])
        
        # Save model.h5 on to google storage
        Model.save('ranking-model.h5')
        with file_io.FileIO('ranking-model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/ranking-model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    
    parser.add_argument(
      '--query-embeddings',
      help = 'eval data',
      required = True
    )
    
    parser.add_argument(
      '--product-embeddings',
      help = 'eval data',
      required = True
    )
    
        
    parser.add_argument(
      '--X-train-data',
      help = 'training data',
      required = True
    )
    
    parser.add_argument(
      '--X-test-data',
      help = 'eval data',
      required = True
    )
    
    parser.add_argument(
      '--y-train-data',
      help = 'eval data',
      required = True
    )
    
    parser.add_argument(
      '--y-test-data',
      help = 'eval data',
      required = True
    )
    
    
    
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)