import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.models import load_model,Model,model_from_json
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import GRU,Embedding,Activation,ReLU,AveragePooling2D,MaxPool2D,BatchNormalization,Conv1D,Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling,RandomUniform

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, roc_curve,auc,precision_recall_curve
from tensorflow.keras.initializers import glorot_normal
import shutil
import matplotlib.pyplot as plt
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization
from keras_bert import get_custom_objects
from tensorflow.python.keras.layers.core import Reshape, Permute
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import multiply
from tensorflow.python.keras.layers.core import Dense, Dropout, Lambda, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.compat.v1 import keras


def transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, 2)
    ytest = to_categorical(ytest, 2)
    yval = to_categorical(yval, 2)
    return xtrain, xtest, ytrain, ytest,xval,yval



class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len=None, embed_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
    def call(self, a):
        position_embedding = np.array([[pos / np.power(10000, 2. * i / self.embed_dim) for i in range(self.embed_dim)] for pos in range(self.seq_len)])
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        return position_embedding + a
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({'seq_len' : self.seq_len,'embed_dim' : self.embed_dim,})
    #     return config


def AttentionBlock(head_num,pos_embedding):
	initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
	attention_output = MultiHeadAttention(head_num=head_num)([pos_embedding, pos_embedding, pos_embedding])
	laynorm1 = LayerNormalization()(attention_output + pos_embedding)
	linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
	linear1 = Dropout(0.5)(linear1)
	linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
	linear2 = Dropout(0.5)(linear2)
	laynorm2 = LayerNormalization()(laynorm1 + linear2)
	laynorm2 = Dropout(0.5)(laynorm2)
	return laynorm2

def Blocks(num,head_num,pos_embedding):
	x = pos_embedding
	for i in range(num):
		x = AttentionBlock(head_num,x)
	return x
	
def new_model(input_size):
  initializer = VarianceScaling(mode='fan_avg', distribution='normal') # uniform fan_avg
  input_value = Input(shape=input_size)
  head_num = 4
  print('1:',input_value.shape) #1, 19, 4
  input_value2 = tf.transpose(input_value, perm=[0, 2, 3, 1])
  print('2:',input_value2.shape) #19, 4, 1
  conv_1_output = Conv2D(64, (1, input_size[2]), activation='relu', padding='valid', data_format='channels_last',kernel_initializer=initializer)(input_value2)
  print('3:',conv_1_output.shape) #19, 1, 64
  #conv_1_output = BatchNormalization()(conv_1_output)
  #conv_1_output = Dropout(0.5)(conv_1_output)
  conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
  print('4:',conv_1_output_reshape.shape) # 19, 64
  conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape) # 19, 32
  print('5:',conv_1_output_reshape_average.shape) # 19, 32
  conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape) # 19, 32
  print('6:',conv_1_output_reshape_max.shape) # 19, 32
  input_value1 = Reshape((input_size[1],input_size[2]))(input_value) # 19, 4# input_value1 = Reshape((input_size[1],input_size[2]))(input_value)
  print('7:',input_value1.shape) # 19, 4
  #print('8:',Concatenate(axis=-1)([input_value1,conv_1_output_reshape_average, conv_1_output_reshape_max]).shape)
  # 19, 68
  bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.5,activation='relu',kernel_initializer=initializer))(Concatenate(axis=-1)([input_value1,conv_1_output_reshape_average, conv_1_output_reshape_max])) # (input_value1) 
  #bidirectional_1_output = Activation('relu')(bidirectional_1_output)
  # 19, 64
  #bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
  #bidirectional_1_output_ln = Dropout(0.5)(bidirectional_1_output_ln)
  pos_embedding = PositionalEncoding(seq_len=input_size[1], embed_dim=64)(bidirectional_1_output)
  # 19, 64
  attention_out = Blocks(1,head_num,pos_embedding) # 19, 64
  flatten_output = Flatten()(attention_out) # 1216
  linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output)) # 256
  #linear_1_output = BatchNormalization()(linear_1_output)
  linear_1_output_dropout = Dropout(0.5)(linear_1_output)
  #linear_1_output_dropout = linear_1_output
  linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output_dropout)
  linear_2_output_dropout = Dropout(0.5)(linear_2_output)
  #linear_2_output_dropout = linear_2_output
  linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
  #linear_3_output_dropout = Dropout(0.5)(linear_3_output) 
  linear_3_output_dropout = linear_3_output
  #   linear_4_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_3_output_dropout)
  #   linear_4_output_dropout = Dropout(0.5)(linear_4_output) 
  #   linear_5_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_4_output_dropout)
  #   linear_5_output_dropout = Dropout(0.25)(linear_5_output) 
  model = Model(input_value, linear_3_output)
  return model


#        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')  
#        input_value = Input(shape=input_size)  
#        conv_1_output = Conv2D(10, (1, 7), padding='same', activation='relu',kernel_initializer=initializer)(input_value)
#        conv_1_output_bn = BatchNormalization()(conv_1_output)
#        conv_1_output = Dropout(0.5)(conv_1_output_bn)
#        conv_2_output = Conv2D(10, (1, 6), padding='same',activation='relu',kernel_initializer=initializer)(input_value)
#        conv_2_output_bn = BatchNormalization()(conv_2_output)

#        conv_2_output = Dropout(0.5)(conv_2_output_bn)
#        conv_3_output = Conv2D(10, (1, 5), padding='same',activation='relu',kernel_initializer=initializer)(input_value)
#        conv_3_output_bn = BatchNormalization()(conv_3_output)
#        conv_3_output = Dropout(0.5)(conv_3_output_bn)
#        conv_4_output = Conv2D(10, (1, 4), padding='same',activation='relu',kernel_initializer=initializer)(input_value)
#        conv_4_output_bn = BatchNormalization()(conv_4_output)
#        conv_4_output = Dropout(0.5)(conv_4_output_bn)
#        #branches = [input_value, conv_1_output_bn, conv_2_output_bn, conv_3_output_bn, conv_4_output_bn]
#        branches = [input_value, conv_1_output, conv_2_output, conv_3_output, conv_4_output]
#        mixed = Concatenate(axis=-1)(branches)
#        mixed = Reshape((input_size[1], 54))(mixed)
#        bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True,dropout=0.5, input_size=(input_size[1], 54), kernel_initializer=initializer))(mixed)
#        conv_2_output = Conv2D(64, (1, input_size[-1]), activation='relu', padding='valid', data_format='channels_first',kernel_initializer=initializer)(input_value)
#        conv_2_output = BatchNormalization()(conv_2_output)
#        conv_2_output = Dropout(0.5)(conv_2_output)
#        conv_2_output_reshape = Reshape(tuple([x for x in conv_2_output.shape.as_list() if x != 1 and x is not None]))(conv_2_output)
#        conv_2_output_reshape2 = tf.transpose(conv_2_output_reshape, perm=[0, 2, 1])
#        conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2_output_reshape2)
#        conv_2_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_2_output_reshape2)
#
#        conv_3_output = Conv2D(64, (1, input_size[-1]), activation='relu', padding='valid', data_format='channels_first',kernel_initializer=initializer)(input_value)
#        conv_3_output = BatchNormalization()(conv_3_output)
#        conv_3_output = Dropout(0.5)(conv_3_output)
#        conv_3_output_reshape = Reshape(tuple([x for x in conv_3_output.shape.as_list() if x != 1 and x is not None]))(conv_3_output)
#        conv_3_output_reshape2 = tf.transpose(conv_3_output_reshape, perm=[0, 2, 1])
#        conv_3_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_3_output_reshape2)
#        conv_3_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_3_output_reshape2)

#   attention_1_output = MultiHeadAttention(head_num=head_num)([pos_embedding, pos_embedding, pos_embedding])
#   residual1 = attention_1_output + pos_embedding
#   laynorm1 = LayerNormalization()(residual1)
#   linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
#   linear1 = Dropout(0.5)(linear1)
#   linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
#   linear2 = Dropout(0.5)(linear2)
#   residual2 = laynorm1 + linear2
#   laynorm2 = LayerNormalization()(residual2)
#   laynorm2 = Dropout(0.5)(laynorm2)
#   
#   attention_2_output = MultiHeadAttention(head_num=head_num)([laynorm2, laynorm2, laynorm2])
#   residual3 = attention_2_output + laynorm2
#   laynorm3 = LayerNormalization()(residual3)
#   linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
#   linear3 = Dropout(0.5)(linear3)
#   linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
#   linear4 = Dropout(0.5)(linear4)
#   residual4 = laynorm3 + linear4
#   laynorm4 = LayerNormalization()(residual4)
#   laynorm4 = Dropout(0.5)(laynorm4)
#
#   attention_3_output = MultiHeadAttention(head_num=head_num)([laynorm4, laynorm4, laynorm4])
#   residual5 = attention_3_output + laynorm4
#   laynorm5 = LayerNormalization()(residual5)
#   linear5 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm5)
#   linear5 = Dropout(0.5)(linear5)
#   linear6 = Dense(64, activation='relu', kernel_initializer=initializer)(linear5)
#   linear6 = Dropout(0.5)(linear6)
#   residual6 = laynorm5 + linear6
#   laynorm6 = LayerNormalization()(residual6)
#   laynorm6 = Dropout(0.5)(laynorm6)
#   
#   attention_4_output = MultiHeadAttention(head_num=head_num)([laynorm6, laynorm6, laynorm6])
#   residual7 = attention_4_output + laynorm6
#   laynorm7 = LayerNormalization()(residual7)
#   linear7 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm7)
#   linear7 = Dropout(0.5)(linear7)
#   linear8 = Dense(64, activation='relu', kernel_initializer=initializer)(linear7)
#   linear8 = Dropout(0.5)(linear8)
#   residual8 = laynorm7 + linear8
#   laynorm8 = LayerNormalization()(residual8)
#   laynorm8 = Dropout(0.5)(laynorm8)

#   attention_5_output = MultiHeadAttention(head_num=head_num)([laynorm8, laynorm8, laynorm8])
#   residual9 = attention_5_output + laynorm8
#   laynorm9 = LayerNormalization()(residual9)
#   linear9 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm9)
#   linear9 = Dropout(0.5)(linear9)
#   linear10 = Dense(64, activation='relu', kernel_initializer=initializer)(linear9)
#   linear10 = Dropout(0.5)(linear10)
#   residual10 = laynorm9 + linear10
#   laynorm10 = LayerNormalization()(residual10)
#   laynorm10 = Dropout(0.5)(laynorm10)
