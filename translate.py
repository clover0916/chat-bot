from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
import random
import numpy.random as nr
import keras
import sys
import math
import time
import pickle
import gc
import os

from keras.layers.core import Dense
from keras.layers.core import Masking
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model

from pyknp import Jumanpp
import codecs

#単語ファイルロード
with open('words.pickle', 'rb') as ff :
    words=pickle.load(ff)         

#Encoder Inputデータをロード
with open('e.pickle', 'rb') as f :
    e = pickle.load(f)

#Decoder Inputデータをロード
with open('d.pickle', 'rb') as g :
    d = pickle.load(g)

#ラベルデータをロード
with open('t.pickle', 'rb') as h :
    t = pickle.load(h)

#maxlenロード
with open('maxlen.pickle', 'rb') as maxlen :
    [maxlen_e, maxlen_d] = pickle.load(maxlen)


n_split=int(e.shape[0]*0.95)            #訓練データとテストデータを95:5に分割
e_train,e_test=np.vsplit(e,[n_split])   #エンコーダインプットデータを訓練用とテスト用に分割
d_train,d_test=np.vsplit(d,[n_split])   #デコーダインプットデータを訓練用とテスト用に分割
t_train,t_test=np.vsplit(t,[n_split])   #ラベルデータを訓練用とテスト用に分割

#単語ファイルロード
with open('words.pickle', 'rb') as ff :
    words=pickle.load(ff)         

#Encoder Inputデータをロード
with open('e.pickle', 'rb') as f :
    e = pickle.load(f)

#Decoder Inputデータをロード
with open('d.pickle', 'rb') as g :
    d = pickle.load(g)

#ラベルデータをロード
with open('t.pickle', 'rb') as h :
    t = pickle.load(h)

#maxlenロード
with open('maxlen.pickle', 'rb') as maxlen :
    [maxlen_e, maxlen_d] = pickle.load(maxlen)


n_split=int(e.shape[0]*0.95)            #訓練データとテストデータを95:5に分割
e_train,e_test=np.vsplit(e,[n_split])   #エンコーダインプットデータを訓練用とテスト用に分割
d_train,d_test=np.vsplit(d,[n_split])   #デコーダインプットデータを訓練用とテスト用に分割
t_train,t_test=np.vsplit(t,[n_split])   #ラベルデータを訓練用とテスト用に分割

vec_dim = 400
epochs = 10
batch_size = 80
input_dim = len(words)
output_dim = input_dim
n_hidden = int(vec_dim*1.5 ) #隠れ層の次元

prediction = Dialog(maxlen_e,maxlen_d,n_hidden,input_dim,vec_dim,output_dim)
emb_param = 'param_seq2seq0212.hdf5'
row = e_train.shape[0]
e_train = e_train.reshape(row,maxlen_e)
d_train = d_train.reshape(row,maxlen_d)
t_train = t_train.reshape(row,maxlen_d)
model = prediction.train(e_train, d_train,t_train,batch_size,epochs,emb_param)
plot_model(model, show_shapes=True,to_file='seq2seq0212.png') #ネットワーク図出力
model.save_weights(emb_param)                                #学習済みパラメータセーブ

row2 = e_test.shape[0]
e_test = e_test.reshape(row2,maxlen_e)
d_test = d_test.reshape(row2,maxlen_d)
#t_test=t_test.reshape(row2,maxlen_d)
print()
perplexity = prediction.eval_perplexity(model,e_test,d_test,t_test,batch_size) 
print('Perplexity=',perplexity)