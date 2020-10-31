# coding: utf-8
from __future__ import print_function
#*********************************************************************************************
#                                                                                            *
#    訓練データ、ラベルデータ等をロードする                                                  *
#                                                                                            *
#*********************************************************************************************
def load_data() :
    #import pickle
    #import numpy as np
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
    print(e_train.shape,d_train.shape,t_train.shape)
    return e_train,e_test,d_train,d_test,t_train,t_test, maxlen_e, maxlen_d, words


#*********************************************************************************************
#                                                                                            *
#    訓練処理                                                                                *
#                                                                                            *
#*********************************************************************************************

def prediction(epochs, batch_size ,input_dim, param_name, e, e_t, d, d_t, t, t_t) :
    #import math
    #from dialog_categorize import Dialog

    vec_dim = 400
    #input_dim = len(words)
    output_dim = math.ceil(input_dim / 8)
    n_hidden = int(vec_dim*1.5 ) #隠れ層の次元

    prediction = Dialog(maxlen_e,maxlen_d,n_hidden,input_dim,vec_dim,output_dim)
    emb_param = param_name+'.hdf5'
    row = e.shape[0]

    e_train = e.reshape(row,maxlen_e)
    d_train = d.reshape(row,maxlen_d)
    t_train = t

    #t_train = t_train.reshape(row,maxlen_d)
    model = prediction.train(e_train, d_train,t_train,batch_size,epochs,emb_param)
    plot_model(model, show_shapes=True,to_file='seq2seq0212.png') #ネットワーク図出力

    model.save_weights(emb_param)                                #学習済みパラメータセーブ

    row2 = e_t.shape[0]
    e_test = e_t.reshape(row2,maxlen_e)
    d_test = d_t.reshape(row2,maxlen_d)
    t_test = t_t
    print()
    print(t_test.shape)
    perplexity = prediction.eval_perplexity(model,e_test,d_test,t_test,batch_size) 
    print('Perplexity=',perplexity)

    del prediction

#*********************************************************************************************
#                                                                                            *
#    メイン処理                                                                              *
#                                                                                            *
#*********************************************************************************************

if __name__ == '__main__':

    from dialog_categorize import Dialog

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

    from keras.utils import plot_model
    from pyknp import Juman
    Jumanpp = Juman()
    import codecs

    args = sys.argv
    #args = ['','param_001','40','500']                                    # jupyter上で実行するとき用

    param_name = args[1]
    epochs = int(args[2])
    batch_size = int(args[3])

    e_train,e_test,d_train,d_test,t_train,t_test, maxlen_e, maxlen_d, words = load_data() 

    input_dim = len(words)

    prediction(epochs, batch_size ,input_dim, param_name, e_train,e_test,d_train,d_test,t_train,t_test) 

