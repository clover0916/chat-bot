# coding: utf-8


from keras.layers.core import Dense
from keras.layers.core import Masking
from keras.layers.core import Activation
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

import tensorflow as tf
import numpy as np
import csv
import random
import numpy.random as nr
import keras
import sys
import math
import pickle
import time
import gc
import os

class Color:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[38;5;10m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[38;5;14m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'

class Dialog :
    def __init__(self,maxlen_e,maxlen_d,n_hidden,input_dim,vec_dim,output_dim):
        self.maxlen_e=maxlen_e
        self.maxlen_d=maxlen_d
        self.n_hidden=n_hidden
        self.input_dim=input_dim
        self.vec_dim=vec_dim
        self.output_dim=output_dim

    #**************************************************************
    #                                                             *
    # ニューラルネットワーク定義                                  *
    #                                                             *
    #**************************************************************
    def create_model(self):
        print('#3')
        #=========================================================
        #エンコーダー（学習／応答文作成兼用）
        #=========================================================
        encoder_input = Input(shape=(self.maxlen_e,), dtype='int32', name='encorder_input')
        e_i = Embedding(output_dim=self.vec_dim, input_dim=self.input_dim, #input_length=self.maxlen_e,
                        mask_zero=True, 
                        embeddings_initializer=uniform(seed=20170719))(encoder_input)
        e_i=BatchNormalization(axis=-1)(e_i)
        e_i=Masking(mask_value=0.0)(e_i)
        e_i_fw1, state_h_fw1, state_c_fw1 =LSTM(self.n_hidden, name='encoder_LSTM_fw1'  , #前向き1段目
                                                return_sequences=True,return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                )(e_i) 
        encoder_LSTM_fw2 =LSTM(self.n_hidden, name='encoder_LSTM_fw2'  ,       #前向き2段目
                                                return_sequences=True,return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                dropout=0.5, recurrent_dropout=0.5
                                                )  

        e_i_fw2, state_h_fw2, state_c_fw2 = encoder_LSTM_fw2(e_i_fw1)
        e_i_bw0=e_i
        e_i_bw1, state_h_bw1, state_c_bw1 =LSTM(self.n_hidden, name='encoder_LSTM_bw1'  ,  #後ろ向き1段目
                                                return_sequences=True,return_state=True, go_backwards=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                )(e_i_bw0) 
        e_i_bw2, state_h_bw2, state_c_bw2 =LSTM(self.n_hidden, name='encoder_LSTM_bw2'  ,  #後ろ向き2段目
                                                return_sequences=True,return_state=True, go_backwards=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                dropout=0.5, recurrent_dropout=0.5
                                                )(e_i_bw1)            

        encoder_outputs = keras.layers.add([e_i_fw2,e_i_bw2],name='encoder_outputs')
        state_h_1=keras.layers.add([state_h_fw1,state_h_bw1],name='state_h_1')
        state_c_1=keras.layers.add([state_c_fw1,state_c_bw1],name='state_c_1')
        state_h_2=keras.layers.add([state_h_fw2,state_h_bw2],name='state_h_2')
        state_c_2=keras.layers.add([state_c_fw2,state_c_bw2],name='state_c_2')

        # Batch Normalization
        encoder_outputs = BatchNormalization(axis=-1)(encoder_outputs)
        state_h_1 = BatchNormalization(axis=-1)(state_h_1)
        state_c_1 = BatchNormalization(axis=-1)(state_c_1)
        state_h_2 = BatchNormalization(axis=-1)(state_h_2)
        state_c_2 = BatchNormalization(axis=-1)(state_c_2)

        encoder_states1 = [state_h_1,state_c_1] 
        encoder_states2 = [state_h_2,state_c_2]

        encoder_model = Model(inputs=encoder_input, 
                              outputs=[encoder_outputs,state_h_1,state_c_1,state_h_2,state_c_2])    #エンコーダモデル        


        print('#4')        
        #=========================================================
        #デコーダー（学習用）
        # デコーダを、完全な出力シークエンスを返し、内部状態もまた返すように設定します。
        # 訓練モデルではreturn_sequencesを使用しませんが、推論では使用します。
        #=========================================================
        a_states1=encoder_states1
        a_states2=encoder_states2
        #---------------------------------------------------------
        #レイヤー定義
        #---------------------------------------------------------
        decode_LSTM1 = LSTM(self.n_hidden, name='decode_LSTM1',
                            return_sequences=True, return_state=True,
                            kernel_initializer=glorot_uniform(seed=20170719), 
                            recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                            )
        decode_LSTM2 =LSTM(self.n_hidden, name='decode_LSTM2',
                           return_sequences=True, return_state=True,
                           kernel_initializer=glorot_uniform(seed=20170719), 
                           recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                           dropout=0.5, recurrent_dropout=0.5
                           )                  

        Dense1=Dense(self.n_hidden,name='Dense1',
                           kernel_initializer=glorot_uniform(seed=20170719))
        Dense2=Dense(self.n_hidden,name='Dense2',     #次元を減らす
                           kernel_initializer=glorot_uniform(seed=20170719))              
        a_Concat1=keras.layers.Concatenate(axis=-1)
        a_decode_input_slice1 = Lambda(lambda x: x[:,0,:],output_shape=(1,self.vec_dim,),name='slice1')
        a_decode_input_slice2 = Lambda(lambda x: x[:,1:,:],name='slice2')
        a_Reshape1 = keras.layers.Reshape((1,self.vec_dim))

        a_Dot1 = keras.layers.Dot(-1,name='a_Dot1')
        a_Softmax = keras.layers.Softmax(axis=-1,name='a_Softmax')
        a_transpose = keras.layers.Reshape((self.maxlen_e,1),name='Transpose') 
        a_Dot2 = keras.layers.Dot(1,name='a_Dot2')
        a_Concat2 = keras.layers.Concatenate(-1,name='a_Concat2')
        a_tanh = Lambda(lambda x: K.tanh(x),name='tanh')
        a_Concat3 = keras.layers.Concatenate(axis=-1,name='a_Concat3')

        decoder_Dense_cat = Dense(8, name='decoder_Dense_cat',activation='softmax' ,
                              kernel_initializer=glorot_uniform(seed=20170719)) 
        decoder_Dense_mod = Dense(self.output_dim, name='decoder_Dense_mod',activation='softmax' ,
                              kernel_initializer=glorot_uniform(seed=20170719))         

        #--------------------------------------------------------
        #ループ前処理
        #--------------------------------------------------------
        a_output=Lambda(lambda x: K.zeros_like(x[:,-1,:]),output_shape=(1,self.n_hidden,))(encoder_outputs) 
        a_output=keras.layers.Reshape((1,self.n_hidden))(a_output)
        #---------------------------------------------------------
        #入力定義
        #---------------------------------------------------------
        decoder_inputs = Input(shape=(self.maxlen_d,), dtype='int32', name='decorder_inputs')        
        d_i = Embedding(output_dim=self.vec_dim, input_dim=self.input_dim, #input_length=self.maxlen_d,
                        mask_zero=True,
                        embeddings_initializer=uniform(seed=20170719))(decoder_inputs)
        d_i=BatchNormalization(axis=-1)(d_i)
        d_i=Masking(mask_value=0.0)(d_i)          
        d_input=d_i
        #---------------------------------------------------------
        # メイン処理（ループ）
        #---------------------------------------------------------
        for i in range(0,self.maxlen_d) :
            d_i_timeslice = a_decode_input_slice1(d_i)
            if i <= self.maxlen_d-2 :
                d_i=a_decode_input_slice2(d_i)
            d_i_timeslice=a_Reshape1(d_i_timeslice)

            lstm_input = a_Concat1([a_output,d_i_timeslice])         #前段出力とdcode_inputをconcat
            d_i_1, h1, c1 =decode_LSTM1(lstm_input,initial_state=a_states1) 
            h_output, h2, c2 =decode_LSTM2(d_i_1,initial_state=a_states2)            

            a_states1=[h1,c1]
            a_states2=[h2,c2]
            #------------------------------------------------------
            #attention
            #------------------------------------------------------
            a_o = h_output
            a_o=Dense1(a_o)
            a_o = a_Dot1([a_o,encoder_outputs])                           #encoder出力の転置行列を掛ける
            a_o= a_Softmax(a_o)                                           #softmax
            a_o= a_transpose (a_o) 
            a_o = a_Dot2([a_o,encoder_outputs])                           #encoder出力行列を掛ける
            a_o = a_Concat2([a_o,h_output])                               #ここまでの計算結果とLSTM出力をconcat
            a_o = Dense2(a_o)  
            a_o = a_tanh(a_o)                                             #tanh
            a_output=a_o                                                  #次段attention処理向け出力
            #a_output = decoder_output_BatchNormal(a_output)               # 次段attention処理向け出力のbatchNomalization
            if i == 0 :                                                  #docoder_output
                d_output=a_o
            else :
                d_output=a_Concat3([d_output,a_o]) 

        d_output=keras.layers.Reshape((self.maxlen_d,self.n_hidden))(d_output)        


        print('#5')
        #---------------------------------------------------------
        # 出力、モデル定義、コンパイル
        #---------------------------------------------------------        
        decoder_outputs_cat = decoder_Dense_cat(d_output)
        decoder_outputs_mod = decoder_Dense_mod(d_output)
        model = Model(inputs=[encoder_input, decoder_inputs], outputs=[decoder_outputs_cat ,decoder_outputs_mod])
        model.compile(loss='categorical_crossentropy',optimizer="Adam", metrics=['accuracy'])



        #=========================================================
        #デコーダー（応答文作成）
        #=========================================================
        print('#6')
        #---------------------------------------------------------
        #入力定義
        #---------------------------------------------------------        
        decoder_state_input_h_1 = Input(shape=(self.n_hidden,),name='input_h_1')
        decoder_state_input_c_1 = Input(shape=(self.n_hidden,),name='input_c_1')
        decoder_state_input_h_2 = Input(shape=(self.n_hidden,),name='input_h_2')
        decoder_state_input_c_2 = Input(shape=(self.n_hidden,),name='input_c_2')        
        decoder_states_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
        decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]  
        decoder_states_inputs=[decoder_state_input_h_1, decoder_state_input_c_1,
                               decoder_state_input_h_2, decoder_state_input_c_2]
        decoder_input_c = Input(shape=(1,self.n_hidden),name='decoder_input_c')
        decoder_input_encoded = Input(shape=(self.maxlen_e,self.n_hidden),name='decoder_input_encoded')
        #---------------------------------------------------------
        # LSTM
        #---------------------------------------------------------            
        #LSTM１段目
        decoder_i_timeslice = a_Reshape1(a_decode_input_slice1(d_input))
        l_input = a_Concat1([decoder_input_c, decoder_i_timeslice])      #前段出力とdcode_inputをconcat
        #l_input = decoder_input_BatchNorm(l_input)                       # decoderLSTM入力のbatchNomalization
        decoder_lstm_1,state_h_1, state_c_1  =decode_LSTM1(l_input,
                                                     initial_state=decoder_states_inputs_1)  #initial_stateが学習の時と違う
        #LSTM２段目
        decoder_lstm_2, state_h_2, state_c_2  =decode_LSTM2(decoder_lstm_1,
                                                      initial_state=decoder_states_inputs_2) 
        decoder_states=[state_h_1,state_c_1,state_h_2, state_c_2]
        #---------------------------------------------------------
        # Attention
        #---------------------------------------------------------            
        attention_o = Dense1(decoder_lstm_2)
        attention_o = a_Dot1([attention_o, decoder_input_encoded])                   #encoder出力の転置行列を掛ける
        attention_o = a_Softmax(attention_o)                                         #softmax
        attention_o = a_transpose (attention_o) 
        attention_o = a_Dot2([attention_o, decoder_input_encoded])                    #encoder出力行列を掛ける
        attention_o = a_Concat2([attention_o, decoder_lstm_2])                        #ここまでの計算結果とLSTM出力をconcat

        attention_o = Dense2(attention_o)  
        decoder_o = a_tanh(attention_o)                                               #tanh

        print('#7')
        #---------------------------------------------------------
        # 出力、モデル定義
        #---------------------------------------------------------                
        decoder_res_cat = decoder_Dense_cat(decoder_o)
        decoder_res_mod = decoder_Dense_mod(decoder_o)
        decoder_model = Model(
        [decoder_inputs,decoder_input_c,decoder_input_encoded] + decoder_states_inputs,
        [decoder_res_cat, decoder_res_mod, decoder_o] + decoder_states)                                           

        return model ,encoder_model ,decoder_model

    #**************************************************************
    #                                                             *
    # 評価                                                        *
    #                                                             *
    #**************************************************************
    def eval_perplexity(self,model,e_test,d_test,t_test,batch_size) :
        row=e_test.shape[0]
        s_time = time.time()
        n_batch = math.ceil(row/batch_size)
        n_loss=0
        sum_loss=0.

        for i in range(0,n_batch) :
            s = i*batch_size
            e = min([(i+1) * batch_size,row])
            e_on_batch = e_test[s:e,:]
            d_on_batch = d_test[s:e,:]
            t_on_batch = t_test[s:e,:,:]
            t_on_batch_cat = np_utils.to_categorical(t_on_batch[:,:,0],8)
            t_on_batch_mod = np_utils.to_categorical(t_on_batch[:,:,1],self.output_dim)
            mask_cat = np.zeros((e-s,self.maxlen_d,8),dtype=np.float32)
            mask_mod = np.zeros((e-s,self.maxlen_d,self.output_dim),dtype=np.float32)
            for j in range(0,e-s) :
                n_dim=self.maxlen_d-list(d_on_batch[j,:]).count(0.)
                mask_cat[j,0:n_dim,:]=1  
                mask_mod[j,0:n_dim,:]=1  
                n_loss += n_dim

            #予測
            y_pred_cat,y_pred_mod = model.predict_on_batch([e_on_batch, d_on_batch])

            #categorical_crossentropy計算
            y_pred_cat = -np.log(np.maximum(y_pred_cat,1e-7))
            y_pred_mod = -np.log(np.maximum(y_pred_mod,1e-7))
            loss_cat = t_on_batch_cat * y_pred_cat
            loss_mod = t_on_batch_mod * y_pred_mod
            sum_loss += (mask_cat * loss_cat).sum() + (mask_mod * loss_mod).sum()

            #perplexity計算
            perplexity = pow(math.e, sum_loss/n_loss)
            elapsed_time = time.time() - s_time
            sys.stdout.write(Color.GREEN + "\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s    "+"\t"+
                                "{0:.4f}".format(perplexity)+"                 " + Color.END)   
            sys.stdout.flush()

        print()

        return perplexity

    #**************************************************************
    #                                                             *
    #  train_on_batchメイン処理                                   *
    #                                                             *
    #**************************************************************
    def on_batch(self,model,j,e_input,d_input,target,batch_size) :
        e_i=e_input
        d_i=d_input
        t_l=target        
        z=list(zip(e_i,d_i,t_l))
        nr.shuffle(z)                               #シャッフル
        e_i,d_i,t_l=zip(*z)
        e_train=np.array(e_i).reshape(len(e_i),self.maxlen_e)
        d_train=np.array(d_i).reshape(len(d_i),self.maxlen_d)
        t_train=np.array(t_l).reshape(len(t_l),self.maxlen_d ,2) 

        n_split=int(e_train.shape[0]*0.05)  
        e_val=e_train[:n_split,:]
        d_val=d_train[:n_split,:]
        t_val=t_train[:n_split,:,:]

        #損失関数、評価関数の平均計算用リスト
        list_loss_cat =[]
        list_loss_mod =[]
        list_loss = []
        list_accuracy_cat =[]
        list_accuracy_mod =[]

        s_time = time.time()
        row=e_train.shape[0]
        n_batch = math.ceil(row/batch_size)
        for i in range(0,n_batch) :
            s = i*batch_size
            e = min([(i+1) * batch_size,row])
            e_on_batch = e_train[s:e,:]
            d_on_batch = d_train[s:e,:]
            #　ラベルテンソルをカテゴリビットごとにスライスする
            t_on_batch = t_train[s:e,:,:]
            t_on_batch_cat = np_utils.to_categorical(t_on_batch[:,:,0],8)
            t_on_batch_mod = np_utils.to_categorical(t_on_batch[:,:,1],self.output_dim)
            result=model.train_on_batch([e_on_batch, d_on_batch],[t_on_batch_cat, t_on_batch_mod])
            list_loss_cat.append(result[0])
            list_loss_mod.append(result[1])
            list_loss.append(result[2])
            list_accuracy_cat.append(result[3])
            list_accuracy_mod.append(result[4])
            elapsed_time = time.time() - s_time
            sys.stdout.write(Color.CYAN + "\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s    "+"\t"+
                            "{0:.4f}".format(np.average(list_loss_cat))+"\t"+
                            "{0:.4f}".format(np.average(list_loss_mod))+"\t"+
                            "{0:.4f}".format(np.average(list_accuracy_cat))+"\t"+
                            "{0:.4f}".format(np.average(list_accuracy_mod)) + Color.END) 
            sys.stdout.flush()
            del e_on_batch,d_on_batch,t_on_batch, t_on_batch_cat,  t_on_batch_mod 

        #perplexity評価
        print()
        val_perplexity=self.eval_perplexity(model,e_val,d_val,t_val,batch_size)
        loss= np.average(list_loss)

        return val_perplexity, loss

    #**************************************************************
    #                                                             *
    #  学習                                                       *
    #                                                             *
    #**************************************************************
    def train(self, e_input, d_input,target,batch_size,epochs, emb_param) :

        print ('#2',target.shape)
        model, _, _ = self.create_model()  
        if os.path.isfile(emb_param) :
            model.load_weights(emb_param)               #埋め込みパラメータセット
        print ('#8')     
        #=========================================================
        # train on batch
        #=========================================================
        e_i = e_input
        d_i = d_input
        t_l = target

        row=e_input.shape[0]
        loss_bk =10000
        patience = 0

        for j in range(0,epochs) :
            print(Color.CYAN,"Epoch ",j+1,"/",epochs,Color.END)
            val_perplexity,val_loss  = self.on_batch(model,j,e_i,d_i,t_l,batch_size)
            model.save_weights(emb_param)
            #-----------------------------------------------------
            # EarlyStopping
            #-----------------------------------------------------            
            if j == 0 or val_loss <= loss_bk:
                loss_bk = val_loss 
                patience = 0
            elif patience < 2  :
                patience += 1
            else :
                print('EarlyStopping') 
                break 

        return model