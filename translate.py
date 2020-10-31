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

from pyknp import Juman
jumanpp = Juman()
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

class Dialog :
    def __init__(self,maxlen_e,maxlen_d,n_hidden,input_dim,vec_dim,output_dim):
        self.maxlen_e=maxlen_e
        self.maxlen_d=maxlen_d
        self.n_hidden=n_hidden
        self.input_dim=input_dim
        self.vec_dim=vec_dim
        self.output_dim=output_dim

    def create_model(self):
        print('#3')
        #エンコーダー
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
                                                #dropout=0.5, recurrent_dropout=0.5
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
                                                #dropout=0.5, recurrent_dropout=0.5
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
        encoder_states1 = [state_h_1,state_c_1] 
        encoder_states2 = [state_h_2,state_c_2]

        encoder_model = Model(inputs=encoder_input, 
                              outputs=[encoder_outputs,state_h_1,state_c_1,state_h_2,state_c_2])    #エンコーダモデル        


        print('#4')        
        #デコーダー（学習用）
        # デコーダを、完全な出力シークエンスを返し、内部状態もまた返すように設定します。
        # 訓練モデルではreturn_sequencesを使用しませんが、推論では使用します。
        a_states1=encoder_states1
        a_states2=encoder_states2

        #レイヤー定義
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
        a_Reshape1=keras.layers.Reshape((1,self.vec_dim))
        a_Dot1=keras.layers.Dot(-1,name='a_Dot1')
        a_Softmax=keras.layers.Softmax(axis=-1,name='a_Softmax')
        a_transpose = keras.layers.Reshape((self.maxlen_e,1),name='Transpose') 
        a_Dot2=keras.layers.Dot(1,name='a_Dot2')
        a_Concat2=keras.layers.Concatenate(-1,name='a_Concat2')
        a_tanh=Lambda(lambda x: K.tanh(x),name='tanh')
        a_Concat3=keras.layers.Concatenate(axis=-1,name='a_Concat3')
        decoder_Dense = Dense(self.output_dim,activation='softmax', name='decoder_Dense',
                              kernel_initializer=glorot_uniform(seed=20170719))        

        a_output=Lambda(lambda x: K.zeros_like(x[:,-1,:]),output_shape=(1,self.n_hidden,))(encoder_outputs) 
        a_output=keras.layers.Reshape((1,self.n_hidden))(a_output)

        decoder_inputs = Input(shape=(self.maxlen_d,), dtype='int32', name='decorder_inputs')        
        d_i = Embedding(output_dim=self.vec_dim, input_dim=self.input_dim, #input_length=self.maxlen_d,
                        mask_zero=True,
                        embeddings_initializer=uniform(seed=20170719))(decoder_inputs)
        d_i=BatchNormalization(axis=-1)(d_i)
        d_i=Masking(mask_value=0.0)(d_i)          
        d_input=d_i

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

            #attention
            a_o = h_output
            a_o=Dense1(a_o)
            a_o = a_Dot1([a_o,encoder_outputs])                           #encoder出力の転置行列を掛ける
            a_o= a_Softmax(a_o)                                           #softmax
            a_o= a_transpose (a_o) 
            a_o = a_Dot2([a_o,encoder_outputs])                           #encoder出力行列を掛ける
            a_o = a_Concat2([a_o,h_output])                               #ここまでの計算結果とLSTM出力をconcat
            a_o=Dense2(a_o)  
            a_o=a_tanh(a_o)                                               #tanh
            a_output=a_o                                                  #次段attention処理向け出力
            if i == 0 :                                                  #docoder_output
                d_output=a_o
            else :
                d_output=a_Concat3([d_output,a_o]) 

        d_output=keras.layers.Reshape((self.maxlen_d,self.n_hidden))(d_output)        

        print('#5')
        decoder_outputs = decoder_Dense(d_output)
        model = Model(inputs=[encoder_input, decoder_inputs], outputs=decoder_outputs)
        model.compile(loss="categorical_crossentropy",optimizer="Adam", metrics=['categorical_accuracy'])

        #デコーダー（応答文作成）
        print('#6')
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
        #LSTM１段目
        decoder_i_timeslice = a_Reshape1(a_decode_input_slice1(d_input))
        l_input = a_Concat1([decoder_input_c, decoder_i_timeslice])      #前段出力とdcode_inputをconcat
        decoder_lstm_1,state_h_1, state_c_1  =decode_LSTM1(l_input,
                                                     initial_state=decoder_states_inputs_1)  #initial_stateが学習の時と違う
        #LSTM２段目
        decoder_lstm_2, state_h_2, state_c_2  =decode_LSTM2(decoder_lstm_1,
                                                      initial_state=decoder_states_inputs_2) 
        decoder_states=[state_h_1,state_c_1,state_h_2, state_c_2]

        #attention
        attention_o = Dense1(decoder_lstm_2)
        attention_o = a_Dot1([attention_o, decoder_input_encoded])                   #encoder出力の転置行列を掛ける
        attention_o = a_Softmax(attention_o)                                         #softmax
        attention_o = a_transpose (attention_o) 
        attention_o = a_Dot2([attention_o, decoder_input_encoded])                    #encoder出力行列を掛ける
        attention_o = a_Concat2([attention_o, decoder_lstm_2])                        #ここまでの計算結果とLSTM出力をconcat

        attention_o = Dense2(attention_o)  
        decoder_o = a_tanh(attention_o)                                               #tanh

        print('#7')
        decoder_res = decoder_Dense(decoder_o)
        decoder_model = Model(
        [decoder_inputs,decoder_input_c,decoder_input_encoded] + decoder_states_inputs,
        [decoder_res, decoder_o] + decoder_states)                                           

        return model ,encoder_model ,decoder_model

    #評価
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
            t_on_batch = t_test[s:e,:]
            t_on_batch = np_utils.to_categorical(t_on_batch,self.output_dim)
            #mask行列作成
            mask1 = np.zeros((e-s,self.maxlen_d,self.output_dim),dtype=np.float32)
            for j in range(0,e-s) :
                n_dim=maxlen_d-list(d_on_batch[j,:]).count(0.)
                mask1[j,0:n_dim,:]=1  
                n_loss += n_dim

            mask2=mask1.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            #予測
            y_predict1=model.predict_on_batch([e_on_batch, d_on_batch])
            #category_crossentropy計算
            y_predict2=np.maximum(y_predict1,0.00001)
            y_predict2 = -np.log(y_predict2)
            y_predict3=y_predict2.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            target=t_on_batch.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            target1=target*mask2                         #マスキング
            loss=np.dot(y_predict3,target1.T)
            sum_loss += loss[0,0]
            #perplexity計算
            perplexity=pow(math.e, sum_loss/n_loss)
            elapsed_time = time.time() - s_time
            sys.stdout.write("\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s "+"\t"+
                                "{0:.4f}".format(perplexity)+"                 ")   
            sys.stdout.flush()
            del e_on_batch,d_on_batch,t_on_batch
            del mask1,mask2
            del y_predict1,y_predict2,y_predict3
            del target,target1
            gc.collect()

        print()
        return perplexity

    #train_on_batchメイン処理
    def on_batch(self,model,j,e_train,d_train,t_train,e_val,d_val,t_val,batch_size) :
        #損失関数、評価関数の平均計算用リスト
        list_loss =[]
        list_accuracy=[]

        s_time = time.time()
        row=e_train.shape[0]
        n_batch = math.ceil(row/batch_size)
        for i in range(0,n_batch) :
            s = i*batch_size
            e = min([(i+1) * batch_size,row])
            e_on_batch = e_train[s:e,:]
            d_on_batch = d_train[s:e,:]
            t_on_batch = t_train[s:e,:]
            t_on_batch = np_utils.to_categorical(t_on_batch,self.output_dim)
            result=model.train_on_batch([e_on_batch, d_on_batch],t_on_batch)
            list_loss.append(result[0])
            list_accuracy.append(result[1])
            elapsed_time = time.time() - s_time
            sys.stdout.write("\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s "+"\t"+
                            "{0:.4f}".format(np.average(list_loss))+"\t"+
                            "{0:.4f}".format(np.average(list_accuracy))+"                 ")   
            sys.stdout.flush()
            del e_on_batch,d_on_batch,t_on_batch

        #perplexity評価
        print()
        val_perplexity=self.eval_perplexity(model,e_val,d_val,t_val,batch_size)
        loss= np.average(list_loss)
        del list_loss,  list_accuracy        

        return val_perplexity

    # 学習
    def train(self, e_input, d_input,target,batch_size,epochs, emb_param) :

        print ('#2',target.shape)
        model, _, _ = self.create_model()  
        if os.path.isfile(emb_param) :
            model.load_weights(emb_param)               #埋め込みパラメータセット
        print ('#8')        
        # train on batch

        e_i = e_input
        d_i = d_input
        t_l = target

        n_split = int(e_i.shape[0]*0.9)                 #訓練データとテストデータを9:1に分割
        e_train,e_val = np.vsplit(e_i,[n_split])   #エンコーダインプットデータを訓練用と評価用に分割
        d_train,d_val = np.vsplit(d_i,[n_split])   #デコーダインプットデータを訓練用と評価用に分割
        t_train,t_val = np.vsplit(t_l,[n_split])   #ラベルデータを訓練用と評価用に分割 

        row = e_input.shape[0]
        loss_bk = 10000
        for j in range(0,epochs) :
            print("Epoch ",j+1,"/",epochs)
            val_perplexity = self.on_batch(model,j,e_train,d_train,t_train,e_val,d_val,t_val,batch_size)
            model.save_weights(emb_param)  
            #EarlyStopping
            if j == 0 or val_perplexity <= loss_bk:
                loss_bk = val_perplexity 
            else  :
                print('EarlyStopping') 
                break 

        return model            

    def response(self,e_input,length) :
        # Encode the input as state vectors.
        encoder_outputs,state_h_1,state_c_1,state_h_2,state_c_2 = encoder_model.predict(e_input)
        states_value=[state_h_1,state_c_1,state_h_2,state_c_2]

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))
        # Populate the first character of target sequence with the start character.
        target_seq[0,  0] = word_indices['RESRES']
        decoder_input_c = encoder_outputs[:,-1,:].reshape((1,1,self.n_hidden))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        #stop_condition = False
        decoded_sentence = ''
        for i in range(0,length) :
            output_tokens, d_output, h1, c1,h2,c2 = decoder_model.predict(
                [target_seq,decoder_input_c,encoder_outputs]+ states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, 0, :])
            sampled_char = indices_word[sampled_token_index]

            # Exit condition: find stop character.
            if sampled_char == 'REQREQ' :
                break
            decoded_sentence += sampled_char    
            # Update the target sequence (of length 1).
            if i==length-1:
                break
            target_seq[0,0] = sampled_token_index 
            decoder_input_c = d_output
            # Update states
            states_value = [h1, c1, h2, c2]

        return decoded_sentence            

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