# coding: utf-8

#*************************************************************************************
#                                                                                    *
#   import宣言                                                                       *
#                                                                                    *
#*************************************************************************************

from __future__ import print_function
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

import discord
import json
client = discord.Client()

from keras.utils import plot_model
from pyknp import Juman
Jumanpp =Juman()
import codecs


#*************************************************************************************
#                                                                                    *
#   辞書ファイル等ロード                                                             *
#                                                                                    *
#*************************************************************************************

def load_data() :

    #辞書をロード
    with open('word_indices.pickle', 'rb') as f :
        word_indices=pickle.load(f)         #単語をキーにインデックス検索

    with open('indices_word.pickle', 'rb') as g :
        indices_word=pickle.load(g)         #インデックスをキーに単語を検索

    #単語ファイルロード
    with open('words.pickle', 'rb') as ff :
        words=pickle.load(ff)         

    #maxlenロード
    with open('maxlen.pickle', 'rb') as maxlen :
        [maxlen_e, maxlen_d] = pickle.load(maxlen)

    #各単語の出現頻度順位（降順）
    with open('freq_indices.pickle', 'rb') as f :    
        freq_indices = pickle.load(f)

    #出現頻度→インデックス変換
    with open('indices_freq.pickle', 'rb') as f :    
        indices_freq = pickle.load(f)

    return word_indices ,indices_word ,words ,maxlen_e, maxlen_d,  freq_indices


#*************************************************************************************
#                                                                                    *
#   モデル初期化                                                                     *
#                                                                                    *
#*************************************************************************************

def initialize_models(emb_param ,maxlen_e, maxlen_d ,vec_dim, input_dim,output_dim, n_hidden) :

    dialog= Dialog(maxlen_e, 1, n_hidden, input_dim, vec_dim, output_dim)
    model ,encoder_model ,decoder_model = dialog.create_model()

    param_file = emb_param + '.hdf5'
    model.load_weights(param_file)  

    plot_model(encoder_model, show_shapes=True,to_file='seq2seq0212_encoder.png')
    plot_model(decoder_model, show_shapes=True,to_file='seq2seq0212_decoder.png')

    return model, encoder_model ,decoder_model


#*************************************************************************************
#                                                                                    *
#   入力文の品詞分解とインデックス化                                                 *
#                                                                                    *
#*************************************************************************************

def encode_request(cns_input, maxlen_e, word_indices, words, encoder_model) :
    # Use Juman++ in subprocess mode
    jumanpp = Juman()
    result = jumanpp.analysis(cns_input)
    input_text=[]
    for mrph in result.mrph_list():
        input_text.append(mrph.midasi)

    mat_input=np.array(input_text)

    #入力データe_inputに入力文の単語インデックスを設定
    e_input=np.zeros((1,maxlen_e))
    for i in range(0,len(mat_input)) :
        if mat_input[i] in words :
            e_input[0,i] = word_indices[mat_input[i]]
        else :
            e_input[0,i] = word_indices['UNK']

    return e_input


#*************************************************************************************
#                                                                                    *
#   応答文組み立て                                                                   *
#                                                                                    *
#*************************************************************************************

def generate_response(e_input, n_hidden, maxlen_d, output_dim, word_indices,
                      freq_indices, indices_word, encoder_model, decoder_model) :
    # Encode the input as state vectors.
    encoder_outputs, state_h_1, state_c_1, state_h_2, state_c_2 = encoder_model.predict(e_input)
    states_value= [state_h_1, state_c_1, state_h_2, state_c_2]        
    decoder_input_c = np.zeros((1,1,n_hidden) ,dtype='int32')

    decoded_sentence = ''
    target_seq = np.zeros((1,1) ,dtype='int32')
    # Populate the first character of target sequence with the start character.
    target_seq[0,  0] = word_indices['RESRES']
    # 応答文字予測

    for i in range(0,maxlen_d) :
        output_tokens_cat, output_tokens_mod, d_output, h1, c1, h2, c2 = decoder_model.predict(
                    [target_seq,decoder_input_c,encoder_outputs]+ states_value) 

        # 予測単語の出現頻度算出
        n_cat = np.argmax(output_tokens_cat[0, 0, :])
        n_mod = np.argmax(output_tokens_mod[0, 0, :])
        freq = (n_cat * output_dim + n_mod).astype(int)
        #予測単語のインデックス値を求める
        sampled_token_index = freq_indices[freq]
        #予測単語
        sampled_char = indices_word[sampled_token_index]
        # Exit condition: find stop character.
        if sampled_char == 'REQREQ' :
            break
        decoded_sentence += sampled_char  

        # Update the target sequence (of length 1).
        if i == maxlen_d-1:
            break
        target_seq[0,0] = sampled_token_index 

        decoder_input_c = d_output
        # Update states
        states_value = [h1, c1, h2, c2]  

    return decoded_sentence


#*************************************************************************************
#                                                                                    *
#   メイン処理                                                                       *
#                                                                                    *
#*************************************************************************************

if __name__ == '__main__':


    vec_dim = 400
    n_hidden = int(vec_dim*1.5 )                 #隠れ層の次元

    args = sys.argv

    #args[1] = 'param_003'                                              # jupyter上で実行するとき用    

    #データロード
    word_indices ,indices_word ,words ,maxlen_e, maxlen_d ,freq_indices = load_data()
    #入出力次元
    input_dim = len(words)
    output_dim = math.ceil(len(words) / 8)
    #モデル初期化
    model, encoder_model ,decoder_model = initialize_models(args[1] ,maxlen_e, maxlen_d,
                                                            vec_dim, input_dim, output_dim, n_hidden)

    sys.stdin = codecs.getreader('utf_8')(sys.stdin)


    json_open = open('./config.json', 'r')
    config = json.load(json_open)

    @client.event
    async def on_ready():
      print('We have logged in as {0.user}'.format(client))

    @client.event
    async def on_message(message):
      if not message.content.startswith(config['prefix']) and message.author == client.user:
        return
      if message.channel.name == 'clover-chat':
        e_input = encode_request(message.content, maxlen_e, word_indices, words, encoder_model)
        decoded_sentence = generate_response(e_input, n_hidden, maxlen_d, output_dim, word_indices, freq_indices, indices_word, encoder_model, decoder_model)
        await message.channel.send(decoded_sentence)
        
    client.run(config['token'], reconnect=True)
