# coding: utf-8

#*******************************************************************************
#                                                                              *
# 応答文生成                                                                   *
#                                                                              *
#*******************************************************************************    
def generate_reply(param, tweeted_sentence) :

    #--------------------------------------------------------------*
    # 入力文の品詞分解とインデックス化                             *
    #--------------------------------------------------------------*
    e_input = gen_res.encode_request(tweeted_sentence, maxlen_e, word_indices, words, encoder_model)

    #--------------------------------------------------------------*
    # 応答文組み立て                                               *
    #--------------------------------------------------------------*       
    decoded_sentence = gen_res.generate_response(e_input, n_hidden, maxlen_d, output_dim, word_indices, 
                                         freq_indices, indices_word, encoder_model, decoder_model)

    return decoded_sentence

#*******************************************************************************
#                                                                              *
# メイン処理                                                                   *
#                                                                              *
#*******************************************************************************    
if __name__ == '__main__':

    import discord
    import json
    import time, sys
    import re
    import emoji
    import sys
    import pickle
    import os
    import math
    
    client = discord.Client()
    json_open = open('./config.json', 'r')
    config = json.load(json_open)

    #--------------------------------------------------------------------------*
    #                                                                          *
    # ニューラルネットワーク初期化                                             *
    #                                                                          *
    #--------------------------------------------------------------------------*
    import response as gen_res

    args = sys.argv
    #args[1] = 'param_001'                                       # jupyter上で実行するとき用

    vec_dim = 400
    n_hidden = int(vec_dim*1.5 )                 #隠れ層の次元

    #データロード
    word_indices ,indices_word ,words ,maxlen_e, maxlen_d ,freq_indices = gen_res.load_data()
    #入出力次元
    input_dim = len(words)
    output_dim = math.ceil(len(words) / 8)
    #モデル初期化
    model, encoder_model ,decoder_model = gen_res.initialize_models(args[1] ,maxlen_e, maxlen_d,
                                                            vec_dim, input_dim, output_dim, n_hidden)
                                                            
    @client.event
    async def on_ready():
      print('We have logged in as {0.user}'.format(client))

    @client.event
    async def on_message(message):
      if not message.content.startswith(config['prefix']) and message.author == client.user:
        return
      if message.channel.name == 'clover-chat':
        res_text = generate_reply(args[1], message.content)
        await message.channel.send(res_text)
        
    client.run(config['token'], reconnect=True)