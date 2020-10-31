import numpy as np
import csv
import glob
import pickle

#*******************************************************************************
#                                                                              *
#   単語補正                                                                   *
#                                                                              *
#*******************************************************************************
def modification(word) :
    print(word)
    if word[:3] == 'REQ:' :
        modified = ['REQREQ', word[4:]]
    elif word[:3] == 'RES:' :
        modified = ['REQREQ', word[4:]]
    elif word[0] == '@' :
        modified = ['UNK']
    elif word == 'EOS' :
        modified = ['UNK']
    else :
        modified = [word]
    print(modified)
    return modified

#*******************************************************************************
#                                                                              *
#   品詞分解                                                                   *
#                                                                              *
#*******************************************************************************
def decomposition(file, jumanpp) :
    f=open(file, 'r')
    df1 = csv.reader(f)
    data = [ v for v in df1]
    print('number of rows :', len(data))

    parts = []
    for i in range(len(data)) :
        word = data[i][0].replace(" ", "")
        if len(word.encode('utf-8')) <= 4096 :
            datas = modification(data[i][0])
            result = jumanpp.analysis(datas)
        else :
            print(i, ' skip')
            continue
        for mrph in result.mrph_list():
            parts += mrph.midasi
        if i % 5000 == 0 :
            print(i)
    return parts
#*******************************************************************************
#                                                                              *
#   メイン処理                                                                 *
#                                                                              *
#*******************************************************************************
from pyknp import Juman
jumanpp = Juman()
file_list=glob.glob('corpus/*')
file_list.sort()
print(len(file_list))

parts_list = []
for j in range(len(file_list)) :
    print(file_list[j])
    parts_list += decomposition(file_list[j], jumanpp)

with open('parts_list.pickle', 'wb') as f :    
    pickle.dump(parts_list , f)