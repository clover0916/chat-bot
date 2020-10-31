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
    if len(word) > 7 and word[:7] == 'SSSSUNK' :
        modified = ['SSSS', word[7:]]
    elif len(word) > 4 and word[:4] == 'SSSS' :
        modified = ['SSSS', word[4:]]
    elif word == 'UNKUNK' :
        modified = ['UNK']
    elif len(word) > 3 and word[:3] == 'UNK' :
        modified = ['UNK', word[3:]]
    else :
        modified = [word]
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
        if len(data[i][0].encode('utf-8')) <= 4096 :
            result = jumanpp.analysis(data[i][0])
        else :
            print(i, ' skip')
            continue
        for mrph in result.mrph_list():
            parts += modification(mrph.midasi)
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