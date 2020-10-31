import numpy as np
import csv
import glob
import pickle

#*******************************************************************************
#                                                                              *
#   単語補正                                                                   *
#                                                                              *
#*******************************************************************************

#*******************************************************************************
#                                                                              *
#   品詞分解                                                                   *
#                                                                              *
#*******************************************************************************
def decomposition(file, jumanpp) :
    f=open(file, 'r')
    df1 = csv.reader(f)
    print('number of rows :', len(df))

    parts = []
    for i in range(len(df1)) :
        if len(df1.encode('utf-8')) <= 4096 :
            word = df1.replace(' ', '　')
            result = jumanpp.analysis(df1)
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