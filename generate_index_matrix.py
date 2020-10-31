# coding: utf-8

def generate_mat() :

    file_list = glob.glob('list_corpus/*')
    print('ファイル数 =',len(file_list))
    mat=[]
    for i in range (0,len(file_list)) :
        with open(file_list[i],'rb') as f :
            generated_list=pickle.load(f)         #生成リストロード
            mat.extend(generated_list)
            print(i)
            del generated_list


    mat.append('REQREQ')

    print(len(mat))
    words = sorted(list(set(mat)))
    cnt = np.zeros(len(words))

    print('total words:', len(words))
    word_indices = dict((w, i) for i, w in enumerate(words))    #単語をキーにインデックス検索
    indices_word = dict((i, w) for i, w in enumerate(words))    #インデックスをキーに単語を検索

    #単語の出現数をカウント
    for j in range (0,len(mat)):
        cnt[word_indices[mat[j]]] += 1
        
    #単語インデックス配列作成
    mat_urtext = np.zeros((len(mat),1),dtype=int)
    for i in range(0,len(mat)):
        if mat[i] in word_indices :           #出現頻度の低い単語のインデックスをunkのそれに置き換え
            mat_urtext[i,0] = word_indices[mat[i]]
        else:
            mat_urtext[i,0] = word_indices['UNK']

    print(mat_urtext.shape)

    #作成した辞書をセーブ
    with open('word_indices.pickle', 'wb') as f :
        pickle.dump(word_indices , f)

    with open('indices_word.pickle', 'wb') as g :
        pickle.dump(indices_word , g)

    #単語ファイルセーブ
    with open('words.pickle', 'wb') as h :
        pickle.dump(words , h)

    #コーパスセーブ
    with open('mat_urtext.pickle', 'wb') as ff :
        pickle.dump(mat_urtext , ff)    

if __name__ == '__main__':
    import numpy as np
    import pickle
    import glob

    generate_mat()