#分かち書き作成
# 学習済みモデルを作成するため。分かち書きファイルを作成
import MeCab
import os, glob

wakati_file = './data/wakati.txt'

def wakati(text):
        result = []
        #各行に分ける(windowsの改行コード\r\nを想定)
        lines = text.split('\r')
        for line in lines:
            line = line.replace('\n','')

            # 形態素解析
            tagger = MeCab.Tagger("-d /var/lib/mecab/dic/mecab-ipadic-neologd")

            for chunk in tagger.parse(line).splitlines()[:-1]:
                (surface, hinshi) = chunk.split('\t')
                if hinshi.startswith('名詞') or hinshi.startswith('動詞') or hinshi.startswith('形容詞'):
    #           文章の一番最初に出てくる名詞or動詞or形容詞について処理を実行する
                    if '*' in hinshi.split(",")[6]:
    #                     基本形が存在しない場合、表層形を返す
                        result.append(surface)
                    else :
    #                     基本形を返す
                        result.append(hinshi.split(",")[6])
                elif hinshi.startswith('感動詞'):
                    result.append(surface)
        return result

def main(file):
    words = []
    file_dir = os.path.abspath(file)
    try:
        bindata = open(file_dir, 'rb').read()
        text = bindata.decode('shift_jis')
        words = wakati(text)
    except:
        try:
            text = bindata.decode('utf-8')
        except:
            try:
                text = bindata.decode('cp932')
            except Exception as e:
                print('error!',e)
                exit(0)

    with open(wakati_file, 'a', encoding='utf-8') as f:
        f.write(' '.join(words))

if __name__ == '__main__':
    if os.path.exists(wakati_file):
#         すでにわかち書きが存在する場合削除
        os.remove(wakati_file)
        print('remove -> ' + wakati_file)
#   learningフォルダ直下のテキストファイルを読み込む
    files = glob.glob('./data/learning/*.txt')
    for file in files:
        if not 'wakati' in file:
            print(file)
            main(file)