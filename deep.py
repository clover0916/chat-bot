from gensim.models import word2vec

wakati_file = './data/wakati.txt'
model_file = './data/daily_w2v_model.model'

if __name__ == '__main__':
    file_dir = wakati_file
    w2v_data = word2vec.LineSentence(file_dir)
    model = word2vec.Word2Vec(w2v_data, size=100, window=3, hs=1, min_count=1, sg=1)
    model.save(model_file)