from word2vecReader import Word2Vec
import time
import re
import numpy as np

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):

        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def load_w2v():
    model_path = '../data/word2vec_twitter_model/word2vec_twitter_model.bin'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model


with Timer("w2v"):
    # w2vmodel = {}
    w2vmodel = load_w2v()



def load_lexicon_unigram():
    filename = 'unigrams-pmilexicon.txt'
    default_vector = [0]


    file_path = "../data/lexicon_data/"+filename

    raw_model = dict()
    norm_model = dict()



    # print data_type, default_vector
    raw_model["<PAD/>"] = default_vector

    with open(file_path, 'r') as document:
        for line in document:
            line_token = re.split(r'\t', line)

            data_vec=[]
            key=''

            for idx, tk in enumerate(line_token):
                if idx == 0:
                    key = tk

                elif idx == 1:
                    data_vec.append(float(tk))

                else:
                    continue


            assert(key != '')
            raw_model[key] = data_vec

    values = np.array(raw_model.values())
    new_val = np.copy(values)

    for i in range(len(raw_model.values()[0])):
        pos = np.max(values, axis=0)[i]
        neg = np.min(values, axis=0)[i]
        mmax = max(abs(pos), abs(neg))
        print pos, neg, mmax

        new_val[:, i] = values[:, i] / mmax

    keys = raw_model[1].keys()
    dictionary = dict(zip(keys, new_val))
    norm_model[index] = dictionary

    data_type = file_path[index].replace("../data/lexicon_data/", "")
    default_vector = default_vector_dic[data_type]

    dictionary["<PAD/>"] = default_vector
    # models.append(dictionary)

    return norm_model, raw_model


