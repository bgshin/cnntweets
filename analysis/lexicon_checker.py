#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
from cnntweets.utils import cnn_data_helpers
from cnntweets.cnn_models.w2v_lex_cnn import W2V_LEX_CNN
from cnntweets.utils.word2vecReader import Word2Vec
import gc
import pickle
from cnntweets.utils.butils import Timer


# Data Preparatopn
# ==================================================

def load_lexicon_unigram():
    default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                      'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                      'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                      'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                      'unigrams-pmilexicon.txt':[0,0,0],
                      'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                      'BL.txt': [0]}

    file_list = ['EverythingUnigramsPMIHS.txt',
                  'HS-AFFLEX-NEGLEX-unigrams.txt',
                  'Maxdiff-Twitter-Lexicon_0to1.txt',
                  'S140-AFFLEX-NEGLEX-unigrams.txt',
                  'unigrams-pmilexicon.txt',
                  'unigrams-pmilexicon_sentiment_140.txt',
                  'BL.txt']


    file_path = ["../data/lexicon_data/"+files for files in file_list ]

    raw_model = [dict() for x in range(len(file_path))]
    norm_model = [dict() for x in range(len(file_path))]

    data_type_list = []
    for index, each_model in enumerate(raw_model):
        data_type = file_path[index].replace("../data/lexicon_data/", "")
        data_type_list.append(data_type)
        default_vector = default_vector_dic[data_type]
        print data_type

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector

        with open(file_path[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec=[]
                key=''


                for idx, tk in enumerate(line_token):
                    if idx == 0:
                        key = tk
                    else:
                        try:
                            data_vec.append(float(tk))
                        except:
                            pass


                assert(key != '')
                each_model[key] = data_vec

    for index, each_model in enumerate(norm_model):
        data_type = data_type_list[index]
        print data_type

    # for m in range(len(raw_model)):
        values = np.array(raw_model[index].values())
        new_val = np.copy(values)


        #print 'model %d' % index
        for i in range(len(raw_model[index].values()[0])):
            pos = np.max(values, axis=0)[i]
            neg = np.min(values, axis=0)[i]
            mmax = max(abs(pos), abs(neg))
            #print pos, neg, mmax

            new_val[:, i] = values[:, i] / mmax

        keys = raw_model[index].keys()
        dictionary = dict(zip(keys, new_val))

        norm_model[index] = dictionary

    # for index in range(7):
    #     data_type = data_type_list[index]
    #     print 'save', data_type
    #     with open(data_type.replace('txt', '') + 'pickle', 'wb') as handle:
    #         pickle.dump(norm_model[index], handle)

    return norm_model



def load_lex():
    default_vector_dic = {'EverythingUnigramsPMIHS': [0],
                          'HS-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                          'Maxdiff-Twitter-Lexicon_0to1': [0.5],
                          'S140-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                          'unigrams-pmilexicon': [0, 0, 0],
                          'unigrams-pmilexicon_sentiment_140': [0, 0, 0],
                          'BL': [0]}
    lexfile_list = ['EverythingUnigramsPMIHS.pickle',
                    'HS-AFFLEX-NEGLEX-unigrams.pickle',
                    'Maxdiff-Twitter-Lexicon_0to1.pickle',
                    'S140-AFFLEX-NEGLEX-unigrams.pickle',
                    'unigrams-pmilexicon.pickle',
                    'unigrams-pmilexicon_sentiment_140.pickle',
                    'BL.pickle']

    norm_model = []

    for idx, lexfile in enumerate(lexfile_list):
        fname = './new_lex/%s' % lexfile

        with open(fname, 'rb') as handle:
            each_model = pickle.load(handle)
            # default_vector = default_vector_dic[lexfile.replace('.pickle', '')]
            # each_model["<PAD/>"] = default_vector
            norm_model.append(each_model)


    return norm_model

def checker():


    with Timer("loading lex.pickle..."):
        unigram_lexicon_model2 = load_lexicon_unigram()

    with Timer("loading lex..."):
        unigram_lexicon_model1 = load_lex()


    print len(unigram_lexicon_model1), len(unigram_lexicon_model2)


    for idx, model in enumerate(unigram_lexicon_model1):
        print idx
        for ii, key in enumerate(model.keys()):
            a = np.array(unigram_lexicon_model1[idx][key])
            b = np.array(unigram_lexicon_model2[idx][key])
            if np.abs(np.sum(a-b))>0:
                print 'wrong', key,np.sum(a-b), a, b

        print np.sum(np.array(unigram_lexicon_model1[idx]["<PAD/>"])-np.array(unigram_lexicon_model2[idx]["<PAD/>"]))
        print unigram_lexicon_model1[idx]["<PAD/>"], unigram_lexicon_model2[idx]["<PAD/>"],
        # print ii
        # print unigram_lexicon_model1[i]['good'], unigram_lexicon_model2[i]['good']







if __name__ == "__main__":
    checker()

