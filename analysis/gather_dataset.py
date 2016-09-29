#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
from cnntweets.utils import cnn_data_helpers
from cnntweets.cnn_models.w2v_lex_cnn import W2V_LEX_CNN
from cnntweets.cnn_models.w2v_cnn import W2V_CNN
from cnntweets.cnn_models.preattention_cnn import TextCNNAttention2VecIndividual
from cnntweets.utils.word2vecReader import Word2Vec
import gc
import pickle
from cnntweets.utils.butils import Timer

# Parameters
# ==================================================


tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# os.system('cls' if os.name == 'nt' else 'clear')


# Data Preparatopn
# ==================================================

def load_lexicon_unigram(lexdim):
    if lexdim==6:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'HS-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
                              'S140-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'unigrams-pmilexicon.txt': [0],
                              'unigrams-pmilexicon_sentiment_140.txt': [0]}

    elif lexdim == 2:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'unigrams-pmilexicon.txt': [0]}

    elif lexdim == 4:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'unigrams-pmilexicon.txt': [0, 0, 0]}

    elif lexdim == 15:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                          'BL.txt': [0]}
    else:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                          'BL.txt': [0]}

    file_path = ["../data/lexicon_data/"+files for files in os.listdir("../data/lexicon_data") if files.endswith(".txt")]
    if lexdim == 2 or lexdim == 4:
        raw_model = [dict() for x in range(2)]
        norm_model = [dict() for x in range(2)]
        file_path = ['../data/lexicon_data/EverythingUnigramsPMIHS.txt', '../data/lexicon_data/unigrams-pmilexicon.txt']
    else:
        raw_model = [dict() for x in range(len(file_path))]
        norm_model = [dict() for x in range(len(file_path))]

    data_type_list = []
    for index, each_model in enumerate(raw_model):
        data_type = file_path[index].replace("../data/lexicon_data/", "")
        data_type_list.append(data_type)
        # if lexdim == 2 or lexdim == 4:
        #     if data_type not in ['EverythingUnigramsPMIHS.txt', 'unigrams-pmilexicon.txt']:
        #         continue

        default_vector = default_vector_dic[data_type]

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector

        with open(file_path[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec=[]
                key=''

                if lexdim == 2 or lexdim == 6:
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk

                        elif idx == 1:
                            data_vec.append(float(tk))

                        else:
                            continue

                else: # 4 or 14
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


    for index in range(7):
        with open(data_type.replace('txt','')+'pickle', 'wb') as handle:
            pickle.dump(norm_model[index], handle)

    with open('all.pickle', 'wb') as handle:
        pickle.dump(norm_model, handle)

    return norm_model


def load_w2v(w2vdim):
    fname = '../data/emory_w2v/w2v-%d.bin' % w2vdim
    w2vmodel = Word2Vec.load_word2vec_format(fname, binary=True)  # C binary format
    return w2vmodel

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
        fname = '../data/le/new/%s' % lexfile

        with open(fname, 'rb') as handle:
            each_model = pickle.load(handle)
            # default_vector = default_vector_dic[lexfile.replace('.pickle', '')]
            # each_model["<PAD/>"] = default_vector
            norm_model.append(each_model)


    return norm_model

def load_dataset(w2vdim, max_len):
    with Timer("loading w2v..."):
        w2vmodel = load_w2v(w2vdim)

    with Timer("loading lex..."):
        # unigram_lexicon_model = load_lex()
        unigram_lexicon_model = load_lexicon_unigram(15)

    with Timer("loading test dataset..."):
        x_test, y_test, x_lex_test, _ = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len,
                                                                   rottenTomato=False, multichannel=False)

    del(w2vmodel)
    gc.collect()

    return x_test, y_test, x_lex_test


def load_model_cnna2vind(x_test, y_test, x_lex_test, w2vdim, lexdim, lexnumfilters, w2vnumfilters, wrong_index,
                             wrong_pred):

    savepath = './models/model-3400-cnna2vind'

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNAttention2VecIndividual(
                sequence_length=60,
                num_classes=3,
                embedding_size=w2vdim,
                embedding_size_lex=lexdim,
                num_filters_lex=lexnumfilters,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=w2vnumfilters,
                attention_depth_w2v=50,
                attention_depth_lex=20,
                l2_reg_lambda=2.0,
                l1_reg_lambda=0)

            with Timer("restore model..."):
                saver = tf.train.Saver(tf.all_variables())
                saver.restore(sess,savepath)

                var = [v for v in tf.trainable_variables()]

                vs = []
                for v in var:
                    vs.append(sess.run(v))


            print 'hello'

            x_test_wrong = x_test[wrong_index]
            y_test_wrong = y_test[wrong_index]
            x_lex_test_wrong = x_lex_test[wrong_index]


            def dev_step(x_batch, y_batch, x_batch_lex=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    # lexicon
                    cnn.input_x_lexicon: x_batch_lex,
                    cnn.dropout_keep_prob: 1.0
                }

                accuracy, h_pool_flat, appended_pool, predictions, _b, scores, w2v_pool_sq, lex_pool_sq = sess.run(
                    [ cnn.accuracy, cnn.h_pool_flat, cnn.appended_pool, cnn.predictions, cnn._b, cnn.scores,
                      cnn.w2v_pool_sq, cnn.lex_pool_sq],
                    feed_dict)


                if accuracy==0:
                    return h_pool_flat[0], appended_pool[0], predictions[0], w2v_pool_sq[0], lex_pool_sq[0], _b, scores, False

                return h_pool_flat[0], appended_pool[0], predictions[0], w2v_pool_sq[0], lex_pool_sq[0], _b, scores, True

            def test_step(x_batch, y_batch, x_batch_lex=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    # lexicon
                    cnn.input_x_lexicon: x_batch_lex,
                    cnn.dropout_keep_prob: 1.0
                }

                loss, accuracy = sess.run(
                    [cnn.loss, cnn.accuracy],
                    feed_dict)

                return accuracy

            pool_list = []
            appended_pool_list = []
            pred_list = []
            correct_list = []
            score_list = []
            b_list = []
            gold_list = []
            w2v_pool_sq_list = []
            lex_pool_sq_list = []
            for idx in range(len(x_test_wrong)):
                h_pool_flat, appended_pool, prediction, w2v_pool_sq, lex_pool_sq, _b, score, correct = \
                    dev_step(tuple([x_test_wrong[idx]]), tuple([y_test_wrong[idx]]), tuple([x_lex_test_wrong[idx]]))
                pool_list.append(h_pool_flat)
                appended_pool_list.append(appended_pool)
                pred_list.append(prediction)
                correct_list.append(correct)
                score_list.append(score)
                b_list.append(_b)
                gold_list.append(y_test_wrong[idx])
                w2v_pool_sq_list.append(w2v_pool_sq)
                lex_pool_sq_list.append(lex_pool_sq)


            print len(pool_list), len(pool_list[0]), len(appended_pool_list), len(appended_pool_list[0])


            with open('./attention_sigma_analysis.pickle', 'wb') as handle:
                pickle.dump(wrong_index, handle)
                pickle.dump(pool_list , handle)
                pickle.dump(appended_pool_list, handle)
                pickle.dump(pred_list , handle)
                pickle.dump(correct_list , handle)
                pickle.dump(score_list, handle)
                pickle.dump(b_list, handle)
                pickle.dump(gold_list, handle)
                pickle.dump(wrong_pred, handle)
                pickle.dump(w2v_pool_sq_list, handle)
                pickle.dump(lex_pool_sq_list, handle)

                pickle.dump(vs[-2], handle)


            acc = test_step(x_test, y_test, x_lex_test)
            print 'acc=%f' % acc
            print 'correct_list', correct_list.count(True), len(correct_list)


def load_model_w2v(x_test, y_test, w2vdim, w2vnumfilters):
    savepath = './models/model-3400-w2v'


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = W2V_CNN(
                sequence_length=60,
                num_classes=3,
                embedding_size=w2vdim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=w2vnumfilters,
                l2_reg_lambda=2.0,
                l1_reg_lambda=0)

            with Timer("restore model..."):
                saver = tf.train.Saver(tf.all_variables())
                saver.restore(sess, savepath)

                var = [v for v in tf.trainable_variables()]

                vs = []
                for v in var:
                    vs.append(sess.run(v))


            print 'hello'


            def dev_step(x_batch, y_batch, x_batch_lex=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    # lexicon
                    # cnn.input_x_lexicon: x_batch_lex,
                    cnn.dropout_keep_prob: 1.0
                }

                accuracy, predictions, scores = sess.run(
                    [cnn.accuracy, cnn.predictions, cnn.scores],
                    feed_dict)

                if accuracy == 0:
                    return predictions[0], scores, False

                return predictions[0], scores, True

            def test_step(x_batch, y_batch, x_batch_lex=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    # lexicon
                    # cnn.input_x_lexicon: x_batch_lex,
                    cnn.dropout_keep_prob: 1.0
                }

                loss, accuracy, predictions, scores = sess.run(
                    [cnn.loss, cnn.accuracy, cnn.predictions, cnn.scores],
                    feed_dict)

                # loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                #     [cnn.loss, cnn.accuracy,
                #      cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                #     feed_dict)

                return accuracy

            pred = []
            wrong_index=[]
            wrong_pred = []
            for idx in range(len(x_test)):
                prediction, score, correct = \
                    dev_step(tuple([x_test[idx]]), tuple([y_test[idx]]))
                    # dev_step(tuple([x_test[idx]]), tuple([y_test[idx]]), tuple([x_lex_test[idx]]))
                    # dev_step(x_test, y_test)
                pred.append(prediction)
                if correct==False:
                    wrong_index.append(idx)
                    wrong_pred.append(prediction)



            acc = test_step(x_test, y_test)
            print 'acc=%f' % acc

    print 'predlen', len(pred)
    print 'wrong_indexlen', len(wrong_index)
    return pred, wrong_index, wrong_pred


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    w2vdim = 400
    max_len = 60
    lexdim = 15
    w2vnumfilters = 64
    lexnumfilters = 9

    x_test=[]
    y_test=[]
    x_lex_test=[]

    x_test, y_test, x_lex_test = load_dataset(w2vdim, max_len)
    pred, wrong_index, wrong_pred = load_model_w2v(x_test, y_test, w2vdim, w2vnumfilters)

    load_model_cnna2vind(x_test, y_test, x_lex_test, w2vdim, lexdim, lexnumfilters, w2vnumfilters, wrong_index, wrong_pred)



