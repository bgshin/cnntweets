#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import argparse
import os
from cnntweets.utils import cnn_data_helpers
from cnntweets.cnn_models.w2v_lex_cnn import W2V_LEX_CNN
from cnntweets.utils.word2vecReader import Word2Vec
import gc
import pickle
from cnntweets.utils.butils import Timer

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("test_every", 100000, "Evaluate model on test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
os.system('cls' if os.name == 'nt' else 'clear')



# Data Preparatopn
# ==================================================


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
        fname = '../data/le/%s' % lexfile

        with open(fname, 'rb') as handle:
            each_model = pickle.load(handle)
            default_vector = default_vector_dic[lexfile.replace('.pickle', '')]
            each_model["<PAD/>"] = default_vector
            norm_model.append(each_model)


    return norm_model

def load_dataset():
    with Timer("loading w2v..."):
        w2vmodel = load_w2v(w2vdim)

    with Timer("loading lex..."):
        unigram_lexicon_model = load_lex()

    with Timer("loading test dataset..."):
        x_test, y_test, x_lex_test = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len)
    del(w2vmodel)
    gc.collect()

    return x_test, y_test, x_lex_test


def load_model(x_test, y_test, x_lex_test, w2vdim, lexdim, lexnumfilters, w2vnumfilters):
    savepath = './models/model-5000-w2vlex-32-8'


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = W2V_LEX_CNN(
                # sequence_length=x_test.shape[1],
                sequence_length=60,
                num_classes=3,
                embedding_size=w2vdim,
                embedding_size_lex=lexdim,
                num_filters_lex=lexnumfilters,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=w2vnumfilters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            with Timer("restore model..."):
                saver = tf.train.Saver(tf.all_variables())
                saver.restore(sess,savepath)

                var = [v for v in tf.trainable_variables()]

                vs = []
                for v in var:
                    vs.append(sess.run(v))

            # if senti == 'objective':
            #     y.append([0, 1, 0])
            #
            # elif senti == 'positive':
            #     y.append([0, 0, 1])
            #
            # else:  # negative
            #     y.append([1, 0, 0])



            print 'hello'

            index_neg = np.where(y_test[:, 0] == 1)[0]
            index_obj = np.where(y_test[:, 1] == 1)[0]
            index_pos = np.where(y_test[:, 2] == 1)[0]

            x_test_neg = x_test[index_neg]
            x_test_obj = x_test[index_obj]
            x_test_pos = x_test[index_pos]

            y_test_neg = y_test[index_neg]
            y_test_obj = y_test[index_obj]
            y_test_pos = y_test[index_pos]

            x_lex_test_neg = x_lex_test[index_neg]
            x_lex_test_obj = x_lex_test[index_obj]
            x_lex_test_pos = x_lex_test[index_pos]


            # batches = cnn_data_helpers.batch_iter(
            #     list(zip(x_test, y_test, x_lex_test)), 1, 1)

            # for idx, batch in enumerate(batches):
            #     x_batch, y_batch, x_batch_lex = zip(*batch)
            #     print idx, y_batch

            def dev_step(x_batch, y_batch, x_batch_lex=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    # lexicon
                    cnn.input_x_lexicon: x_batch_lex,
                    cnn.dropout_keep_prob: 1.0
                }

                accuracy, h_pool, h_pool_flat, predictions, _b, scores = sess.run(
                    [ cnn.accuracy, cnn.h_pool, cnn.h_pool_flat, cnn.predictions, cnn._b, cnn.scores],
                    feed_dict)

                if accuracy==0:
                    return h_pool_flat[0], predictions[0], _b, scores, False

                return h_pool_flat[0], predictions[0], _b, scores, True

            pool_neg = []
            pred_neg = []
            correct_neg = []
            score_neg = []
            b_neg = []
            gold_neg = []
            for idx in range(len(x_test_neg)):
                h_pool_flat, prediction, _b, score, correct = \
                    dev_step(tuple([x_test_neg[idx]]), tuple([y_test_neg[idx]]), tuple([x_lex_test_neg[idx]]))
                pool_neg.append(h_pool_flat)
                pred_neg.append(prediction)
                correct_neg.append(correct)
                score_neg.append(score)
                b_neg.append(_b)
                gold_neg.append(y_test_neg[idx])

            pool_obj = []
            pred_obj = []
            correct_obj = []
            score_obj = []
            b_obj = []
            gold_obj = []
            for idx in range(len(x_test_obj)):
                h_pool_flat, prediction, _b, score, correct = \
                    dev_step(tuple([x_test_obj[idx]]), tuple([y_test_obj[idx]]), tuple([x_lex_test_obj[idx]]))
                pool_obj.append(h_pool_flat)
                pred_obj.append(prediction)
                correct_obj.append(correct)
                score_obj.append(score)
                b_obj.append(_b)
                gold_obj.append(y_test_obj[idx])

            pool_pos = []
            pred_pos = []
            correct_pos = []
            score_pos = []
            b_pos = []
            gold_pos = []
            for idx in range(len(x_test_pos)):
                h_pool_flat, prediction, _b, score, correct = \
                    dev_step(tuple([x_test_pos[idx]]), tuple([y_test_pos[idx]]), tuple([x_lex_test_pos[idx]]))
                pool_pos.append(h_pool_flat)
                pred_pos.append(prediction)
                correct_pos.append(correct)
                score_pos.append(score)
                b_pos.append(_b)
                gold_pos.append(y_test_pos[idx])

            print len(index_neg), len(index_obj), len(index_pos)
            print len(pool_neg), len(pool_obj), len(pool_pos)


            with open('./sigma_analysis.pickle', 'wb') as handle:
                pickle.dump([index_neg, index_obj, index_pos], handle)
                pickle.dump([pool_neg, pool_obj, pool_pos] , handle)
                pickle.dump([pred_neg, pred_obj, pred_pos] , handle)
                pickle.dump([correct_neg, correct_obj, correct_pos] , handle)
                pickle.dump([score_neg, score_obj, score_pos], handle)
                pickle.dump([b_neg, b_obj, b_pos], handle)
                pickle.dump([gold_neg, gold_obj, gold_pos], handle)


                pickle.dump(vs[-2], handle)

                # pickle.dump(index_neg, handle)
                # pickle.dump(index_obj, handle)
                # pickle.dump(index_pos, handle)
                #
                # pickle.dump(pool_neg, handle)
                # pickle.dump(pool_obj, handle)
                # pickle.dump(pool_pos, handle)
                #
                # pickle.dump(pred_neg, handle)
                # pickle.dump(pred_obj, handle)
                # pickle.dump(pred_pos, handle)
                #
                # pickle.dump(correct_neg, handle)
                # pickle.dump(correct_obj, handle)
                # pickle.dump(correct_pos, handle)
                #
                # pickle.dump(vs[-2], handle)









if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    w2vdim = 400
    max_len = 60
    lexdim = 15
    w2vnumfilters = 32
    lexnumfilters = 8

    x_test=[]
    y_test=[]
    x_lex_test=[]

    x_test, y_test, x_lex_test = load_dataset()
    load_model(x_test, y_test, x_lex_test, w2vdim, lexdim, lexnumfilters, w2vnumfilters)


