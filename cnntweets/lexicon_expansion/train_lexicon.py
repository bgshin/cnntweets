import tensorflow as tf
import numpy as np
from cnntweets.utils.cnn_data_helpers import batch_iter

import sys
import os
import time
from regression import MultiClassRegression, MultiClassRegression2Layer
import pickle
from cnntweets.utils.butils import Timer
from sklearn.utils import shuffle

# Misc Parameters
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")




def load_data(lexfile):
    with Timer("loading pickle for %s" % lexfile):
        with open('../../data/le/%s' % lexfile, 'rb') as handle:
            x = pickle.load(handle)
            y = pickle.load(handle)


    return x, y



def find_best_model(dataset, w2vdim, outputdim, neg_output):

    with tf.Graph().as_default():
        min_loss_dev = 100000000000
        min_avg_loss_dev = 100000000000
        step_min_loss_dev = 0
        loss_tst_at_min_loss_dev = 0
        avgloss_tst_at_min_loss_dev = 0

        sess = tf.Session()

        with sess.as_default():
            mcr = MultiClassRegression(w2vdim, outputdim, l2_reg_lambda=0.0, neg_output=neg_output)
            # mcr = MultiClassRegression2Layer(w2vdim, outputdim, l2_reg_lambda=0.2)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)

            # # construct an optimizer to minimize cost and fit line to my data
            # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

            grads_and_vars = optimizer.compute_gradients(mcr.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    mcr.input_x: x_batch,
                    mcr.input_y: y_batch,
                }

                _, step, scores, loss, avgloss = sess.run(
                    [train_op, global_step, mcr.scores, mcr.loss, mcr.avgloss],
                    feed_dict)

                return loss, avgloss


            def dev_step(x_batch, y_batch):
                feed_dict = {
                    mcr.input_x: x_batch,
                    mcr.input_y: y_batch,
                }

                _, step, scores, loss, avgloss = sess.run(
                    [train_op, global_step, mcr.scores, mcr.loss, mcr.avgloss],
                    feed_dict)

                return loss, avgloss

            # Generate batches
            batches = batch_iter(
                list(zip(dataset.x_trn, dataset.y_trn)), FLAGS.batch_size, FLAGS.num_epochs)

            for batch in batches:
                x_batch, y_batch = zip(*batch)



                curr_loss_trn, curr_avg_loss_trn = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                print 'Train: [%d] Curr loss for trn (%f) avgloss(%f)\n' % (
                    current_step, curr_loss_trn, curr_avg_loss_trn)
                sys.stdout.flush()



                if current_step % FLAGS.evaluate_every == 0:
                    # print("Evaluation:")

                    curr_loss_dev, curr_avg_loss_dev = dev_step(x_dev, y_dev)
                    update_tst = False

                    if curr_loss_dev < min_loss_dev:
                        min_loss_dev = curr_loss_dev
                        step_min_loss_dev = current_step
                        update_tst = True

                    if update_tst:
                        curr_loss_tst, curr_avg_loss_tst = dev_step(x_tst, y_tst)
                        loss_tst_at_min_loss_dev = curr_loss_tst

                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                    print 'Status: [%d] Min loss for dev (%f), Min loss for tst (%f), AvgLossTst(%f)\n' % (
                        step_min_loss_dev, min_loss_dev, loss_tst_at_min_loss_dev, curr_avg_loss_tst)
                    sys.stdout.flush()

            print 'Status: [%d] Min loss for dev (%f), Min loss for tst (%f), AvgLossTst(%f)\n' % (
                step_min_loss_dev, min_loss_dev, loss_tst_at_min_loss_dev, curr_avg_loss_tst)
            sys.stdout.flush()


if __name__ == "__main__":
    lexfile_list = ['EverythingUnigramsPMIHS.pickle',
                    'HS-AFFLEX-NEGLEX-unigrams.pickle',
                    'Maxdiff-Twitter-Lexicon_0to1.pickle',
                    'S140-AFFLEX-NEGLEX-unigrams.pickle',
                    'unigrams-pmilexicon.pickle',
                    'unigrams-pmilexicon_sentiment_140.pickle',
                    'BL.pickle']

    range_neg = {'EverythingUnigramsPMIHS': True,
                 'HS-AFFLEX-NEGLEX-unigrams': True, # True, False, False
                 'Maxdiff-Twitter-Lexicon_0to1': False, # 0 to 1
                 'S140-AFFLEX-NEGLEX-unigrams': True, # True, False, False
                 'unigrams-pmilexicon': True, # True, False, False
                 'unigrams-pmilexicon_sentiment_140':  True,
                 'BL': True
                 }  # True, False, False

    lexfile = lexfile_list[6]
    neg_output = range_neg[lexfile.replace('.pickle', '')]

    x_all_order, y_all_order = load_data(lexfile)
    x_all, y_all = shuffle(x_all_order, y_all_order, random_state=0)

    VALIDATION_RATIO = 0.1
    TEST_RATIO = 0.3

    NUM_DATA = x_all.shape[0]
    VALIDATION_SIZE = int(NUM_DATA * VALIDATION_RATIO)
    TEST_SIZE = int(NUM_DATA * TEST_RATIO)
    TRAIN_SIZE = NUM_DATA - (VALIDATION_SIZE + TEST_SIZE)

    x_trn = x_all[0:TRAIN_SIZE]
    y_trn = y_all[0:TRAIN_SIZE]

    x_dev = x_all[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    y_dev = y_all[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

    x_tst = x_all[TRAIN_SIZE + VALIDATION_SIZE:]
    y_tst = y_all[TRAIN_SIZE + VALIDATION_SIZE:]

    class Dataset(object):
        def __init__(self, name):
            self.name = name

    dataset = Dataset(lexfile)
    dataset.x_trn = x_trn
    dataset.y_trn = y_trn

    dataset.x_dev = x_dev
    dataset.y_dev = y_dev

    dataset.x_tst = x_tst
    dataset.y_tst = y_tst

    x_dim = x_all.shape[1]
    y_dim = y_all.shape[1]


    find_best_model(dataset, x_dim, y_dim, neg_output=neg_output)
    print lexfile
    print 'NUM_DATA(%d), y_dim(%d)' % (NUM_DATA, y_dim)






