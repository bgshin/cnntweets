from utils.butils import Timer
from utils import cnn_data_helpers
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf
from utils.word2vecReader import Word2Vec
from cnn_models.w2v_trainable import W2V_CNN_TRAINABLE

max_len = 60
w2vdim = 50
the_base_path = '../data/emory_w2v/'
the_model_path = the_base_path + 'w2v-%d.bin' % w2vdim


#
# input_x = tf.placeholder(tf.int32, [None, 4], name="input_x")
# W = tf.Variable(
#     tf.random_uniform([1000, 50], -1.0, 1.0),
#     trainable=True,
#     name="W")
# embedded_chars = tf.nn.embedding_lookup(W, input_x)
#
#
# xdata = np.array([[0, 5, 17, 33]])
#
#
# feed_dict = {
#     input_x: xdata
# }
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# embedded_chars = sess.run([embedded_chars], feed_dict)

def load_w2v(w2vdim, simple_run = True, source = "twitter"):
    if simple_run:
        return {'a': np.array([np.float32(0.0)] * w2vdim)}

    else:
        if source == "twitter":
            model_path = '../data/emory_w2v/w2v-%d.bin' % w2vdim
        elif source == "amazon":
            model_path = '../data/emory_w2v/w2v-%d-%s.bin' % (w2vdim, source)

        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

        return model


with Timer("w2v"):
    w2vmodel = load_w2v(w2vdim, simple_run=False)

# print matrix[ids]
with Timer("LOADING Data..."):
    x_train, y_train = cnn_data_helpers.build_w2v_data('trn_sample', w2vmodel, max_len)
    x_train2, y_train2 = cnn_data_helpers.load_data_trainable("trn_sample", rottenTomato=False)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_len)
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit_transform(x_train2)
    total_vocab_size = len(vocab_processor.vocabulary_._freq) + 1
    x_train2 = np.array(list(vocab_processor.fit_transform(x_train2)))

# with Timer("LOADING Data..."):
#     x_train, y_train = cnn_data_helpers.load_data_trainable("trn_sample", rottenTomato=False)
#     vocab_processor = learn.preprocessing.VocabularyProcessor(60)
#
#     total_vocab_size = len(vocab_processor.vocabulary_._freq)+1

initW = np.random.uniform(0.0, 0.0, (total_vocab_size, w2vdim))

# with Timer("Assigning w2v..."):
#     # initial matrix with random uniform
#     initW = np.random.uniform(0.0, 0.0, (total_vocab_size, w2vdim))
#
#     for idx, word in enumerate(vocab_processor.vocabulary_._reverse_mapping):
#         if w2vmodel.vocab.has_key(word) == True:
#             initW[idx] = w2vmodel[word]

with Timer("LOADING W2V..."):
    print("LOADING word2vec file {} \n".format(the_model_path))
    # W2V
    with open(the_model_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab_processor.vocabulary_.get(word)
            if idx != 0:
                # print str(idx) + " -> " + word
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

sess = tf.Session()
with sess.as_default():
    cnn = W2V_CNN_TRAINABLE(
        sequence_length=x_train.shape[1],
        num_classes=3,
        vocab_size=total_vocab_size,
        embedding_size=w2vdim,
        filter_sizes=[2],
        num_filters=1,
        embedding_size_lex=15,
        num_filters_lex=0,
        l2_reg_lambda=0.5,
        trainable=False)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.initialize_all_variables())

    with Timer('Initialize embedding weights with w2v...'):
        sess.run(cnn.W.assign(initW))

    feed_dict = {
        cnn.input_x: x_train2,
        cnn.input_y: y_train2,
        # lexicon
        # cnn.input_x_lexicon: x_batch_lex,
        cnn.dropout_keep_prob: 0.5
    }

    _, step, cnnW, embedded_chars = sess.run(
        [train_op, global_step, cnn.W, cnn.embedded_chars],
        feed_dict)

    print embedded_chars
    print x_train
    print cnnW
    print initW

print 'h'

