from cnntweets.utils.butils import Timer
import pickle
import numpy as np


with Timer('loading anal data...'):
    with open('./sigma_analysis.pickle', 'rb') as handle:
        index = pickle.load(handle)
        pool = pickle.load(handle)
        pred = pickle.load(handle)
        correct = pickle.load(handle)

        softmax_weight = pickle.load(handle)


        # index_neg = pickle.load(handle)
        # index_obj = pickle.load(handle)
        # index_pos = pickle.load(handle)
        #
        # pool_neg = pickle.load(handle)
        # pool_obj = pickle.load(handle)
        # pool_pos = pickle.load(handle)
        #
        # pred_neg = pickle.load(handle)
        # pred_obj = pickle.load(handle)
        # pred_pos = pickle.load(handle)
        #
        # correct_neg = pickle.load(handle)
        # correct_obj = pickle.load(handle)
        # correct_pos = pickle.load(handle)



pool_neg = np.array(pool_neg)
pool_obj = np.array(pool_obj)
pool_pos = np.array(pool_pos)


def get_average(pool, w2v=True):
    if w2v:
        pool_data = pool[:,0:4*32]

    else:
        pool_data = pool[:,4*32:]

    return np.mean(pool_data, axis=0)

softmax_weight_neg = softmax_weight[:,0]
softmax_weight_obj = softmax_weight[:,1]
softmax_weight_pos = softmax_weight[:,2]


avg_neg_w2v = get_average(pool_neg, True)
avg_neg_lex = get_average(pool_neg, False)

avg_obj_w2v = get_average(pool_obj, True)
avg_obj_lex = get_average(pool_obj, False)

avg_pos_w2v = get_average(pool_pos, True)
avg_pos_lex = get_average(pool_pos, False)


softmax_weight_neg_w2v = softmax_weight_neg[0:4*32]
softmax_weight_obj_w2v = softmax_weight_obj[0:4*32]
softmax_weight_pos_w2v = softmax_weight_pos[0:4*32]


softmax_weight_neg_lex = softmax_weight_neg[4*32:]
softmax_weight_obj_lex = softmax_weight_obj[4*32:]
softmax_weight_pos_lex = softmax_weight_pos[4*32:]

neg_w2v = np.sum(softmax_weight_neg_w2v*avg_neg_w2v)
neg_lex = np.sum(softmax_weight_neg_lex*avg_neg_lex)

print 'j'