from cnntweets.utils.butils import Timer
import pickle
import numpy as np


with Timer('loading anal data...'):
    with open('./sigma_analysis.pickle', 'rb') as handle:
        index_neg = pickle.load(handle)
        index_obj = pickle.load(handle)
        index_pos = pickle.load(handle)

        pool_neg = pickle.load(handle)
        pool_obj = pickle.load(handle)
        pool_pos = pickle.load(handle)

        pred_neg = pickle.load(handle)
        pred_obj = pickle.load(handle)
        pred_pos = pickle.load(handle)

        correct_neg = pickle.load(handle)
        correct_obj = pickle.load(handle)
        correct_pos = pickle.load(handle)

        softmax_weight = pickle.load(handle)

pool_neg = np.array(pool_neg)
pool_obj = np.array(pool_obj)
pool_pos = np.array(pool_pos)


def get_average(pool, w2v=True):
    if w2v:
        pool_data = pool[:, 0:4 * 32]

    else:
        pool_data = pool[:,4*32:]

    return np.mean(pool_data, axis=0)

avg_neg_w2v = get_average(pool_neg, True)
avg_neg_lex = get_average(pool_neg, False)

print 'j'