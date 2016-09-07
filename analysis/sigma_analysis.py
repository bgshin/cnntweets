from cnntweets.utils.butils import Timer
import pickle
import numpy as np


with Timer('loading anal data...'):
    with open('./sigma_analysis.pickle', 'rb') as handle:
        index_list = pickle.load(handle)
        pool_list = pickle.load(handle)
        pred_list = pickle.load(handle)
        correct_list = pickle.load(handle)
        score_list = pickle.load(handle)
        b_list = pickle.load(handle)

        softmax_weight_list = pickle.load(handle)


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

def get_average(pool, w2v=True):
    if w2v:
        pool_data = pool[:,0:4*32]

    else:
        pool_data = pool[:,4*32:]

    return np.mean(pool_data, axis=0)


def process_one_class(cls):
    pool = np.array(pool_list[cls])
    softmax_weight = softmax_weight_list[:, cls]
    avg_w2v = get_average(pool, True)
    avg_lex = get_average(pool, False)

    softmax_weight_w2v = softmax_weight[0:4 * 32]
    softmax_weight_lex = softmax_weight[4 * 32:]

    w2v = np.sum(softmax_weight_w2v * avg_w2v)
    lex = np.sum(softmax_weight_lex * avg_lex)
    print avg_w2v[0:10]
    print avg_lex[0:10]
    print softmax_weight_w2v[:10]
    print softmax_weight_lex[:10]
    print w2v, lex


process_one_class(0) # 0 = neg
process_one_class(1) # 1 = obj
process_one_class(2) # 2 = pos
print 'j'
