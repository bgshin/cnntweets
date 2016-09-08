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


def get_average(pool, w2v=True):
    if w2v:
        pool_data = pool[:,0:4*32]

    else:
        pool_data = pool[:,4*32:]

    return np.mean(pool_data, axis=0)

def evaluate_data(w2v, lex, w_w2v, w_lex, bias, score):
    # score_estimate = np.sum(w2v*w_w2v)+np.sum(lex*w_lex)+bias

    score_estimate_list = []
    for i in range(3):
        score_estimate = np.sum(w2v * w_w2v[:, i]) + np.sum(lex * w_lex[:, i]) + bias[i]
        score_estimate_list.append(score_estimate)

    if np.sum(score-score_estimate_list)>0.0001:
        print 'worng', score_estimate, score,


def get_each_sum(w2v, lex, w_w2v, w_lex, cls):
    return np.sum(w2v * w_w2v[:, cls]),  np.sum(lex * w_lex[:, cls])


def process_one_class(cls):
    pool = np.array(pool_list[cls])
    softmax_weight = softmax_weight_list[:, cls]
    avg_w2v = get_average(pool, True)
    avg_lex = get_average(pool, False)

    # softmax_weight_w2v = softmax_weight[0:4 * 32]
    # softmax_weight_lex = softmax_weight[4 * 32:]

    softmax_weight_w2v = softmax_weight_list[0:4 * 32, :]
    softmax_weight_lex = softmax_weight_list[4 * 32:, :]

    # w2v = np.sum(softmax_weight_w2v * avg_w2v)
    # lex = np.sum(softmax_weight_lex * avg_lex)

    idx = 0

    for idx in range(len(pool)):
        print idx
        if correct_list[cls][idx]==True:
            evaluate_data(pool[0:,0:4*32][idx], pool[0:,4*32:][idx],
                          softmax_weight_w2v, softmax_weight_lex,
                          b_list[cls][idx], score_list[cls][idx][0])

            score_list[cls][idx][0]
            get_each_sum(pool[0:,0:4*32][idx], pool[0:,4*32:][idx],
                          softmax_weight_w2v, softmax_weight_lex,
                         cls)


    # print avg_w2v[0:10]
    # print avg_lex[0:10]
    # print softmax_weight_w2v[:10]
    # print softmax_weight_lex[:10]
    # print w2v, lex



process_one_class(0) # 0 = neg
process_one_class(1) # 1 = obj
process_one_class(2) # 2 = pos
print 'j'
