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
        gold_list = pickle.load(handle)

        softmax_weight_list = pickle.load(handle)

with Timer('loading anal data...'):
    template_txt = '../data/tweets/txt/%s'
    pathtxt = template_txt % 'tst'

    x_text=[line.split('\t')[2] for line in open(pathtxt, "r").readlines()]
    x_sentiment=[line.split('\t')[1] for line in open(pathtxt, "r").readlines()]




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

    for target in range(3):
        w2vsum_list = []
        lexsum_list = []
        idx_list = []
        for idx in range(len(pool)):
            # print idx
            if correct_list[cls][idx]==True:
                # evaluate_data(pool[0:,0:4*32][idx], pool[0:,4*32:][idx],
                #               softmax_weight_w2v, softmax_weight_lex,
                #               b_list[cls][idx], score_list[cls][idx][0])

                # gold = cls
                w2vsum, lexsum = get_each_sum(pool[0:,0:4*32][idx], pool[0:,4*32:][idx],
                              softmax_weight_w2v, softmax_weight_lex,
                             target)

                w2vsum_list.append(w2vsum)
                lexsum_list.append(lexsum)
                idx_list.append(idx)

        print map(int, correct_list[target]).count(0), map(int, correct_list[target]).count(1)
        print map(int, correct_list[target]).count(1)*1.0/len(correct_list[target])
        print target, len(pool), len(w2vsum_list), len(lexsum_list)
        print '%f\t%f\t%f\t%f' % (np.mean(w2vsum_list), np.std(w2vsum_list), np.max(w2vsum_list), np.min(w2vsum_list))
        print '%f\t%f\t%f\t%f' % (np.mean(lexsum_list), np.std(lexsum_list), np.max(lexsum_list), np.min(lexsum_list))
        # map(int, w2vsum_list> (np.mean(w2vsum_list)+np.std(w2vsum_list)))

    alpha_list = [i / 50.0 for i in range(50, 500)]
    for alpha in alpha_list:
        if map(int, w2vsum_list > (np.mean(w2vsum_list) + alpha * np.std(w2vsum_list))).count(1)<=10:
            print 'num', map(int, w2vsum_list > (np.mean(w2vsum_list) + alpha * np.std(w2vsum_list))).count(1)
            selected_idx = np.where((w2vsum_list > (np.mean(w2vsum_list) + alpha * np.std(w2vsum_list))) == True)
            selected_index = np.array(idx_list)[selected_idx]
            big_w2v_index = index_list[cls][selected_index]

            print np.array(x_sentiment)[big_w2v_index]
            print np.array(x_text)[big_w2v_index]
            break

    for alpha in alpha_list:
        if map(int, lexsum_list > (np.mean(lexsum_list) + alpha * np.std(lexsum_list))).count(1) <= 10:
            print 'num', map(int, lexsum_list > (np.mean(lexsum_list) + alpha * np.std(lexsum_list))).count(1)
            selected_idx = np.where((lexsum_list > (np.mean(lexsum_list) + alpha * np.std(lexsum_list))) == True)
            selected_index = np.array(idx_list)[selected_idx]
            big_lex_index = index_list[cls][selected_index]
            print np.array(x_sentiment)[big_lex_index]
            print np.array(x_text)[big_lex_index]
            break




    # print avg_w2v[0:10]
    # print avg_lex[0:10]
    # print softmax_weight_w2v[:10]
    # print softmax_weight_lex[:10]
    # print w2v, lex



process_one_class(0) # 0 = neg
process_one_class(1) # 1 = obj
process_one_class(2) # 2 = pos
print 'j'
