from cnntweets.utils.butils import Timer
import pickle
import numpy as np


with Timer('loading anal data...'):
    with open('./attention_sigma_analysis.pickle', 'rb') as handle:
        wrong_index = pickle.load(handle)
        pool_list = pickle.load(handle)
        appended_pool_list = pickle.load(handle)
        pred_list = pickle.load(handle)
        correct_list = pickle.load(handle)
        score_list = pickle.load(handle)
        b_list = pickle.load(handle)
        gold_list = pickle.load(handle)
        wrong_pred = pickle.load(handle)
        w2v_pool_sq_list = pickle.load(handle)
        lex_pool_sq_list = pickle.load(handle)

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


def get_each_sum(w2v, lex, w2vatt, lexatt, w_w2v, w_lex, w_w2vatt, w_lexatt, targetcls):
    return np.sum(w2v * w_w2v[:, targetcls]),  np.sum(lex * w_lex[:, targetcls]), \
           np.sum(w2vatt * w_w2vatt[:, targetcls]),  np.sum(lexatt * w_lexatt[:, targetcls])


def analysis(cls):
    pool = np.array(pool_list)
    appended_pool = np.array(appended_pool_list)
    avg_w2v = get_average(pool, True)
    avg_lex = get_average(pool, False)

    # softmax_weight_w2v = softmax_weight[0:4 * 32]
    # softmax_weight_lex = softmax_weight[4 * 32:]


    # 64*4+9*4 = 292
    # 707-292 = 415
    N_w2v = 64*4
    N_lex = 9*4
    N_w2vatt = 400
    N_lexatt = 15
    softmax_weight_w2v = softmax_weight_list[0:N_w2v, :]
    softmax_weight_lex = softmax_weight_list[N_w2v:N_w2v+N_lex, :]
    softmax_weight_w2vatt = softmax_weight_list[N_w2v+N_lex:N_w2v+N_lex+N_w2vatt, :]
    softmax_weight_lexatt = softmax_weight_list[N_w2v+N_lex+N_w2vatt:, :]

    # w2v = np.sum(softmax_weight_w2v * avg_w2v)
    # lex = np.sum(softmax_weight_lex * avg_lex)

    idx = 0
    w2vsum_all = []
    lexsum_all = []
    w2vattsum_all = []
    lexattsum_all = []

    for nodetarget in range(3):
        w2vsum_list = []
        lexsum_list = []
        w2vattsum_list = []
        lexattsum_list = []
        idx_list = []
        for idx in range(len(pool)):
            prediction = pred_list[idx]
            # print idx
            if correct_list[idx]==True and prediction==cls:
                # gold = cls
                w2vsum, lexsum, w2vattsum, lexattsum  = get_each_sum(appended_pool[0:, 0:N_w2v][idx],
                                              appended_pool[0:, N_w2v:N_w2v+N_lex][idx],
                                              appended_pool[0:, N_w2v+N_lex:N_w2v+N_lex+N_w2vatt][idx],
                                              appended_pool[0:, N_w2v+N_lex+N_w2vatt:][idx],
                                              softmax_weight_w2v, softmax_weight_lex,
                                              softmax_weight_w2vatt, softmax_weight_lexatt,
                                                                     nodetarget)

                w2vsum_list.append(w2vsum)
                lexsum_list.append(lexsum)
                w2vattsum_list.append(w2vattsum)
                lexattsum_list.append(lexattsum)
                idx_list.append(idx)


        print 'cls=%d, target=%d' % (cls, nodetarget)

        print len(idx_list)*1.0/len(correct_list)*100
        print '%f\t%f\t%f\t%f' % (np.mean(w2vsum_list), np.std(w2vsum_list), np.max(w2vsum_list), np.min(w2vsum_list))
        print '%f\t%f\t%f\t%f' % (np.mean(lexsum_list), np.std(lexsum_list), np.max(lexsum_list), np.min(lexsum_list))
        print '%f\t%f\t%f\t%f' % (np.mean(w2vattsum_list), np.std(w2vattsum_list), np.max(w2vattsum_list), np.min(w2vattsum_list))
        print '%f\t%f\t%f\t%f' % (np.mean(lexattsum_list), np.std(lexattsum_list), np.max(lexattsum_list), np.min(lexattsum_list))
        # map(int, w2vsum_list> (np.mean(w2vsum_list)+np.std(w2vsum_list)))

        w2vsum_all.append(w2vsum_list)
        lexsum_all.append(lexsum_list)
        w2vattsum_all.append(w2vattsum_list)
        lexattsum_all.append(lexattsum_list)


    # len(len(idx_list)) means number of data that predict correctly with gold 'cls'
    # i want to select only decision of EAV(w2vatt+lexatt) got correct
    for n, idx in enumerate(idx_list):
        score_w2v = [0, 0, 0]
        score_lex = [0, 0, 0]
        score_w2vatt = [0, 0, 0]
        score_lexatt = [0, 0, 0]
        score_others = [0,0,0]
        score_EAV = [0,0,0]
        score_all = [0, 0, 0]
        for target in range(3):
            score_w2v[target] = w2vsum_all[target][n]
            score_lex[target] = lexsum_all[target][n]
            score_w2vatt[target] = w2vattsum_all[target][n]
            score_lexatt[target] = lexattsum_all[target][n]
            score_others[target] = w2vsum_all[target][n] + lexsum_all[target][n]
            score_EAV[target] = w2vattsum_all[target][n] + lexattsum_all[target][n]
            score_all[target] = score_others[target] + score_EAV[target]

            gold_label = np.argmax(gold_list[idx])
            if np.argmax(score_others) != gold_label and np.argmax(score_EAV) == gold_label \
                    and pred_list[idx] == gold_label:
                print x_text[wrong_index[idx]]

        print 'w2v(%d), lex(%d), w2vatt(%d), lexatt(%d), others(%d), EAV(%d), all(%d), gold(%d), pred(%d)' \
              %(np.argmax(score_w2v), np.argmax(score_lex), np.argmax(score_w2vatt), np.argmax(score_lexatt),
                np.argmax(score_others), np.argmax(score_EAV), np.argmax(score_all),
                np.argmax(gold_list[idx]), pred_list[idx])

        # print 'w2v-%d(%f), lex-%d(%f), w2vatt-%d(%f), lexatt-%d(%f), others-%d(%f), EAV-%d(%f), all-%d(%f)' \
        #       % (cls, score_w2v[cls], cls, score_lex[cls],
        #          cls, score_w2vatt[cls], cls, score_lexatt[cls],
        #          cls, score_others[cls], cls, score_EAV[cls],
        #          cls, score_all[cls])












    #
    #
    # alpha_list = [i / 50.0 for i in range(50, 500)]
    # for alpha in alpha_list:
    #     if map(int, w2vsum_list > (np.mean(w2vsum_list) + alpha * np.std(w2vsum_list))).count(1)<=10:
    #         print 'num', map(int, w2vsum_list > (np.mean(w2vsum_list) + alpha * np.std(w2vsum_list))).count(1)
    #         selected_idx = np.where((w2vsum_list > (np.mean(w2vsum_list) + alpha * np.std(w2vsum_list))) == True)
    #         selected_index = np.array(idx_list)[selected_idx]
    #         big_w2v_index = index_list[cls][selected_index]
    #
    #         print np.array(x_sentiment)[big_w2v_index]
    #         print np.array(x_text)[big_w2v_index]
    #         break
    #
    # for alpha in alpha_list:
    #     if map(int, lexsum_list > (np.mean(lexsum_list) + alpha * np.std(lexsum_list))).count(1) <= 10:
    #         print 'num', map(int, lexsum_list > (np.mean(lexsum_list) + alpha * np.std(lexsum_list))).count(1)
    #         selected_idx = np.where((lexsum_list > (np.mean(lexsum_list) + alpha * np.std(lexsum_list))) == True)
    #         selected_index = np.array(idx_list)[selected_idx]
    #         big_lex_index = index_list[cls][selected_index]
    #         print np.array(x_sentiment)[big_lex_index]
    #         print np.array(x_text)[big_lex_index]
    #         break
    #



    # print avg_w2v[0:10]
    # print avg_lex[0:10]
    # print softmax_weight_w2v[:10]
    # print softmax_weight_lex[:10]
    # print w2v, lex



analysis(0) # 0 = neg
analysis(1) # 1 = obj
analysis(2) # 2 = pos
print 'j'
