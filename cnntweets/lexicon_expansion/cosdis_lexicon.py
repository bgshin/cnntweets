import argparse
import threading
from multiprocessing.pool import ThreadPool
import pickle
from cnntweets.utils.butils import Timer
from cnntweets.utils.word2vecReader import Word2Vec
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexindex', default=0, choices=[0, 1, 2, 3, 4, 5, 6], type=int)
    parser.add_argument('--nthread', default=40, type=int)
    parser.add_argument('--w2vdim', default=400, type=int)

    args = parser.parse_args()

    lexfile_list = ['EverythingUnigramsPMIHS.pickle',
                    'HS-AFFLEX-NEGLEX-unigrams.pickle',
                    'Maxdiff-Twitter-Lexicon_0to1.pickle',
                    'S140-AFFLEX-NEGLEX-unigrams.pickle',
                    'unigrams-pmilexicon.pickle',
                    'unigrams-pmilexicon_sentiment_140.pickle',
                    'BL.pickle']

    default_vector_dic = {'EverythingUnigramsPMIHS': [0],
                          'HS-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                          'Maxdiff-Twitter-Lexicon_0to1': [0.5],
                          'S140-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                          'unigrams-pmilexicon': [0, 0, 0],
                          'unigrams-pmilexicon_sentiment_140': [0, 0, 0],
                          'BL': [0]}

    lexindex = args.lexindex
    lexfile = lexfile_list[lexindex]
    lock = threading.Lock()

    print 'ADDITIONAL PARAMETER\n lexindex: %d\n lexfile: %s\n nthread: %d' % (
        args.lexindex, lexfile, args.nthread)
    sys.stdout.flush()

    with Timer("Loading lexicon..."):
        with open('../../data/le/%s' % lexfile, 'rb') as handle:
            lex_model = pickle.load(handle)

    initial_size = len(lex_model)
    print '=============initial size of lex_model is %d===============' % initial_size
    sys.stdout.flush()

    with Timer("Loading w2v..."):
        fname = '../../data/emory_w2v/w2v-%d.bin' % args.w2vdim
        w2vmodel = Word2Vec.load_word2vec_format(fname, binary=True)  # C binary format
    sys.stdout.flush()


    with Timer("splitting.."):
        voca = w2vmodel.vocab.keys()[100]
        N_voca = len(voca)
        print 'num_voca', N_voca

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        voca_for_thread = list(chunks(voca, N_voca / args.nthread))

    sys.stdout.flush()

    def expand_lex(sub_voca):
        num_exp = 0
        for key in sub_voca:
            if key in lex_model:
                continue

            top10_list = w2vmodel.most_similar(key)

            for cand in top10_list:
                if cand[0] in lex_model:
                    cand_vec = lex_model[cand[0]]
                    adj_vec = cand_vec * cand[1]
                    with lock:
                        lex_model[key] = adj_vec
                    num_exp = num_exp + 1
                    break

        print 'exped', num_exp
        sys.stdout.flush()


    def expand_lex_for_maxdiff(sub_voca):
        num_exp = 0
        for key in sub_voca:
            if key in lex_model:
                continue

            top10_list = w2vmodel.most_similar(key)

            for cand in top10_list:
                if cand[0] in lex_model:
                    cand_vec = lex_model[cand[0]]
                    adj_vec = [(x - 0.5) * cand[1] + 0.5 for x in cand_vec]
                    with lock:
                        lex_model[key] = adj_vec
                    num_exp = num_exp + 1
                    break

        print 'exped', num_exp
        sys.stdout.flush()

    # 0.4 -0.5 -> -0.1 -> -0.08 +0.5-> 0.42

    with Timer("run threads..."):
        pool = ThreadPool(processes=args.nthread)
        pool.map(expand_lex, voca_for_thread)
        pool.close()
        pool.join()

    expanded_size = len(lex_model)

    print '=============expanded size of lex_model is %d===============' % expanded_size
    print '=============expanded amount = %d' % (expanded_size-initial_size)
    sys.stdout.flush()


    with Timer("saving expanded lex for %s" % lexfile):
        with open('../../data/le/exp_cos.%s' % lexfile, 'wb') as handle:
            pickle.dump(lex_model, handle)




