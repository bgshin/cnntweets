import argparse
import pickle
from cnntweets.utils.butils import Timer
from cnntweets.utils.word2vecReader import Word2Vec
import sys


def get_candidate(init_model, w2vmodel):
    candidates = set()


    for idx, word in enumerate(init_model.keys()):
        if word in w2vmodel:
            top10 = w2vmodel.most_similar(word)
            for cand in top10:
                if cand[0] not in init_model:
                    candidates.add(cand[0])

        if idx%1000 == 0:
            print len(init_model), idx


    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexindex', default=0, choices=[0, 1, 2, 3, 4, 5, 6], type=int)
    parser.add_argument('--w2vdim', default=400, type=int)


    args = parser.parse_args()

    lexfile_list = ['EverythingUnigramsPMIHS.pickle',
                    'HS-AFFLEX-NEGLEX-unigrams.pickle',
                    'Maxdiff-Twitter-Lexicon_0to1.pickle',
                    'S140-AFFLEX-NEGLEX-unigrams.pickle',
                    'unigrams-pmilexicon.pickle',
                    'unigrams-pmilexicon_sentiment_140.pickle',
                    'BL.pickle']

    lexindex = args.lexindex
    lexfile = lexfile_list[lexindex]

    print 'ADDITIONAL PARAMETER\n lexindex: %d\n lexfile: %s\n w2vdim: %d' % (
        args.lexindex, lexfile, args.w2vdim)
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

    with Timer("get_candidate..."):
        cand_words = get_candidate(lex_model, w2vmodel)

    print '=============expanded amount = %d' % (len(cand_words))
    sys.stdout.flush()

    with Timer("Loading all-expanded lexicon..."):
        fname = '../../data/le/exp_1.1.%s' % lexfile
        with open(fname, 'rb') as handle:
            exp_lex_model = pickle.load(handle)

    sys.stdout.flush()

    with Timer("appending to old lex model..."):
        for w in cand_words:
            lex_model[w] = exp_lex_model[w]

    sys.stdout.flush()

    expanded_size = len(lex_model)
    print '=============expanded_size= %d' % (expanded_size)
    sys.stdout.flush()

    with Timer("saving expanded lex for %s" % lexfile):
        with open('../../data/le/exp_compact.%s' % lexfile, 'wb') as handle:
            pickle.dump(lex_model, handle)




