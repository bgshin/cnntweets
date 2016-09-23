## Lexicon dataset


## Source
* From [this](http://saifmohammad.com/WebPages/lexicons.html)


## At a glance
| Lexicon                           | Raw Range                       | Normalized Range |
|-----------------------------------|---------------------------------|------------------|
| BL                                | -1 or 1 (binary)                | -                |
| EverythingUnigramsPMIHS           | [-10.7608678862, 10.4950860272] | [-1,1]           |
| HS-AFFLEX-NEGLEX-unigrams         | [-10.025, 10.661]               | [-1,1]           |
| HS-AFFLEX-NEGLEX-unigrams         | [0, 133421]                     | [0,1]            |
| HS-AFFLEX-NEGLEX-unigrams         | [0, 190358]                     | [0,1]            |
| Maxdiff-Twitter-Lexicon_0to1      | [0.008, 0.992]                  | [0,1]            |
| S140-AFFLEX-NEGLEX-unigrams       | [-5.844, 4.495]                 | [-1,1]           |
| S140-AFFLEX-NEGLEX-unigrams       | [0, 381203]                     | [0,1]            |
| S140-AFFLEX-NEGLEX-unigrams       | [0, 469618]                     | [0,1]            |
| unigrams-pmilexicon               | [-6.925, 7.526]                 | [-1,1]           |
| unigrams-pmilexicon               | [0, 195575]                     | [0,1]            |
| unigrams-pmilexicon               | [0, 335321]                     | [0,1]            |
| unigrams-pmilexicon_sentiment_140 | [-4.999, 5.0]                   | [-1,1]           |
| unigrams-pmilexicon_sentiment_140 | [0, 295790]                     | [0,1]            |
| unigrams-pmilexicon_sentiment_140 | [0, 472371]                     | [0,1]            |



## File format

### HashtagSentimentAffLexNegLex
* Name: [NRC Hashtag Affirmative Context Sentiment Lexicon and NRC Hashtag Negated Context Sentiment Lexicon ](https://github.com/WladimirSidorenko/SemEval-2016/blob/master/scripts/data/HashtagSentimentAffLexNegLex/readme.txt)
* Cite
	* @article{kiritchenko2014sentiment,
  title={Sentiment analysis of short informal texts},
  author={Kiritchenko, Svetlana and Zhu, Xiaodan and Mohammad, Saif M},
  journal={Journal of Artificial Intelligence Research},
  volume={50},
  pages={723--762},
  year={2014}
}
* filename: HS-AFFLEX-NEGLEX-unigrams.txt
* f = 3 (3 dim vector)
	* dim=1: <score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
	* dim=2: <Npos> is the number of times the term appears in the positive class, ie. in tweets with positive hashtag or emoticon;
	* dim=3: <Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative hashtag or emoticon.

* N = 43,949

### MaxDiff-Twitter-Lexicon
* Name: [MaxDiff Twitter Sentiment Lexicon](https://github.com/WladimirSidorenko/SemEval-2016/tree/master/scripts/data/MaxDiff-Twitter-Lexicon)
* Cite
	* @article{kiritchenko2014sentiment,
  title={Sentiment analysis of short informal texts},
  author={Kiritchenko, Svetlana and Zhu, Xiaodan and Mohammad, Saif M},
  journal={Journal of Artificial Intelligence Research},
  volume={50},
  pages={723--762},
  year={2014}
}

* filename: Maxdiff-Twitter-Lexicon_0to1.txt
* f = 1 (scalar)
	* range: [0 1], 0 (most negative) and 1 (most positive)
	* **neutral is 0.5**
* N = 1,515


### NRC-Hashtag-Sentiment-Lexicon-v0.1
* Name: [NRC(National Research Council Canada) Hashtag Sentiment Lexicon](https://github.com/pedrobalage/TwitterHybridClassifier/tree/master/Data/Lexicon/NRC-Hashtag-Sentiment-Lexicon-v0.1)
* @InProceedings{MohammadKZ2013,
  author    = {Mohammad, Saif and Kiritchenko, Svetlana and Zhu, Xiaodan},
  title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
  booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
  month     = {June},
  year      = {2013},
  address   = {Atlanta, Georgia, USA}
}
* filename: unigrams-pmilexicon.txt
* f = 3
	* dim = 1: sentimentScore is a real number. A positive score indicates positive sentiment. A negative score indicates negative sentiment.
	* dim = 2: numPositive is the number of times the term co-occurred with a positive marker such as a positive emoticon or a positive hashtag
	* dim = 3: numNegative is the number of times the term co-occurred with a negative marker such as a negative emoticon or a negative hashtag.
* N = 54,129

### Sentiment140-Lexicon-v0.1
* Name: [The Sentiment140 Lexicon](https://github.com/felipebravom/StaticTwitterSent/tree/master/extra/Sentiment140-Lexicon-v0.1)
* @InProceedings{MohammadKZ2013,
  author    = {Mohammad, Saif and Kiritchenko, Svetlana and Zhu, Xiaodan},
  title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
  booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
  month     = {June},
  year      = {2013},
  address   = {Atlanta, Georgia, USA}
}
* filename: unigrams-pmilexicon_sentiment_140.txt
* f = 3
	* dim = 1: Terms with a non-zero PMI score with positive emoticons and PMI score of 0 with negative emoticons were assigned a sentimentScore of 5.
	  Terms with a non-zero PMI score with negative emoticons and PMI score of 0 
	  with positive emoticons were assigned a sentimentScore of -5.

	* dim = 2: numPositive is the number of times the term co-occurred with a positive marker such as a positive emoticon or a positive emoticons.

	* dim = 3: numNegative is the number of times the term co-occurred with a negative marker such as a negative emoticon or a negative emoticons.
* N = 62,468

### Sentiment140AffLexNegLex
* Name: [NRC Sentiment140 Lexicons](https://github.com/balikasg/SemEval2016-Twitter_Sentiment_Evaluation/tree/master/src/lexicons/Sentiment140AffLexNegLex)
* @article{go2009twitter,
  title={Twitter sentiment classification using distant supervision},
  author={Go, Alec and Bhayani, Richa and Huang, Lei},
  journal={CS224N Project Report, Stanford},
  volume={1},
  pages={12},
  year={2009}
}
* filename: S140-AFFLEX-NEGLEX-unigrams.txt
* f = 3
	* dim = 1: <score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
	* dim = 2: <Npos> is the number of times the term appears in the positive class, ie. in tweets with positive hashtag or emoticon;
	* dim = 3: <Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative hashtag or emoticon.

* N = 55,146

### ??
* filename: EverythingUnigramsPMIHS.txt
* f = 1
* N = 4,376


### BL
* filename: BL.txt
* @inproceedings{hu2004mining,
  title={Mining and summarizing customer reviews},
  author={Hu, Minqing and Liu, Bing},
  booktitle={Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={168--177},
  year={2004},
  organization={ACM}
}
*  Bing Liu [7] Opinion Lexicon consists of 6786 words, of which 2006 positive and 4783 negative. 
* f = 1, 
* range: BINARY (-1 or 1)
* normalized range:  BINARY (-1 or 1)
* N = 6,786

