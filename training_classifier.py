import nltk  #is used for natural language processing.
from nltk.corpus import stopwords       #to Know the English words which does not add much meaning to a sentence.
from nltk.tokenize import word_tokenize #extract the tokens from string of characters by using tokenize.
import nltk.classify.util as util       #collection of small Python functions and classes which make common patterns shorter and easier.
from nltk.classify import NaiveBayesClassifier #powerful algorithms for classification based on Bayes.
from nltk.metrics import BigramAssocMeasures   #used in bigrams where A collection of bigram association measures. Each association measure.
from nltk.collocations import BigramCollocationFinder as BCF
from nltk.metrics import ConfusionMatrix
import itertools   #implements a number of iterator building blocks inspired by constructs from APL.
import pickle      # way to convert a python object (list, dict, etc.) into a character stream.


def features(words):
	words = word_tokenize(words)

	scoreF = BigramAssocMeasures.chi_sq

	#bigram count
	n = 150

	bigrams = BCF.from_words(words).nbest(scoreF, n)

	return dict([word,True] for word in itertools.chain(words, bigrams))

def training():
	pos_sen = open("dataset/positive.txt", 'r', encoding = 'latin-1').read()
	neg_sen = open("dataset/negative.txt", 'r', encoding = 'latin-1').read()

	emoji = open("dataset/emoji.txt",'r', encoding = 'latin-1').read()
	pos_emoji = []
	neg_emoji = []
	for i in emoji.split('\n'):
		exp = ''
		if i[len(i)-2] == '-':
			for j in range(len(i) - 2):
				exp += i[j]
			neg_emoji.append(( {exp : True}, 'negative'))
		else:
			for j in range(len(i)-1):
				exp += i[j]
			pos_emoji.append(( {exp : True}, 'positive'))

	prev = [(features(words), 'positive') for words in pos_sen.split('\n')]
	nrev = [(features(words), 'negative') for words in neg_sen.split('\n')]
	ncutoff = int(len(nrev)*3/4)
	pcutoff = int(len(prev)*3/4)
	pos_set = prev + pos_emoji;
	neg_set = nrev + neg_emoji;
	train_data=pos_set+neg_set;
	#real_classifier = NaiveBayesClassifier.train(train_data);
	train_set = nrev[:ncutoff] + prev[:pcutoff] + pos_emoji + neg_emoji;
	test_set = nrev[ncutoff:] + prev[pcutoff:]
	'''if len_train >= len(train_set):
		len_train = len(train_set)-1'''
	test_classifier = NaiveBayesClassifier.train(train_set);
	print(len(train_set));
	print ("Accuracy is : ", util.accuracy(test_classifier, test_set) * 100);
	# SAVE IN FILE TO AVOID TRAIINING THE DATA AGAIN
	save_doc = open("classifier.pickle", 'wb')
	pickle.dump(test_classifier, save_doc)
	save_doc.close();ncutoff = int(len(nrev)*3/4);pcutoff = int(len(prev)*3/4);
'''
	#show_roc(lrb, X_test, y_test)
for i in range(1000, 14000, 1000):
	print('i: '+str(i))
	training(i)
'''
#training(1)
'''
	 TO TEST ACCURACY OF CLASSIFIER UNCCOMMENT THE CODE BELOW
	 ACCURACY : 78.1695423855964

	 ncutoff = int(len(nrev)*3/4)
	 pcutoff = int(len(prev)*3/4)
	 train_set = nrev[:ncutoff] + prev[:pcutoff] + pos_emoji + neg_emoji
	 test_set = nrev[ncutoff:] + prev[pcutoff:]
	 test_classifier = NaiveBayesClassifier.train(train_set)

	 print ("Accuracy is : ", util.accuracy(test_classifier, test_set) * 100)
'''