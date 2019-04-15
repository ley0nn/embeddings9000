'''
SVM systems for germeval, modified for English data from twitter and reddit, with crossvalidation and no test phase
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''
#### PARAMS #############################################
# source = 'Twitter'      ## options: Twitter, Reddit - Leon: I applied this on semeval&waseem
# source = 'Reddit'       #Leon: I applied this on wikimedia
# source = ''

# ftr = 'ngram'
# ftr = 'embeddings'
# ftr = 'embeddings+ngram'

# cls = 'bilstm'
# cls = ''

# dataSet = 'other_waseem_standardVSwikimedia'
# dataSet = 'other_waseem_wikimediaVSstandard'
# dataSet = 'other_standard_wikimediaVSwaseem'
# dataSet = 'waseem_standard_wikimediaVSother'
# dataSet = 'waseem_standard_wikimedia_otherVSstackoverflow'
# dataSet = 'waseem_standard_wikimedia_otherVSstandardTest_otherTest'

# dataSet = 'other'
# dataSet = 'standard'
# dataSet = 'WaseemHovy'
# dataSet = 'wikimedia'

# modelh5 = 'models/B2_model.h5'            #reddit_general + semeval
# modelh5 = 'models/B3_model.h5'          #reddit_polarised + semeval
# modelh5 = 'models/B4_model.h5'          #twitter_glove + semeval
# modelh5 = 'models/B5_model.h5'          #twitter_polarised_2016 + semeval
# modelh5 = 'models/CVWaseem_reddit_general_ruby_weights-improvement-{epoch:02d}-{loss:.2f}.h5'
# modelh5 = 'models/CVWaseem_reddit_polarised_ruby_weights-improvement-{epoch:02d}-{loss:.2f}.h5'
# modelh5 = 'models/CVWaseem_twitter_polarised_2016_weights-improvement-{epoch:02d}-{loss:.2f}.h5'
# modelh5 = 'models/CVWaseem_glove_twitter_weights-improvement-{epoch:02d}-{loss:.2f}.h5'

# modelh5 = 'models/CVWikimedia_reddit_general_ruby_weights-improvement-{epoch:02d}-{loss:.2f}.h5'
# modelh5 = 'models/CVWikimedia_reddit_polarised_ruby_weights-improvement-{epoch:02d}-{loss:.2f}.h5'
# modelh5 = 'models/CVWikimedia_twitter_polarised_2016_weights-improvement-{epoch:02d}-{loss:.2f}.h5'
# modelh5 = 'models/CVWikimedia_glove_twitter_weights-improvement-{epoch:02d}-{loss:.2f}.h5'

# tknzr = 'models/B2_tokenizer.pickle'    #reddit_general + semeval
# tknzr = 'models/B3_tokenizer.pickle'  #reddit_polarised + semeval
# tknzr = 'models/B4_tokenizer.pickle'  #twitter_glove + semeval
# tknzr = 'models/B5_tokenizer.pickle'  #twitter_polarised_2016 + semeval
# tknzr = 'models/CVWaseem_tokenizer.pickle'
# tknzr = 'models/CVWikimedia_tokenizer.pickle'

# trainPath = '../../Full_Tweets_June2016_Dataset.csv'          # WaseemHovy - waseemhovy
# trainPath = '../../4563973/toxicity_annotated_comments.tsv'     # Wikimedia toxicity_annotated_comments
# trainPath = '../../public_development_en/train_en.tsv'        # SemEval - standard
# trainPath = '../../english/agr_en_train.csv'                    # Facebook english - other

# testPath = ''
# testPath = '../../public_development_en/dev_en.tsv'         # SemEval - standard
# testPath = '../../english/agr_en_dev.csv'                    # Facebook english - other

# path_to_embs = '../../embeddings/reddit_general.txt'
# path_to_embs = '../../embeddings/reddit_polarised.txt'
# path_to_embs = '../../embeddings/reddit_general_ruby.txt'
# path_to_embs = '../../embeddings/reddit_polarised_ruby.txt'
# path_to_embs = '../../embeddings/twitter_polarised_2016.txt'
# path_to_embs = '../../embeddings/glove.twitter.27B.200d.txt'

# evlt = 'cv10'
# evlt = 'traintest'

# clean = 'none'
# clean = 'std'     # PPsmall
# clean = 'ruby'    # PPbig

glove_embeds_path = '../../embeddings/glove.twitter.27B.200d.txt'

#########################################################

import helperFunctions
import transformers
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm
import os

import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle

from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from nltk.tokenize import TweetTokenizer, word_tokenize, MWETokenizer
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

seed = 1337
np.random.seed(seed)


MWET = MWETokenizer([   ('<', 'url', '>'),
						('<', 'user', '>'),
						('<', 'smile', '>'),
						('<', 'lolface', '>'),
						('<', 'sadface', '>'),
						('<', 'neutralface', '>'),
						('<', 'heart', '>'),
						('<', 'number', '>'),
						('<', 'hashtag', '>'),
						('<', 'allcaps', '>'),
						('<', 'repeat', '>'),
						('<', 'elong', '>'),
					], separator='')

def ntlktokenizer(x):
	tokens = word_tokenize(x)           # tokenize
	tokens = MWET.tokenize(tokens)      # fix <url> and <user> etc.

	return ' '.join(tokens)

if __name__ == '__main__':
# python3 SVM_original.py -src '' -ftr 'embeddings' -cls 'bilstm' -ds 'WaseemHovy' -mh5 'models/CVWaseem_reddit_general_ruby_weights-improvement-{epoch:02d}-{loss:.2f}.h5' -tknzr 'models/CVWaseem_tokenizer.pickle' -trnp '../../Full_Tweets_June2016_Dataset.csv' -tstp '' -pte '../../embeddings/reddit_general_ruby.txt' -evlt '' -cln 'none'
	parser = argparse.ArgumentParser(description='ALW')
	parser.add_argument('-src', type=str, help='source')
	parser.add_argument('-ftr', type=str, help='ftr')
	parser.add_argument('-cls', type=str, help='cls')
	parser.add_argument('-ds', type=str, help='dataSet')
	parser.add_argument('-mh5', type=str, help='modelh5')
	parser.add_argument('-tknzr', type=str, help='tknzr')
	parser.add_argument('-trnp', type=str, help='trainPath')
	parser.add_argument('-tstp', type=str, help='testPath')
	parser.add_argument('-pte', type=str, help='path_to_embs')
	parser.add_argument('-evlt', type=str, help='evlt')
	parser.add_argument('-cln', type=str, help='clean')
	args = parser.parse_args()

	source = args.src
	ftr = args.ftr
	cls = args.cls
	dataSet = args.ds
	modelh5 = args.mh5
	tknzr = args.tknzr
	trainPath = args.trnp
	testPath = args.tstp
	path_to_embs = args.pte
	evlt = args.evlt
	clean = args.cln

	TASK = 'binary'
	#TASK = 'multi'

	'''
	Preparing data
	'''

	print('Reading in ' + source + ' training data...' + dataSet)

	IDsTrain, Xtrain, Ytrain, IDsTest, Xtest, Ytest = helperFunctions.loaddata(dataSet, trainPath, testPath, cls, TASK)

	print('Done reading in data...')
	print('Train labels', set(Ytrain))
	print('Test labels', set(Ytest))
	print(cls)

	offensiveRatio = Ytrain.count('OFF')/len(Ytrain)
	nonOffensiveRatio = Ytrain.count('NOT')/len(Ytrain)

	# Minimal preprocessing / cleaning

	if clean == 'std':
		Xtrain = helperFunctions.clean_samples(Xtrain)
	if clean == 'ruby':
		Xtrain = helperFunctions.clean_samples_ruby(Xtrain)
		if testPath != '':
			Xtest = helperFunctions.clean_samples_ruby(Xtest)

	print(len(Xtrain), 'training samples!')
	'''
	Preparing vectorizer and classifier
	'''

	# Vectorizing data / Extracting features
	print('Preparing tools (vectorizer, classifier) ...')

	# unweighted word uni and bigrams
	### This gives the stop_words may be inconsistent warning
	if source == 'Twitter':
		tokenizer = TweetTokenizer().tokenize
	elif source == 'Reddit':
		# tokenizer = None
		tokenizer = ntlktokenizer
	else:
		tokenizer = None

		### TODO: define tokenizer for Reddit data

	if ftr == 'ngram':
		count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
		count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
		vectorizer = FeatureUnion([ ('word', count_word),
									('char', count_char)])

	elif ftr == 'embeddings':
		print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
		embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
		# glove_embeds = {}
		if path_to_embs == glove_embeds_path:
			glove_embeds = embeddings
		else:
			glove_embeds, glove_vocab = helperFunctions.load_embeddings(glove_embeds_path)
		print('Done')
		vectorizer = features.Embeddings(embeddings, glove_embeds, pool='max')

	elif ftr == 'embeddings+ngram':
		count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
		count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
		# path_to_embs = 'embeddings/model_reset_random.bin'
		print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
		embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
		# glove_embeds = {}
		if path_to_embs == glove_embeds_path:
			glove_embeds = embeddings
		else:
			glove_embeds, glove_vocab = helperFunctions.load_embeddings(glove_embeds_path)
		print('Done')
		vectorizer = FeatureUnion([ ('word', count_word),
									('char', count_char),
									('word_embeds', features.Embeddings(embeddings, glove_embeds, pool='max'))])

	if cls == 'bilstm':
		from BiLSTM import biLSTM
		training = True
		output = False
		cv = False
		traindev = True
		if traindev:
			Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=seed)
		Ytest, Yguess = biLSTM(Xtrain, Ytrain, Xtest, Ytest, training, output, embeddings, tknzr, modelh5, cv)


	# Set up SVM classifier with unbalanced class weights
	if TASK == 'binary' and cls != 'bilstm':
		# cl_weights_binary = None
		cl_weights_binary = {'NOT':1/nonOffensiveRatio, 'OFF':1/offensiveRatio}
		clf = LinearSVC(class_weight=cl_weights_binary)
	elif TASK == 'multi':
		# cl_weights_multi = None
		cl_weights_multi = {'OTHER':0.5,
							'ABUSE':3,
							'INSULT':3,
							'PROFANITY':4}
		clf = LinearSVC(class_weight=cl_weights_multi)

	if cls != 'bilstm':
		classifier = Pipeline([
								('vectorize', vectorizer),
								('classify', clf)])



	'''
	Actual training and predicting:
	'''

	if evlt == 'cv10':
		print('10-fold cross validation results:')
		results = (cross_validate(classifier, Xtrain, Ytrain,cv=10, verbose=1))
		# print(results)
		print(sum(results['test_score']) / 10)
		print('\n\nDone, used the following parameters:')
		print('train: {}'.format(trainPath))
		if ftr != 'ngram':
			print('embed: {}'.format(path_to_embs))
		print('feats: {}'.format(ftr))
		print('prepr: {}'.format(clean))
		print('sourc: {} - datas: {}'.format(source, dataSet))
	elif evlt == 'traintest':
		if cls != 'bilstm':
			classifier.fit(Xtrain,Ytrain)
			Yguess = classifier.predict(Xtest)
		print('train test results:')
		print(accuracy_score(Ytest, Yguess))
		print(precision_recall_fscore_support(Ytest, Yguess, average='weighted'))
		print(classification_report(Ytest, Yguess))
		print('\n\nDone, used the following parameters:')
		print('train: {}'.format(trainPath))
		print('tests: {}'.format(testPath))
		if ftr != 'ngram':
			print('embed: {}'.format(path_to_embs))
		print('feats: {}'.format(ftr))
		print('prepr: {}'.format(clean))
		print('sourc: {} - datas: {}'.format(source, dataSet))






	#######
