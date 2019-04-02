'''
SVM systems for germeval, modified for English data from twitter and reddit, with crossvalidation and no test phase
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''
#### PARAMS #############################################
source = 'Twitter'      ## options: Twitter, Reddit
# source = 'Reddit'

# dataSet = 'other'
# dataSet = 'WaseemHovy'
dataSet = 'standard'
# dataSet = 'wikimedia'

# dataSet = 'other_waseem_standardVSwikimedia'
# dataSet = 'other_waseem_wikimediaVSstandard'
# dataSet = 'other_standard_wikimediaVSwaseem'
# dataSet = 'waseem_standard_wikimediaVSother'
# dataSet = 'waseem_standard_wikimedia_otherVSstackoverflow'
# dataSet = 'waseem_standard_wikimedia_otherVSstandardTest_otherTest'

# ftr = 'ngram'
# ftr = 'embeddings'
ftr = 'embeddings+ngram'

evlt = 'cv10'
# evlt = 'traintest'

# clean = 'none'
# clean = 'std'     # PPsmall
clean = 'ruby'    # PPbig

# trainPath = '../../english/agr_en_train.csv'                    # Facebook english - other
# trainPath = '../../Full_Tweets_June2016_Dataset.csv'          # WaseemHovy - waseemhovy
trainPath = '../../public_development_en/train_en.tsv'        # SemEval - standard
# trainPath = '../../4563973/toxicity_annotated_comments.tsv'     # Wikimedia toxicity_annotated_comments

testPath = ''
# testPath = '../../public_development_en/dev_en.tsv'         # SemEval - standard
# testPath = '../../english/agr_en_dev.csv'                    # Facebook english - other

# path_to_embs = '../../embeddings/reddit_general.txt'
# path_to_embs = '../../embeddings/reddit_general_ruby.txt'
# path_to_embs = '../../embeddings/reddit_polarised.txt'
# path_to_embs = '../../embeddings/reddit_polarised_ruby.txt'
path_to_embs = '../../embeddings/twitter_polarised_2016.txt'
# path_to_embs = '../../glove.twitter.27B/glove.twitter.27B.200d.txt'


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
    print(tokens)

    return ' '.join(tokens)

if __name__ == '__main__':

    TASK = 'binary'
    #TASK = 'multi'

    '''
    Preparing data
    '''


    print('Reading in ' + source + ' training data...' + dataSet)
    IDsTrain = []
    Xtrain = []
    Ytrain = []
    IDsTest = []
    Xtest = []
    Ytest = []
    if dataSet == 'WaseemHovy':
        if TASK == 'binary':
            IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus_WaseemHovy(trainPath)
        else:
            IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus_WaseemHovy(trainPath)
    elif dataSet == 'standard':
        IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus(trainPath)
        if testPath == '../../public_development_en/dev_en.tsv':
            IDsTest,Xtest,Ytest = helperFunctions.read_corpus(testPath)
    elif dataSet == 'other_waseem_standardVSwikimedia':
        IDsWaseem,Xwaseem,Ywaseem = helperFunctions.read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv')
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = helperFunctions.read_corpus('../../public_development_en/train_en.tsv')
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = helperFunctions.read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv')
        IDsOther,Xother,Yother = helperFunctions.read_corpus_otherSet('../../english/agr_en_train.csv')
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
    elif dataSet == 'other_waseem_wikimediaVSstandard':
        IDsWaseem,Xwaseem,Ywaseem = helperFunctions.read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv')
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = helperFunctions.read_corpus('../../public_development_en/train_en.tsv')
        IDsWikimedia,Xwikimedia,Ywikimedia = helperFunctions.read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv')
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = helperFunctions.read_corpus_otherSet('../../english/agr_en_train.csv')
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
    elif dataSet == 'other_standard_wikimediaVSwaseem':
        IDsTest,Xtest,Ytest = helperFunctions.read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv')
        IDsStandard,Xstandard,Ystandard = helperFunctions.read_corpus('../../public_development_en/train_en.tsv')
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = helperFunctions.read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv')
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = helperFunctions.read_corpus_otherSet('../../english/agr_en_train.csv')
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
    elif dataSet == 'waseem_standard_wikimediaVSother':
        IDsWaseem,Xwaseem,Ywaseem = helperFunctions.read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv')
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = helperFunctions.read_corpus('../../public_development_en/train_en.tsv')
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = helperFunctions.read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv')
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = helperFunctions.read_corpus_otherSet('../../english/agr_en_train.csv')
    elif dataSet == 'waseem_standard_wikimedia_otherVSstandardTest_otherTest':
        IDsWaseem,Xwaseem,Ywaseem = helperFunctions.read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv')
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = helperFunctions.read_corpus('../../public_development_en/train_en.tsv')
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = helperFunctions.read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv')
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = helperFunctions.read_corpus_otherSet('../../english/agr_en_train.csv')
    # elif dataSet == 'waseem_standard_wikimedia_otherVSstackoverflow':
    #     IDsWaseem,Xwaseem,Ywaseem = helperFunctions.read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv')
    #     for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
    #         IDsTrain.append(id)
    #         Xtrain.append(x)
    #         Ytrain.append(y)
    #     IDsStandard,Xstandard,Ystandard = helperFunctions.read_corpus('../../public_development_en/train_en.tsv')
    #     for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
    #         IDsTrain.append(id)
    #         Xtrain.append(x)
    #         Ytrain.append(y)
    #     IDsOther,Xother,Yother = helperFunctions.read_corpus_otherSet('../../english/agr_en_train.csv')
    #     for id,x,y in zip(IDsOther,Xother,Yother):
    #         IDsTrain.append(id)
    #         Xtrain.append(x)
    #         Ytrain.append(y)
    #     IDsWikimedia,Xwikimedia,Ywikimedia = helperFunctions.read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv')
    #     for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
    #         IDsTrain.append(id)
    #         Xtrain.append(x)
    #         Ytrain.append(y)
    #     IDsTest,Xtest,Ytest = helperFunctions.read_corpus_stackoverflow('...')
    elif dataSet == 'wikimedia':
        IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus_wikimedia(trainPath)
    else:
        IDsTrain,Xtrain,Ytrain = helperFunctions.read_corpus_otherSet(trainPath)
        ## TODO: implement reading function for the Reddit data


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
    else:
        # tokenizer = None
        tokenizer = ntlktokenizer
        ### TODO: define tokenizer for Reddit data

    if ftr == 'ngram':
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
        vectorizer = FeatureUnion([ ('word', count_word),
                                    ('char', count_char)])

    elif ftr == 'embeddings':
        print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        print('Done')
        vectorizer = features.Embeddings(embeddings, pool='max')

    elif ftr == 'embeddings+ngram':
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
        # path_to_embs = 'embeddings/model_reset_random.bin'
        print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        print('Done')
        vectorizer = FeatureUnion([ ('word', count_word),
                                    ('char', count_char),
                                    ('word_embeds', features.Embeddings(embeddings, pool='max'))])


    # Set up SVM classifier with unbalanced class weights
    if TASK == 'binary':
        # cl_weights_binary = None
        cl_weights_binary = {'NOT':1/nonOffensiveRatio, 'OFF':1/offensiveRatio}
        clf = LinearSVC(class_weight=cl_weights_binary)
    else:
        # cl_weights_multi = None
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        clf = LinearSVC(class_weight=cl_weights_multi)

    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            ('classify', clf)])



    '''
    Actual training and predicting:
    '''

    if evlt == 'cv10':
        print('10-fold cross validation results:')
        results = (cross_validate(classifier, Xtrain, Ytrain,cv=10))
        # print(results)
        print(sum(results['test_score']) / 10)
        print('\n\nDone, used the following parameters:')
        print('train: {}'.format(trainPath))
        if ftr != 'ngram':
            print('embed: {}'.format(path_to_embs))
        print('feats: {}'.format(ftr))
        print('prepr: {}'.format(clean))
    elif evlt == 'traintest':
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






    #######
