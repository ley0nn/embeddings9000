# stackoverflow_path = '/data/s2548798/Leon/stackoverflow/StackOverflow/stack_comments.csv'

import re
import csv
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from random import shuffle
import pandas as pd
import os

def loaddata(dataSet, trainPath, testPath, cls, TASK):

    IDsTrain = []
    Xtrain = []
    Ytrain = []
    IDsTest = []
    Xtest = []
    Ytest = []
    if dataSet == 'WaseemHovy':
        if TASK == 'binary':
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)
        else:
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)
    elif dataSet == 'standard':
        IDsTrain,Xtrain,Ytrain = read_corpus(trainPath,cls)
        if testPath == '../../public_development_en/dev_en.tsv':
            IDsTest,Xtest,Ytest = read_corpus(testPath,cls)
    elif dataSet == 'other_waseem_standardVSwikimedia':
        IDsWaseem,Xwaseem,Ywaseem = read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv',cls)
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = read_corpus('../../public_development_en/train_en.tsv',cls)
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add SemEval dev-data
        IDsStandard_test,Xstandard_test,Ystandard_test = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y in zip(IDsStandard_test,Xstandard_test,Ystandard_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = read_corpus_otherSet('../../english/agr_en_train.csv',cls)
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add Facebook dev-data
        IDsOther_test,Xother_test,Yother_test = read_corpus_otherSet('../../english/agr_en_dev.csv',cls)
        for id,x,y in zip(IDsOther_test,Xother_test,Yother_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv',cls)

    elif dataSet == 'other_waseem_wikimediaVSstandard':
        IDsWaseem,Xwaseem,Ywaseem = read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv',cls)
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv',cls)
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = read_corpus_otherSet('../../english/agr_en_train.csv',cls)
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add Facebook dev-data
        IDsOther_test,Xother_test,Yother_test = read_corpus_otherSet('../../english/agr_en_dev.csv',cls)
        for id,x,y in zip(IDsOther_test,Xother_test,Yother_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = read_corpus('../../public_development_en/train_en.tsv',cls)
        # Also add SemEval dev-data
        IDsTest2,Xtest2,Ytest2 = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y in zip(IDsTest2,Xtest2,Ytest2):
            IDsTest.append(id)
            Xtest.append(x)
            Ytest.append(y)

    elif dataSet == 'other_standard_wikimediaVSwaseem':
        IDsStandard,Xstandard,Ystandard = read_corpus('../../public_development_en/train_en.tsv',cls)
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add SemEval dev-data
        IDsStandard_test,Xstandard_test,Ystandard_test = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y in zip(IDsStandard_test,Xstandard_test,Ystandard_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv',cls)
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = read_corpus_otherSet('../../english/agr_en_train.csv',cls)
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add Facebook dev-data
        IDsOther_test,Xother_test,Yother_test = read_corpus_otherSet('../../english/agr_en_dev.csv',cls)
        for id,x,y in zip(IDsOther_test,Xother_test,Yother_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv',cls)

    elif dataSet == 'waseem_standard_wikimediaVSother':
        IDsWaseem,Xwaseem,Ywaseem = read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv',cls)
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = read_corpus('../../public_development_en/train_en.tsv',cls)
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add SemEval dev-data
        IDsStandard_test,Xstandard_test,Ystandard_test = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y in zip(IDsStandard_test,Xstandard_test,Ystandard_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv',cls)
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = read_corpus_otherSet('../../english/agr_en_train.csv',cls)
        IDsTest2,Xtest2,Ytest2 = read_corpus_otherSet('../../english/agr_en_dev.csv',cls)
        for id,x,y in zip(IDsTest2,Xtest2,Ytest2):
            IDsTest.append(id)
            Xtest.append(x)
            Ytest.append(y)

    elif dataSet == 'waseem_standard_wikimedia_otherVSstandardTest_otherTest':
        IDsWaseem,Xwaseem,Ywaseem = read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv',cls)
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = read_corpus('../../public_development_en/train_en.tsv',cls)
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv',cls)
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = read_corpus_otherSet('../../english/agr_en_train.csv',cls)
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsTest,Xtest,Ytest = read_corpus_otherSet('../../english/agr_en_dev.csv',cls)
        IDsTest2,Xtest2,Ytest2 = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y in zip(IDsTest2,Xtest2,Ytest2):
            IDsTest.append(id)
            Xtest.append(x)
            Ytest.append(y)

    elif dataSet == 'waseem_standard_wikimedia_otherVSstackoverflow':
        IDsTest,Xtest,Ytest = read_corpus_stackoverflow('/data/s2548798/Leon/stackoverflow/StackOverflow/stack_comments.csv',cls)
        IDsWaseem,Xwaseem,Ywaseem = read_corpus_WaseemHovy('../../Full_Tweets_June2016_Dataset.csv',cls)
        for id,x,y in zip(IDsWaseem,Xwaseem,Ywaseem):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsStandard,Xstandard,Ystandard = read_corpus('../../public_development_en/train_en.tsv',cls)
        for id,x,y in zip(IDsStandard,Xstandard,Ystandard):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add SemEval dev-data
        IDsStandard_test,Xstandard_test,Ystandard_test = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y in zip(IDsStandard_test,Xstandard_test,Ystandard_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsOther,Xother,Yother = read_corpus_otherSet('../../english/agr_en_train.csv',cls)
        for id,x,y in zip(IDsOther,Xother,Yother):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        # Also add Facebook dev-data
        IDsOther_test,Xother_test,Yother_test = read_corpus_otherSet('../../english/agr_en_dev.csv',cls)
        for id,x,y in zip(IDsOther_test,Xother_test,Yother_test):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
        IDsWikimedia,Xwikimedia,Ywikimedia = read_corpus_wikimedia('../../4563973/toxicity_annotated_comments.tsv',cls)
        for id,x,y in zip(IDsWikimedia,Xwikimedia,Ywikimedia):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)

    elif dataSet == 'wikimedia':
        IDsTrain,Xtrain,Ytrain = read_corpus_wikimedia(trainPath,cls)
    else:
        IDsTrain,Xtrain,Ytrain = read_corpus_otherSet(trainPath,cls)
        ## TODO: implement reading function for the Reddit data

    return IDsTrain, Xtrain, Ytrain, IDsTest, Xtest, Ytest

def read_corpus_otherSet(corpus_file, cls, binary=True): #Facebook Enlgish
    '''Reading in data from corpus file'''
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        fi = fi.readlines()
        ids = []
        tweets = []
        labels = []

        line = 0
        while line < (len(fi)):
            d = fi[line].strip().split(',')
            while d[-1] not in ['CAG', 'NAG', 'OAG']:
                line += 1
                dataPart = fi[line].strip().split(',')
                d += dataPart

            data = [d[0], "".join(d[1:len(d)-1]) ,d[len(d)-1]]

            # making sure no missing labels
            if len(data) != 3:
                raise IndexError('Missing data for tweet "%s"' % data[0])
            #print(data)
            ids.append(data[0])
            tweets.append(data[1])
            if binary:
                if cls == 'bilstm':
                    if data[2] == 'NAG':
                        labels.append(0)
                    else:
                        labels.append(1)
                else:
                    if data[2] == 'NAG':
                        labels.append('NOT')
                    else:
                        labels.append('OFF')
            else:
                labels.append(data[2])
            line += 1
    #print(labels)
    print("read " + str(len(tweets)) + " tweets.")
    return ids, tweets, labels

def read_corpus_WaseemHovy(corpus_file,cls):
    '''Reading in data from corpus file'''
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='ISO-8859-1') as fi:
        for line in fi:
            data = line.strip().split(',')
            ids.append(data[0])
            if len(data)<3:
                continue
            if len(data)>3:
                tweets.append("".join(data[1:len(data) - 2]))
            else:
                tweets.append(data[1])
            if cls == 'bilstm':
                if data[len(data)-1] == 'none':
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                if data[len(data)-1] == 'none':
                    labels.append('NOT')
                else:
                    labels.append('OFF')
    mapIndexPosition = list(zip(ids, tweets, labels))
    shuffle(mapIndexPosition)
    ids, tweets, labels = zip(*mapIndexPosition)

    return ids, tweets, labels

def read_corpus(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
            if cls == 'bilstm':
                if data[2] == '1':
                    labels.append(1)
                elif data[2] == '0':
                    labels.append(0)
            else:
                if data[2] == '1':
                    labels.append('OFF')
                elif data[2] == '0':
                    labels.append('NOT')

    return ids[1:], tweets[1:], labels


def read_corpus_wikimedia(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    if cls == 'bilstm':
        label_dict = {True: 1, False: 0}
    else:
        label_dict = {True: 'OFF', False: 'NOT'}

    comments = pd.read_csv(corpus_file, sep = '\t')
    ids = list(comments['rev_id'])
    comments = pd.read_csv(corpus_file, sep = '\t', index_col = 0)
    annotations = pd.read_csv('../../4563973/toxicity_annotations.tsv',  sep = '\t')
    # len(annotations['rev_id'].unique())
    # labels a comment as an atack if the majority of annoatators did so
    labels = annotations.groupby('rev_id')['toxicity_score'].mean() > 0.5
    # join labels and comments
    comments['toxicity_score'] = labels
    # remove newline and tab tokens
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    comments['toxicity_score'] = comments['toxicity_score'].map(label_dict)
    tweets = list(comments['comment'])
    labels = list(comments['toxicity_score'])

    return ids, tweets, labels


def read_corpus_stackoverflow(corpus_file,cls):
    '''Reading in data from corpus file'''

    Offensive = []
    Unwelcoming = []
    Not_Constructive_Or_Off_Topic = []
    Obsolete = []
    Other = []
    No_Longer_Needed = []
    Too_Chatty = []

    with open(corpus_file) as csvfile:
        next(csvfile)
        readCSV = csv.reader(csvfile, delimiter=',')
        for line in readCSV:
            text = line[1]
            flag = line[2]
            if len(line) < 3 or flag == 'NA':
                continue

            if flag == 'Comment Rude Or Offensive':
                Offensive.append(text)
            elif flag == 'CommentUnwelcoming':
                Unwelcoming.append(text)
            elif flag == 'Comment Not Constructive Or Off Topic':
                Not_Constructive_Or_Off_Topic.append(text)
            elif flag == 'Comment Obsolete':
                Obsolete.append(text)
            elif flag == 'Comment Other':
                Other.append(text)
            elif flag == 'CommentNoLongerNeeded':
                No_Longer_Needed.append(text)
            elif flag == 'Comment Too Chatty':
                Too_Chatty.append(text)

    n = 10000
    random.seed(1337)

    list1 = random.sample(Unwelcoming, n)
    list2 = random.sample(Not_Constructive_Or_Off_Topic, n)
    list3 = random.sample(Obsolete, n)
    list4 = random.sample(Other, n)
    list5 = random.sample(No_Longer_Needed, n)
    list6 = random.sample(Too_Chatty, n)

    not_list = list1 + list2 + list3 + list4 + list5+ list6

    ids = []
    tweets = []
    labels = []
    count = 0

    for comment in Offensive:
        ids.append(count)
        tweets.append(comment)
        labels.append('OFF')
        count += 1
    for comment in not_list:
        ids.append(count)
        tweets.append(comment)
        labels.append('NOT')
        count += 1

    mapIndexPosition = list(zip(ids, tweets, labels))
    shuffle(mapIndexPosition)
    ids, tweets, labels = zip(*mapIndexPosition)

    return ids, tweets, labels


def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings stored as txt
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''

    print('Using embeddings: ', embedding_file)
    if embedding_file.endswith('.txt'):
        w2v = {}
        vocab = []
        try:
            f = open(embedding_file,'r')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    float(values[1])
                except ValueError:
                    continue
                coefs = np.asarray(values[1:], dtype='float')
                w2v[word] = coefs
                vocab.append(word)
        except UnicodeDecodeError:
            f = open(embedding_file,'rb')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    float(values[1])
                except ValueError:
                    continue
                coefs = np.asarray(values[1:], dtype='float')
                w2v[word] = coefs
                vocab.append(word)

        print ("Done.",len(w2v)," words loaded!")
        return w2v, vocab
        f.close()



def clean_samples(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    new_samples = []
    for tw in samples:

        # tw = re.sub(r'@\S+','User', tw)
        # tw = re.sub(r'\|LBR\|', '', tw)
        # tw = re.sub(r'http\S+\s?', '', tw)
        # tw = re.sub(r'\#', '', tw)
        # new_samples.append(tw)

        tw = re.sub(r'@\S+','<user>', tw)
        tw = re.sub(r'\|LBR\|', '', tw)
        tw = re.sub(r'http\S+\s?', '<url>', tw)
        tw = re.sub(r'\#', '<hashtag>', tw)
        new_samples.append(tw)

    return new_samples

def clean_samples_ruby(samples):
    tmpname = 'tmpdir/tmp_' + str(time.time()) + '.txt'
    with open(tmpname, 'w') as tmp_file:
        for line in samples:
            tmp_file.write(line + '\n')

    command_tmp = 'ruby -n preprocess-twitter.rb ' + tmpname
    clean = os.popen(command_tmp).read().split('\n')

    new_samples = []
    for line in clean[:-1]:
        # if not line:
        #     continue
        new_samples.append(line)
    return new_samples

def load_offense_words(path):
    ow = []
    f = open(path, "r")
    for line in f:
        ow.append(line[:-1])
    return ow


def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''
    labs = sorted(set(Ygold + Yguess.tolist()))
    """ print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:") """

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array



    # printing out precision, recall, f-score for each class in easily readable way
    PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
    """ print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))

    print('-'*50)
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print() """

    return [PRFS, labs]


def mean(list):
    return sum(list)/len(list)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
