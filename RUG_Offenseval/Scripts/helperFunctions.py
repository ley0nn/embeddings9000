
import re
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from random import shuffle
import pandas as pd
import os



def read_corpus_otherSet(corpus_file, binary=True): #Facebook Enlgish
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

def read_corpus_WaseemHovy(corpus_file):
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
            if data[len(data)-1] == 'none':
                labels.append('NOT')
            else:
                labels.append('OFF')
    mapIndexPosition = list(zip(ids, tweets, labels))
    random.seed(1337)
    shuffle(mapIndexPosition)
    ids, tweets, labels = zip(*mapIndexPosition)

    return ids, tweets, labels

def read_corpus(corpus_file, binary=True):
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

            if data[2] == '1':
                labels.append('OFF')
            elif data[2] == '0':
                labels.append('NOT')

    return ids[1:], tweets[1:], labels


def read_corpus_wikimedia(corpus_file, binary=True):
    '''Reading in data from corpus file'''
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
