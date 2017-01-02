##########################################
#### Probabilistic Graphical Models ######
################ Project #################
##########################################

import sys
import itertools
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
import operator
import re
from lib import ibm as ibm
from lib import hmm
from lib import import_corpus as imp

# flake8: noqa

def parseArguments():
    parser = argparse.ArgumentParser(description="Learn a word alignment model")

    parser.add_argument("french_corpus",
        help="path_to_french_corpus")

    parser.add_argument("english_corpus",
        help="path_to_english_corpus")

    parser.add_argument('-m', '-method', nargs='*',
        type=int, help="(optional) Specify the models to learn: 1/IBM1 (defaults to IBM1) 2/ IBM2 3/HMM")

    args = parser.parse_args()
    return args

#############################################
### GENERAL FUNCTIONS IN import_corpus.py ###
#############################################

#############################################
########### IBM METHODS in ibm.py ###########
#############################################

def display_results(fr_dict,en_dict,P,target):
    assert len(fr_dict) == np.size(P)[0], "dimensions mismatch"
    assert len(en_dict) == np.size(P)[1], "dimensions mismatch"
    print(P)


## OUTPUT DISPLAY METHODS ##

def print_P_to_csv(en_dict, fr_dict, P, output_path = 'output.csv'):
    f = open(output_path,'w')
    # header
    f.write(',')
    for en_word in en_dict:
        f.write(en_word)
        f.write(',')
    # remove last comma
    f.seek(0, 2)
    size = f.tell()
    f.truncate(size-1)
    f.write('\n')

    # big loop
    for i in range(P.shape[0]):
        f.write(fr_dict[i])
        f.write(',')
        for j in range(P.shape[1]):
            f.write("%.4f"%P[i,j])
            if j<range(P.shape[1])[-1]:
                f.write(',')
            else:
                f.write('\n')

    # remove last \n
    f.seek(0, 2)
    size = f.tell()
    f.truncate(size - 2)
    f.close()

def most_likely_alignment(fr_sentence, en_sentence, fr_dict,
                            en_dict, P, method_index, lamb=0, p_null=0):
    en_indices = []

    for word in en_sentence:
        en_idx = imp.hash(en_dict, word)
        en_indices.append(en_idx)

    tmp_P = P[:, en_indices[0]]
    for k in range(1, len(en_indices)):
        tmp_P = np.hstack((tmp_P, P[:, en_indices[k]]))

    alignment = []
    # get the most likely alignment
    for i in range(len(fr_sentence)):
        word = fr_sentence[i]
        fr_idx = imp.hash(fr_dict, word)

        if method_index == 1:
            likelihood = tmp_P[fr_idx, :]
        elif method_index == 2:
            likelihood = np.zeros(len(en_sentence))
            #import pdb; pdb.set_trace()
            for j in range(len(en_sentence)):
                likelihood[j] = ibm.b(j, i, len(en_sentence)-1,
                                len(fr_sentence), lamb, p_null) * tmp_P[fr_idx, j]

        alignment.append(np.argmax(likelihood))
    return alignment

def plot_sentence_alignment(fr_corpus, en_corpus, en_dict, fr_dict,
                                P, idx, method_index, lamb=0, p_null=0, figure=plt):
    # idx is the index of the sentence to plot
    # x-axis : English
    # y-axis : French
    en_sentence = re.split(' |\'', en_corpus[idx])
    fr_sentence = re.split(' |\'', fr_corpus[idx])

    alignment = most_likely_alignment(fr_sentence, en_sentence,
                                fr_dict, en_dict, P, method_index, lamb, p_null)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(range(len(fr_sentence)),alignment,'ro',ls='--')
    ax.grid(True)
    plt.xlim((-0.5,len(fr_sentence)-0.5))
    plt.xticks(range(len(fr_sentence)), fr_sentence, rotation=330)
    plt.ylim((-0.5,len(en_sentence)-0.5))
    plt.yticks(range(len(en_sentence)), en_sentence)


##################################################
###################### MAIN ######################
##################################################

def main():
    #### OUTPUT PARAMETERS ####
    n_sentences = 20
    ###########################

    args = parseArguments()

    if args.m is None:
        methods = [1]
    else:
        methods = args.m
    fr_corpus, fr_dict, en_corpus, en_dict = imp.import_all(args.french_corpus, args.english_corpus)

    # Faire tourner sans arguments
    # fr_corpus, fr_dict, en_corpus, en_dict = imp.import_all("corpus_fr.txt", "corpus_en.txt")
    # methods = [1, 2, 3]

    print(fr_dict)
    print(en_dict)

    for method_index in methods:
        assert (method_index > 0 and method_index < 4), "Unsupported method index: %d" % method_index

        if (method_index == 1):
            # IBM1
            P = ibm.IBM1(fr_corpus, en_corpus, fr_dict, en_dict)
            print_P_to_csv(en_dict, fr_dict, P, "output_ibm1.csv")

            for k in range(n_sentences):
                plot_sentence_alignment(fr_corpus, en_corpus, en_dict, fr_dict,
                    P, k, 1)

            plt.show()

        if (method_index == 2):
            # IBM2
            lamb = 1
            p_null = .1
            P = ibm.IBM2(fr_corpus, en_corpus, fr_dict, en_dict, lamb, p_null)
            print_P_to_csv(en_dict, fr_dict, P, "output_ibm2.csv")

            for k in range(n_sentences):
                plot_sentence_alignment(fr_corpus, en_corpus, en_dict, fr_dict,
                    P, k, 2, lamb, p_null)

            plt.show()

        if (method_index == 3):
            # HMM
            A, P, p_initial, gamma, ksi = hmm.EM_HMM(fr_corpus,fr_dict,en_corpus,en_dict)
            # P = np.ones((size_dictF, size_dictE)) / size_dictF

            # # A COMPLETER p_initial,A,P ?
            # gamma, ksi = expectation(fr_corpus,en_corpus,fr_dict,en_dict,A,P,p_initial)
            # alignment = []

            # for fr in range(len(fr_corpus)):
            #     a = viterbi(fr_corpus, en_corpus, fr, p_initial, fr_dict, gamma, ksi)
            #     alignment.append(a)

            # for i in range(size_dictF):
            #     for j in range(size_dictE):
            #         P[i,j] = emission_proba(fr_dict[i],en_dict[j],fr_corpus, en_corpus, fr_dict, gamma)

            print_P_to_csv(en_dict, fr_dict, P, "output_hmm.csv")

            for idx in range(len(fr_corpus)):
                # x-axis : English
                # y-axis : French
                en_sentence = re.split(' |\'', en_corpus[idx])
                fr_sentence = re.split(' |\'', fr_corpus[idx])

                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.plot(range(len(fr_sentence)),alignment,'ro',ls='--')
                ax.grid(True)
                plt.xlim((-0.5,len(fr_sentence)-0.5))
                plt.xticks(range(len(fr_sentence)), fr_sentence, rotation=330)
                plt.ylim((-0.5,len(en_sentence)-0.5))
                plt.yticks(range(len(en_sentence)), en_sentence)
            plt.show()

if __name__ == '__main__':
    main()
