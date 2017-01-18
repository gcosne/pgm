import numpy as np
import sys
import re
import os


def import_corpus(path_to_corpus, n_sentences=None):
    input_file = open(path_to_corpus, 'r')

    corpus = np.array([])
    unique_words = np.array([])

    counter = 0
    for line in input_file:
        counter += 1
        corpus = np.append(corpus, line.split('\n')[0])
        words = re.split('\'| ', line.split('\n')[0])
        unique_words = np.unique(np.append(unique_words, words))
        if (not (n_sentences is None)) and (counter == n_sentences):
            break

    return corpus, unique_words

# returns corpus statistics
def corpus_statistics(corpus):
    n_senteces = len(corpus)
    n_words = {} # dictionnary (n_words,n_corresponding_sentences) 
    counter = 0
    for sentence in corpus:
        sys.stdout.flush()
        n = len(split_sentence(sentence))
        if n in n_words:
            n_words[n] += 1
        else:
            n_words[n] = 1
    print "Corpus statistics:"
    for key in n_words:
        print "%d words -> %d sentences" % (key,n_words[key])
    return n_words


def import_all(path_to_french, path_to_english, n_sentences):
    fr_corpus, fr_dict = import_corpus(path_to_french, n_sentences)
    en_corpus, en_dict = import_corpus(path_to_english, n_sentences)

    print("-------")
    print("IMPORTED FRENCH : %d sentences, %d corrsponding words" % (fr_corpus.size,fr_dict.size))
    print("IMPORTED ENGLISH : %d sentences, %d corrsponding words" % (en_corpus.size,en_dict.size))
    print("-------")

    return fr_corpus, fr_dict, en_corpus, en_dict


def hash(dictionary, word):
    matches = np.where(dictionary == word)
    result = 0
    if len(matches) > 0:
        result = matches[0]
    else:
        result = -1 # word not found
    assert result > -1, "in hash: word not found -> " + word
    return result


def split_sentence(sentence):
    sentence_split = re.split(' |\'', sentence)
    return sentence_split
