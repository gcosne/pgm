# -*- coding: utf-8 -*-
import re
import codecs
import string

def split_sentence(sentence):
    sentence_split = re.split(' |\'', sentence)
    return sentence_split

def transform_clean(length=None):
    fr_input = codecs.open('test.fr', 'r', 'cp1252')
    en_input = codecs.open('test.en', 'r', 'cp1252')
    fr_output = open('clean.fr.10', 'w')
    en_output = open('clean.en.10', 'w')

    if length is None:
        for line in fr_input:
            translator = str.maketrans({key: None for key in string.punctuation})
            l = line.translate(translator)
            l = l.replace("«","")
            l = l.replace("»","")
            l = l.lower()
            l = re.sub(' +', ' ', l)
            l = l.rstrip()
            l = l + '\n'
            fr_output.write(l)

        for line in en_input:
            translator = str.maketrans({key: None for key in string.punctuation})
            l = line.translate(translator)
            l = l.lower()
            l = re.sub(' +', ' ', l)
            l = l.rstrip()
            l = l + '\n'
            en_output.write(l)

        return

    indices_to_keep = []
    counter = 0
    for line in en_input:
        translator = str.maketrans({key: None for key in string.punctuation})
        l = line.translate(translator)
        l = l.lower()
        l = re.sub(' +', ' ', l)
        l = l.rstrip()
        if (len(split_sentence(l))==length):
            l = l + '\n'
            en_output.write(l)
            indices_to_keep.append(counter)
        counter += 1

    counter = 0
    for line in fr_input:
        if counter in indices_to_keep:
            translator = str.maketrans({key: None for key in string.punctuation})
            l = line.translate(translator)
            l = l.replace("«","")
            l = l.replace("»","")
            l = l.lower()
            l = re.sub(' +', ' ', l)
            l = l.rstrip()
            l = l + '\n'
            fr_output.write(l)


transform_clean(10)
