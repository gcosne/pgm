import re
import codecs
import string


fr_input = codecs.open('test.fr', 'r', 'cp1252')
en_input = codecs.open('test.en', 'r', 'cp1252')
fr_output = open('clean.fr', 'w')
en_output = open('clean.en', 'w')

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
