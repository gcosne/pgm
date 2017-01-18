from lib import import_corpus as imp
import sys

accepted_lengths = range(4,12)

CORPUS_EN = 'data/clean.en'
CORPUS_FR = 'data/clean.fr'

output_path_en = "data/clean.en.12"
output_path_fr = "data/clean.fr.12"

f_en = open(output_path_en,'w')
f_fr = open(output_path_fr,'w')

corpus_en,_ = imp.import_corpus(CORPUS_EN)
corpus_fr,_ = imp.import_corpus(CORPUS_FR)

for k in range(len(corpus_en)):
    if len(imp.split_sentence(corpus_en[k])) in accepted_lengths:
        f_en.write('%s\n'%corpus_en[k])
        f_fr.write('%s\n'%corpus_fr[k])

f_en.close()
f_fr.close()

new_corpus_en,_ = imp.import_corpus(output_path_en)

imp.corpus_statistics(new_corpus_en)

