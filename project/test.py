from lib import import_corpus as imp

CORPUS_EN = 'data/clean.en'
CORPUS_FR = 'data/clean.fr'

corpus_en,_ = imp.import_corpus(CORPUS_EN)
corpus_fr,_ = imp.import_corpus(CORPUS_FR)

imp.corpus_statistics(corpus_en)
