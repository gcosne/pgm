#!/bin/bash

main="translate.py"

#CORPUS_FR="data/corpus_fr.txt"
#CORPUS_EN="data/corpus_en.txt"

CORPUS_EN="data/clean.fr.12"
CORPUS_FR="data/clean.en.12"

method="2 3"

python $main $CORPUS_FR $CORPUS_EN -m $method
