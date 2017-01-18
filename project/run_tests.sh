#!/bin/bash

main="translate.py"

CORPUS_FR="data/corpus_fr.txt"
CORPUS_EN="data/corpus_en.txt"

method="3"

python $main $CORPUS_FR $CORPUS_EN -m $method
