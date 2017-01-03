#!/bin/bash

main="translate.py"

CORPUS_FR="data/corpus_fr_hom.txt"
CORPUS_EN="data/corpus_en_hom.txt"

method="1 2 3"

python $main $CORPUS_FR $CORPUS_EN -m $method
