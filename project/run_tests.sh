#!/bin/bash

main="translate.py"

CORPUS_FR="data/clean.fr"
CORPUS_EN="data/clean.en"

method="3"

python $main $CORPUS_FR $CORPUS_EN -m $method
