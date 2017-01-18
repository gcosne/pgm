#!/bin/bash

main="translate.py"

CORPUS_FR="data/clean.fr.12"
CORPUS_EN="data/clean.en.12"

method="3"

python $main $CORPUS_FR $CORPUS_EN -m $method
