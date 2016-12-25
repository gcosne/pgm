#!/bin/bash

main="translate.py"

CORPUS_FR="corpus_fr.txt"
CORPUS_EN="corpus_en.txt"

method="2"

python $main $CORPUS_FR $CORPUS_EN -m $method
