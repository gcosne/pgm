#!/bin/bash

MAIN='classify.py'
DATA_TRAIN='data/EMGaussian.data'
DATA_TEST='data/EMGaussian.test'

QUESTIONS='2 4 5 6 8 9 10' #2 4 5 6 8 9 10 for a full run

python $MAIN $DATA_TRAIN $DATA_TEST -q $QUESTIONS
