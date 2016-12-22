##########################################
#### Probabilistic Graphical Models ######
################ Project #################
##########################################

import sys
import itertools
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
import operator
import re

### STATIC
CONVERGENCE_CRITERIA = 1e-10
N_ITER_MAX = 200
##########

def parseArguments():
	parser = argparse.ArgumentParser(description="Learn a IBM1 model")
	
	parser.add_argument("french_corpus",
		help="path_to_french_corpus")

	parser.add_argument("english_corpus",
		help="path_to_english_corpus")

	parser.add_argument('-m', '-method', nargs='*', 
		type=int, help="(optional) Specify the models to learn: 1/IBM1 (defaults to IBM1)")

	args = parser.parse_args()
	return args

#########################
### GENERAL FUNCTIONS ###
#########################
def import_corpus(path_to_corpus):
	input_file  = open(path_to_corpus,'r')

	corpus = np.array([])
	unique_words = np.array([])

	for line in input_file:
		corpus = np.append(corpus,line.split('\n')[0])
		words = re.split('\'| ',line.split('\n')[0])
		unique_words = np.unique(np.append(unique_words,words))

	return corpus, unique_words

def import_all(path_to_french, path_to_english):
	fr_corpus, fr_dict = import_corpus(path_to_french)
	en_corpus, en_dict = import_corpus(path_to_english)

	print "-------"
	print "IMPORTED FRENCH : %d sentences, %d corrsponding words" % (fr_corpus.size,fr_dict.size)
	print "IMPORTED ENGLISH : %d sentences, %d corrsponding words" % (en_corpus.size,en_dict.size)
	print "-------"

	return fr_corpus, fr_dict, en_corpus, en_dict	

def hash(dictionary,word):
	matches = np.where(dictionary==word)
	result = 0
	if len(matches) > 0:
		result = matches[0]
	else:
		result = -1 # word not found
	assert result>-1, "in hash: word not found -> " + word
	return result

def display_results(fr_dict,en_dict,P,target):
	assert len(fr_dict) == np.size(P)[0], "dimensions mismatch"
	assert len(en_dict) == np.size(P)[1], "dimensions mismatch"
	
	print P

def IBM1(F,E,D_F,D_E):
	# F is the french corpus
	# E is the english corpus
	# D_F is the list of unique french words
	# D_E is the list of unique english words

	size_dictF = len(D_F)
	size_dictE = len(D_E)
	N = len(F)

	Imax=0
	phrase_size_F = np.zeros((len(F),1))
	cpt = 0
	for phrase in F:
		phrase_split = re.split(' |\'',phrase)
		l = len(phrase_split)
		phrase_size_F[cpt] = l
		Imax = max(Imax,l)
		cpt += 1

	Jmax=0
	phrase_size_E = np.zeros((len(E),1))
	cpt = 0
	for phrase in E:
		phrase_split = re.split(' |\'',phrase)
		l = len(phrase_split)
		phrase_size_E[cpt] = l
		Jmax = max(Jmax,l)
		cpt += 1


	P = np.ones((size_dictF,size_dictE)) / size_dictF

	counter = 0

	while(counter<N_ITER_MAX):
		counter += 1
		P_tmp = P
		C_align = np.zeros((size_dictF,size_dictE))
		C_word = np.zeros((size_dictE))

		for n in range(N):
			for i in range(phrase_size_F[n]):
				Z = 0
				for j in range(phrase_size_E[n]):
					indF = hash(D_F,re.split(' |\'',F[n])[i])
					indE = hash(D_E,re.split(' |\'',E[n])[j])
					Z += P[indF,indE]
				for j in range(phrase_size_E[n]):
					indF = hash(D_F,re.split(' |\'',F[n])[i])
					indE = hash(D_E,re.split(' |\'',E[n])[j])
					c = P[indF,indE] / Z
					C_align[indF,indE] += c
					C_word[indE] += c

		for i in range(P.shape[0]):
			for j in range(P.shape[1]):
				P[i,j] = C_align[i,j] / C_word[j]

		#if (np.linalg.norm(P-P_tmp)<CONVERGENCE_CRITERIA
		#):
		#	break

	if counter==N_ITER_MAX:
		print "Warning, in IBM1, reached maximum number of iterations"

	return P


## OUTPUT DISPLAY METHODS ##

def print_P_to_csv(en_dict, fr_dict, P, output_path = 'output.csv'):
	f = open(output_path,'w')
	# header
	f.write(',')
	for en_word in en_dict:
		f.write(en_word)
		f.write(',')
	# remove last comma
	f.seek(-1, os.SEEK_END)
	f.truncate()
	f.write('\n')

	# big loop
	for i in range(P.shape[0]):
		f.write(fr_dict[i])
		f.write(',')
		for j in range(P.shape[1]):
			f.write("%.4f"%P[i,j])
			if j<range(P.shape[1])[-1]:
				f.write(',')
			else:
				f.write('\n')

	# remove last \n
	f.seek(-1, os.SEEK_END)
	f.truncate()
	f.close()

def print_sentence_alignment(fr_corpus,en_corpus,en_dict,fr_dict,P,idx,figure=plt):
	# idx is the index of the sentence to plot
	# x-axis : English
	# y-axis : French
	en_sentence = re.split(' |\'',en_corpus[idx])
	fr_sentence = re.split(' |\'',fr_corpus[idx])

	fr_indices = []

	for word in fr_sentence:
		fr_idx = hash(fr_dict,word)
		fr_indices.append(fr_idx)

	tmp_P = P[fr_indices[0],:]

	for k in range(1,len(fr_indices)):
		tmp_P = np.vstack((tmp_P,P[fr_indices[k],:]))

	alignment = []
	# get the most likely alignment
	for word in en_sentence:
		en_idx = hash(en_dict,word)
		alignment.append(np.argmax(tmp_P[:,en_idx]))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(range(len(en_sentence)),alignment,'ro',ls='--')
	ax.grid(True)
	plt.xlim((-0.5,len(en_sentence)-0.5))
	plt.xticks(range(len(en_sentence)), en_sentence, rotation=330)
	plt.ylim((-0.5,len(fr_sentence)-0.5))
	plt.yticks(range(len(fr_sentence)), fr_sentence)


def print_sentence_alignment_2(fr_corpus,en_corpus,en_dict,fr_dict,P,idx,figure=plt):
	# idx is the index of the sentence to plot
	# x-axis : French
	# y-axis : English
	en_sentence = re.split(' |\'',en_corpus[idx])
	fr_sentence = re.split(' |\'',fr_corpus[idx])

	en_indices = []

	for word in en_sentence:
		en_idx = hash(fr_dict,word)
		en_indices.append(en_idx)

	tmp_P = P[:,en_indices[0]]

	for k in range(1,len(en_indices)):
		tmp_P = np.hstack((tmp_P,P[en_indices[k],:]))

	alignment = []
	# get the most likely alignment
	for word in fr_sentence:
		fr_idx = hash(fr_dict,word)
		alignment.append(np.argmax(tmp_P[fr_idx,:]))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(range(len(fr_sentence)),alignment,'ro',ls='--')
	ax.grid(True)
	plt.xlim((-0.5,len(en_sentence)-0.5))
	plt.xticks(range(len(fr_sentence)), fr_sentence, rotation=330)
	plt.ylim((-0.5,len(en_sentence)-0.5))
	plt.yticks(range(len(en_sentence)), en_sentence)


############################


##################################################
###################### MAIN ######################
##################################################

def main():
	#### OUTPUT PARAMETERS ####
	n_sentences = 4
	###########################
	
	args = parseArguments()

	if args.m is None:
		methods = [1]
	else:
		methods = args.m

	fr_corpus, fr_dict, en_corpus, en_dict = import_all(args.french_corpus, args.english_corpus)
	
	print fr_dict
	print en_dict

	for method_index in methods:
		assert (method_index > 0 and method_index < 2), "Unsupported method index: %d" % method_index

		if (method_index == 1):
			# IBM1
			P = IBM1(fr_corpus,en_corpus,fr_dict,en_dict)
			print_P_to_csv(en_dict, fr_dict, P, "output_ibm1.csv")
			
			for k in range(n_sentences):
				print_sentence_alignment2(fr_corpus,en_corpus,en_dict,fr_dict,P,int(math.floor(random.random()*len(fr_corpus))))
	
			plt.show()
if __name__ == '__main__':
	main()
