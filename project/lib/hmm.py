import numpy as np
import import_corpus as imp


############################
####### EXPECTATION ########
############################

#########################################################################################
########### be careful the fr_sentence and en_sentence are assumed to be splitted already
#########################################################################################

def log_proba_translate(fr_word,en_word,fr_dict,en_dict,P):
	return math.log(P[imp.hash(fr_dict,fr_word),imp.hash(en_dict,en_word)])
	# P is a sparse matrix, but in this method, only non-zero cells should have a contribution

def alpha_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,previous_alpha):
# need to keep track of j, index in the french sentence chain
# A is the transition matrix and it features 3 dimensions. The first dimension is the length of the English word
	I = len(en_sentence)
	result = np.zeros(I)
	for i in range(I):
		tmp_vector = np.log(A[I,i,:]) + previous_alpha #log of previous alpha
		result[i] = log_sum(tmp_vector)
		result[i] += log_proba_translate(fr_sentence[j],en_sentence[i],fr_dict,en_dict,update_P)
	return result

def beta_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,next_beta):
# need to keep track of j, index in the french sentence chain
# A is the transition matrix and it features 3 dimensions. The first dimension is the length of the English word
	I = len(en_sentence)
	result = np.zeros(I)
	for i in range(I):
		tmp_vector = np.log(A[I,:,i]) + next_beta
		for j in range(len(next_beta)):
			tmp_vector[j] += log_proba_translate(fr_sentence[j],en_sentence[i],fr_dict,en_dict,P)
		result[i] = log_sum(tmp_vector)
	return result

def log_sum(log_values):
	max_val = np.max(log_values)
	result = max_val + math.log(np.sum(np.exp(log_values-max_val)))
	return result

def compute_all_alpha(fr_sentence,en_sentence,fr_dict,en_dict,A,P,p_initial):
	# compute all logarithms of alpha messages
	# p_intial is the initial probability distribution over hidden states
	I = len(en_sentence)
	J = len(fr_sentence)
	alphas = np.zeros([I,J])
	# initialization
	for i in range(I):
		alphas[i,0] = math.log(p_initial[i]) + log_proba_translate(fr_sentence[0],en_sentence[i],fr_dict,en_dict,P)
	# log recursion
	for j in range(1,J):
		alphas[:,j] = alpha_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,alphas[:,j-1])
	#alphas = np.exp(alphas)
	return alphas

def compute_all_beta(fr_sentence,en_sentence,fr_dict,en_dict,A,P):
	# compute all logarithms of beta messages
	I = len(en_sentence)
	betas = np.zeros([I,len(fr_sentence)])
	# initialization
	betas[:,-1] = 0
	# log recursion
	for j in range(J-2,-1,-1):
		betas[:,j] = beta_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,betas[:,j+1])
	#betas = np.exp(betas)
	return betas

def cond_proba_unary(log_alphas, log_betas):
	probas = log_alphas + log_betas
	for t in range(probas.shape[1]):
		probas[:,t] = probas[:,t] - log_sum(probas[:,t])
	probas = np.exp(probas)
	return probas

def cond_proba_binary(log_alphas,log_betas, A, P, fr_sentence, en_sentence, fr_dict, en_dict):
	I = len(en_sentence)
	probas = np.zeros([log_alphas.shape[1]-1,A[I].shape[0],A[I].shape[1]])
	for t in range(probas.shape[0]):
		for i in range(probas.shape[1]):
			for j in range(probas.shape[2]):
				probas[t,i,j] += log_alphas[j,t] + log_betas[i,t+1]
				probas[t,i,j] += math.log(A[I,i,j])
				probas[t,i,j] += log_proba_translate(fr_sentence[i],en_sentence[j],fr_dict,en_dict,P)
				probas[t,i,j] -= log_sum(log_alphas[:,t] + log_betas[:,t])
	probas = np.exp(probas)
	return probas

def update_gamma_ksi(fr_sentence, en_sentence, fr_dict, en_dict,A,P,p_initial):
    alphas = compute_all_alpha(fr_sentence,en_sentence,fr_dict,en_dict,A,P,p_initial)
    betas = compute_all_beta(fr_sentence,en_sentence,fr_dict,en_dict,A,P)
    gamma = cond_proba_unary(alphas,betas)
    ksi = cond_proba_binary(log_alphas,log_betas, A, P, fr_sentence, en_sentence, fr_dict, en_dict)
    return gamma.T, ksi

## update all alignment probabilities
## and compile them in arrays gamma and ksi
def expectation(fr_corpus,en_corpus,fr_dict,en_dict,A,P,p_initial):
	n_sentences = len(fr_corpus)
	gamma = np.zeros(n_sentences,dtype=object)
	ksi = np.zeros(n_sentences,dtype=object)

	for t in range(n_sentences):
		fr_sentence = imp.split_sentence(fr_corpus[t])
		en_sentence = imp.split_sentence(en_corpus[t])

		gam, ks = update_gamma_ksi(fr_sentence, en_sentence, fr_dict, en_dict,A,P,p_initial)
		gamma[t] = gam
		ksi[t] = ks

	return gamma, ksi

#### maximization step ####

# c(d=i-ip)
def count(i, ip, fr_corpus, en_corpus, idx_phrase, ksi):
    d = i - ip
    fr_sentence = fr_corpus[idx_phrase]
    en_sentence = en_corpus[idx_phrase]
    I = len(en_sentence)
    J = len(fr_sentence)
    for j in range(0,J-1):
        for k in range(I):
            count += ksi[idx_phrase,j,k+d,k]
    return count

# posterior count of transitions with jump width i-ip c(ip,i,I)
def count_alignment(ip, i, I, fr_corpus, en_corpus, ksi):
    for fr in range(len(fr_corpus)):
        fr_sentence = fr_corpus[fr]
        en_sentence = en_corpus[fr] #fr is the index of the corresponding sentence in the english corpus
        I = len(en_sentence)
        count += count(i, ip, fr_corpus, en_corpus, fr, ksi) * (len(en_sentence) == I)
        return count

# posterior count c(f,e)
def count_emission(f, e, fr_corpus, en_corpus, gamma):
    for fr in range(len(fr_corpus)):
        fr_sentence = fr_corpus[fr]
        en_sentence = en_corpus[fr] #fr is the index of the corresponding sentence in the english corpus
        fr_words = imp.split_sentence(fr_sentence)
        en_words = imp.split_sentence(en_sentence)
        J = len(fr_sentence)
        I = len(en_sentence)
        for i in range(I):
            for j in range(J):
                count += gamma(fr,j,i) * (f==fr_words[i]) * (e==en_words[j])
        return count

# p(i|ip,I)
def update_alignment_proba(i, ip, I, fr_corpus, en_corpus, ksi):
    alignment_proba = count_alignment(ip, i, I, fr_corpus, en_corpus, ksi) / np.sum(count_alignment(ip, ipp, I, fr_corpus, en_corpus, ksi) for ipp in range(I))
    return alignment_proba

# p(f|e)
def update_emission_proba(f, e, fr_corpus, en_corpus, fr_dict, gamma):
    emission_proba = count_emission(f, e, fr_corpus, en_corpus, gamma) / np.sum(count_emission(k, e, fr_corpus, en_corpus, gamma) for k in fr_dict)
    return emission_proba

# p(i)
def update_pi0(gamma):
    pi0_update = gamma[0,:,:] / np.sum(gamma[0,:,:])
    return pi0_update

"""
# the A array will have the sup of lengths of english sentences as a second dimension
def output_counts(fr_corpus,en_corpus,fr_dict,en_dict,A_array,P,p_initial_array):
# steps: sentence per sentence, update the counts in the count array, with is of same size as the P array
# no need to do gather array of different sizes (which all correspond to the lengths of the different sentences)
# assumes that it is better to have rather homogenous number of words in each sentence of the corpus

	counts = np.zeros(P.shape)
	counts_transitions = np.zeros([len(fr_corpus),imp.max_len_sentence(en_corpus)])

	for fr in range(len(fr_corpus)):
		fr_sentence = fr_corpus[fr]
		en_sentence = en_corpus[fr] #fr is the index of the corresponding sentence in the english corpus
		fr_words = imp.split_sentence(fr_sentence)
		en_words = imp.split_sentence(en_sentence)

		A = A_array[fr,0:len(fr_words)]
		p_initial = p_initial_array[fr,0:len(en_words)]

		log_alphas = compute_all_alpha(fr_words,en_words,fr_dict,en_dict,A,P,p_initial)
		log_betas = compute_all_beta(fr_words,en_words,fr_dict,en_dict,A,P)

		proba_unary = cond_proba_unary(log_alphas,log_betas)
		proba_binary = cond_proba_binary(log_alphas,log_betas,A,P,fr_words,en_words,fr_dict,en_dict)

		# go through the sentence words and aggregate the counts
		for fr_idx in range(len(fr_words)):
			for en_idx in range(len(en_words)):
				fr_word = fr_words[fr_idx]
				en_word = fr_words[en_idx]
				counts[imp.hash(fr_dict,fr_word),imp.hash(en_dict,en_word)] += proba_unary[en_idx,fr_idx]
#TODO update the counts_transitions matrix

		# for the counts_transitions matrix
		for d in range(len(fr_words)):
			######## j'en suis a la


	return counts

def update_P(P,counts):
# P is passed as a reference and is directly modified within its cells
	P = counts / np.sum(counts, axis=0)
"""
