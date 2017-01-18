import numpy as np
import import_corpus as imp
import math
import sys

eps = 0.0000001
############################
####### EXPECTATION ########
############################

##########################################################################################
###### be careful the fr_sentence and en_sentence are assumed to be splitted already #####
##########################################################################################

def log_sum(log_values):
    max_val = np.max(log_values)
    result = max_val + math.log(np.sum(np.exp(log_values-max_val)))
    return result

def log_proba_translate(fr_word,en_word,fr_dict,en_dict,P):
    return math.log(eps + P[imp.hash(fr_dict,fr_word),imp.hash(en_dict,en_word)])
    # P is a sparse matrix, but in this method, only non-zero cells should have a contribution

def alpha_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,previous_alpha):
# need to keep track of j, index in the french sentence chain
# A is the transition matrix and it features 3 dimensions. The first dimension is the length of the English word
    I = len(en_sentence)
    idx = 0
    # find the appropriate index in A
    for i in range(len(A)):
        if (A[i].shape[0] == I):
            break
        idx = i + 1
    assert idx<len(A), "index not found in A"

    result = np.zeros(I)
    for i in range(I):
        tmp_vector = np.log(A[idx][i,:]) + previous_alpha #log of previous alpha
        result[i] = log_sum(tmp_vector)
        result[i] += log_proba_translate(fr_sentence[j],en_sentence[i],fr_dict,en_dict,P)
    return result

def beta_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,next_beta):
# need to keep track of j, index in the french sentence chain
# A is the transition matrix and it features 3 dimensions. The first dimension is the length of the English word
    I = len(en_sentence)
    idx = 0
    # find the appropriate index in A
    for i in range(len(A)):
        if (A[i].shape[0] == I):
            break
        idx = i + 1
    assert idx<len(A), "index not found in A"

    result = np.zeros(I)
    for ip in range(I):
        tmp_vector = np.log(A[idx][:,ip]) + next_beta
        # for j in range(len(next_beta)):
        for i in range(len(next_beta)):
            tmp_vector[i] += \
            log_proba_translate(fr_sentence[j],\
                en_sentence[i],\
                fr_dict,en_dict,P)
        result[i] = log_sum(tmp_vector)
    return result


def compute_all_alpha(fr_sentence,en_sentence,fr_dict,en_dict,A,P,p_initial):
    # compute all logarithms of alpha messages
    # p_intial is the initial probability distribution over hidden states
    I = len(en_sentence)
    J = len(fr_sentence)
    alphas = np.zeros([I,J])
    # initialization
    for i in range(I):
        alphas[i,0] = np.log(eps + p_initial[i] / np.sum(p_initial[:I])) \
        + log_proba_translate(fr_sentence[0],en_sentence[i],fr_dict,en_dict,P)
    # log recursion
    for j in range(1,J):
        alphas[:,j] = alpha_rec_log(j,fr_sentence,en_sentence,fr_dict,en_dict,A,P,alphas[:,j-1])
    #alphas = np.exp(alphas)
    return alphas

def compute_all_beta(fr_sentence,en_sentence,fr_dict,en_dict,A,P):
    # compute all logarithms of beta messages
    I = len(en_sentence)
    J = len(fr_sentence)
    betas = np.zeros([I,J])
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
        #probas[:,t] = probas[:,t] - log_sum(log_alphas[:,-1])
    probas = np.exp(probas)
    return probas

def cond_proba_binary(log_alphas,log_betas, A, P, fr_sentence, en_sentence, fr_dict, en_dict):
    I = len(en_sentence)
    J = len(fr_sentence)

    idx = 0
    # find the appropriate index in A
    for i in range(len(A)):
        if (A[i].shape[0] == I):
            break
        idx = i + 1
    assert idx<len(A), "index not found in A"

    # probas = np.zeros([log_alphas.shape[1]-1,A[idx].shape[0],A[idx].shape[1]])
    # ksi = np.zeros([log_alphas.shape[1]-1,I,I])
    ksi = np.zeros([J-1,I,I])
    for j in range(ksi.shape[0]):
        for k in range(ksi.shape[1]):
            for l in range(ksi.shape[2]):
                ksi[j,k,l] += log_alphas[l,j] + log_betas[k,j+1]
                ksi[j,k,l] += math.log(eps + A[idx][k,l])
                ksi[j,k,l] += log_proba_translate(fr_sentence[j+1],en_sentence[k],fr_dict,en_dict,P)
                ksi[j,k,l] -= log_sum(log_alphas[:,j] + log_betas[:,j])
                #ksi[j,k,l] -= log_sum(log_alphas[:,-1])
    ksi = np.exp(ksi)
    return ksi

def update_gamma_ksi(fr_sentence, en_sentence, fr_dict, en_dict,A,P,p_initial):
    log_alphas = compute_all_alpha(fr_sentence,en_sentence,fr_dict,en_dict,A,P,p_initial)
    log_betas = compute_all_beta(fr_sentence,en_sentence,fr_dict,en_dict,A,P)

    gamma = cond_proba_unary(log_alphas,log_betas)
    ksi = cond_proba_binary(log_alphas,log_betas, A, P, fr_sentence, en_sentence, fr_dict, en_dict)

    return gamma.T, ksi

## update all alignment probabilities
## and compile them in arrays gamma and ksi
def expectation(fr_corpus,en_corpus,fr_dict,en_dict,A,P,p_initial):
    n_sentences = len(fr_corpus)

    gamma = np.zeros(n_sentences,dtype=object)
    ksi = np.zeros(n_sentences,dtype=object)

    for t in range(n_sentences):
        print "\rEstep: sentence %d/%d" % (t,n_sentences)
        sys.stdout.flush()
        fr_sentence = imp.split_sentence(fr_corpus[t])
        en_sentence = imp.split_sentence(en_corpus[t])

        gam, ks = update_gamma_ksi(fr_sentence, en_sentence, fr_dict, en_dict,A,P,p_initial)
        gamma[t] = gam
        ksi[t] = ks

    print '\n'
    sys.stdout.flush()
    return gamma, ksi

############################
####### MAXIMIZATION #######
############################

# c(d=i-ip)
def count(d, fr_corpus, en_corpus, idx_phrase, ksi):
    c = 0
    fr_sentence = fr_corpus[idx_phrase]
    en_sentence = en_corpus[idx_phrase]
    fr_words = imp.split_sentence(fr_sentence)
    en_words = imp.split_sentence(en_sentence)
    I = len(en_words)
    J = len(fr_words)
    for j in range(0,J-1):
        for k in range(I):
            if (k+d)<ksi[idx_phrase].shape[1] and (k+d)>-1:
                c += ksi[idx_phrase][j,k+d,k]
    return c

# posterior count of transitions with jump width i-ip c(ip,i,I)
def count_alignment(ip, i, I, fr_corpus, en_corpus, ksi):
    c = 0
    for fr in range(len(fr_corpus)):
        fr_sentence = fr_corpus[fr]
        en_sentence = en_corpus[fr] #fr is the index of the corresponding sentence in the english corpus
        fr_words = imp.split_sentence(fr_sentence)
        en_words = imp.split_sentence(en_sentence)
        d = i - ip # i is the alignment of j+1, ip alignment of j
        if (len(en_words) == I):
            c += count(d, fr_corpus, en_corpus, fr, ksi)
    return c

def alignment_array(I, fr_corpus, en_corpus, ksi):
    result = np.zeros((I,I))
    for i in range(I):
        for ip in range(I):
            result[ip,i] = count_alignment(ip, i, I, fr_corpus, en_corpus, ksi)

    return result

def alignment_transition_2(i,ip,alignment_array):
    return alignment_array[ip,i] / np.sum(alignment_array[ip,:])

# p(i|ip,I)
def alignment_transition(i, ip, I, fr_corpus, en_corpus, ksi):
    alignment_proba = count_alignment(ip, i, I, fr_corpus, en_corpus, ksi) / np.sum(count_alignment(ip, ipp, I, fr_corpus, en_corpus, ksi) for ipp in range(I))
    return alignment_proba

# p(f|e)
# def emission_proba(f, e, fr_corpus, en_corpus, fr_dict, gamma):
#     emission_proba = count_emission(f, e, fr_corpus, en_corpus, gamma) / np.sum(count_emission(k, e, fr_corpus, en_corpus, gamma) for k in fr_dict)
#     return emission_proba


# # p(idx_phrase,i)
# def p_initial(gamma):
#     p_initial = gamma[:,0,:] / np.sum(gamma[:,0,:]) # doute
#     return p_initial

def p_init(gamma,max_I):
    p_init = np.zeros(int(max_I))
    for gam in gamma:
        I = gam.shape[1]
        p_init[:I] = p_init[:I] + gam[0,:]
    p_init = p_init / np.sum(p_init)
    return p_init

# posterior count c(f,e)
def count_emission(f, e, fr_corpus, en_corpus, gamma):
    count = 0
    for fr in range(len(fr_corpus)):

        fr_sentence = fr_corpus[fr]
        en_sentence = en_corpus[fr] #fr is the index of the corresponding sentence in the english corpus

        fr_words = imp.split_sentence(fr_sentence)
        en_words = imp.split_sentence(en_sentence)

        J = len(fr_words)
        I = len(en_words)

        for i in range(I):
            for j in range(J):
                if (f==fr_words[j]) and (e==en_words[i]):
                    count += gamma[fr][j,i]
    return count

def count_emissions(fr_dict, en_dict, fr_corpus, en_corpus, gamma):
    c_emissions = np.zeros([len(fr_dict),len(en_dict)])

    for k in range(len(fr_corpus)):
        fr_words = imp.split_sentence(fr_corpus[k])
        en_words = imp.split_sentence(en_corpus[k])

        for j in range(len(fr_words)):
            fr_idx = imp.hash(fr_dict,fr_words[j])
            for i in range(len(en_words)):
                en_idx = imp.hash(en_dict,en_words[i])
                c_emissions[fr_idx,en_idx] += gamma[k][j,i]

    return c_emissions

# def count_emissions(fr_dict, en_dict, fr_corpus, en_corpus, gamma):
#     c_emissions = np.zeros([len(fr_dict),len(en_dict)])
#     # f_idx = 0
#     # for f in fr_dict:
#     #     e_idx = 0
#     #     for e in en_dict:
#     #         c_emissions[f_idx,e_idx] = count_emission(f, e, fr_corpus, en_corpus, gamma)
#     #         e_idx += 1
#     #     f_idx += 1
#     # return c_emissions
#     for f in range(len(fr_dict)):
#         for e in range(len(en_dict)):
#             c_emissions[f,e] = count_emission(fr_dict[f], en_dict[e], fr_corpus, en_corpus, gamma)
#     return c_emissions

def emission_proba(f,e,fr_dict,en_dict,c_emissions):
    f_idx = imp.hash(fr_dict,f)
    e_idx = imp.hash(en_dict,e)
    emission_proba = c_emissions[f_idx,e_idx] / np.sum(c_emissions[:,e_idx])
    return emission_proba

# update the emission probility matrix
# def update_P(P,fr_corpus, en_corpus, fr_dict, en_dict, gamma):
#     for f in range(P.shape[0]):
#         for e in range(P.shape[1]):
#             P[f,e] = emission_proba(fr_dict[f], en_dict[e], fr_corpus, en_corpus, fr_dict, gamma)



def update_P(P,fr_corpus, en_corpus, fr_dict, en_dict, gamma):
    c_emissions = count_emissions(fr_dict, en_dict, fr_corpus, en_corpus, gamma)
    for f in range(P.shape[0]):
        for e in range(P.shape[1]):
            P[f,e] = emission_proba(fr_dict[f], en_dict[e], fr_dict,en_dict,c_emissions)

# update the 3-dimensionnal transition matrix
def update_A(A,fr_corpus, en_corpus, ksi):
    for I in range(A.shape[0]):
        array = alignment_array(len(A[I]),fr_corpus,en_corpus,ksi)
        for i in range(len(A[I])):
            for ip in range(len(A[I])): #A[I] should be square
                A[I][i,ip] = alignment_transition_2(i, ip, array)

def EM_HMM(fr_corpus,fr_dict,en_corpus,en_dict,P2):
    CONVERGENCE_THR = 0.01 #not used at the moment
    print "Computing HMM"
    ###################
    ##### INIT ########
    ###################

    # get the maximum English sentence length
    # and the number of unique different lengths
    Is = np.zeros(len(en_corpus))
    for e in range(len(en_corpus)):
        Is[e] = len(imp.split_sentence(en_corpus[e]))
    # lengths of English sentences
    lengths = np.unique(Is)
    max_I = np.max(lengths)

    # init A
    A = np.zeros(len(lengths),dtype=object)

    for i in range(len(A)):
        A[i] = (1.0/lengths[i])*np.ones([lengths[i],lengths[i]])
        # A[i] = (0.5/(lengths[i]-1))*np.ones(lengths[i])
        # A[i] = A[i] + (0.5 - (0.5/(lengths[i]-1)))*np.eye(lengths[i])

    ######################################### debug !!!!
    #print A

    # init P as uniform
    # P = np.ones((len(fr_dict), len(en_dict))) / len(fr_dict)
    P = P2

    # init P initial
    p_initial = np.ones(int(max_I)) / max_I

    # print p_initial

    ###########################
    #### FIRST EXPECTATION ####
    ###########################
    gamma, ksi = expectation(fr_corpus,en_corpus,fr_dict,en_dict,A,P,p_initial)

    ###########################
    ### WHILE NOT CONVERGED ###
    ###########################
    counter = 0
    max_iter = 3
    while counter < max_iter:
        counter += 1
        print "--------------------------------------------"
        print "EM iteration : %d\n" % counter,
        sys.stdout.flush()

        # Maximisation
        print 'M step:'
        update_A(A,fr_corpus, en_corpus, ksi)
        update_P(P,fr_corpus, en_corpus, fr_dict, en_dict, gamma)
        p_initial = p_init(gamma,max_I)


        # Expectation
        gamma, ksi = expectation(fr_corpus,en_corpus,fr_dict,en_dict,A,P,p_initial)

    ######################################### debug !!!!
    #print A
    # print A
    #print p_initial

    # print fr_corpus[11]
    # print en_corpus[11]
    # print gamma[11].shape
    # print ksi[11].shape

    print A
    print P
    print p_initial
    return A, P, p_initial, gamma, ksi

############################
######### DECODING #########
############################

def viterbi2(fr_corpus, en_corpus, fr_dict, en_dict, idx_phrase, p_initial, P, A):
    fr_sentence = fr_corpus[idx_phrase]
    en_sentence = en_corpus[idx_phrase]
    fr_words = imp.split_sentence(fr_sentence)
    en_words = imp.split_sentence(en_sentence)
    I = len(en_words)
    J = len(fr_words)

    idx = 0
    # find the appropriate index in A
    for i in range(len(A)):
        if (A[i].shape[0] == I):
            break
        idx = i + 1
    assert idx<len(A), "index not found in A"

    # init result matrices
    state = np.zeros((I,J))
    a = np.zeros((J))
    log_v = np.zeros((I, J))
    log_p = np.log(p_initial)

    # Base case
    for i in range(I):
        # V(i,0) = p(i) p(f_0|e_i)
        log_v[i,0] = np.log(eps + p_initial[i]) + log_proba_translate(fr_words[0],en_words[i],fr_dict,en_dict,P)

    # Recursion
    for j in range(1,J):
        for i in range(I):
            # V(i,j) = max_ip p(i|ip,I) p(fj|ei) V(ip, j-1)
            (log_v[i,j], alignment) = max((log_proba_translate(fr_words[j],en_words[i],fr_dict,en_dict,P) + log_v[ip, j-1], ip) for ip in range(I))
            state[i,j] = alignment #a_j-1

    # Compute the Viterbi decoding
    i_best = np.argmax(log_v[:,J-1])
    a[J-1] = i_best
    for j in range(J-1,0,-1):
        a[j-1] = state[int(a[j]),j]
    return a # list of size J

"""
def viterbi(fr_corpus, en_corpus, idx_phrase, p_initial, fr_dict, en_dict, gamma, ksi, c_emissions):
    fr_sentence = fr_corpus[idx_phrase]
    en_sentence = en_corpus[idx_phrase]
    fr_words = imp.split_sentence(fr_sentence)
    en_words = imp.split_sentence(en_sentence)
    I = len(en_words)
    J = len(fr_words)

    state = np.zeros((I,J))
    a = np.zeros((J))
    log_v = np.zeros((I, J))

    # Base case
    for i in range(I):
        # V(i,0) = p(i) p(f_0|e_i)
        log_v[i,0] = np.log(eps + p_initial[i]) + np.log(eps + emission_proba(fr_words[0],en_words[i],fr_dict,en_dict,c_emissions))

    # Recursion
    for j in range(1,J):
        for i in range(I):
            # V(i,j) = max_ip p(i|ip,I) p(fj|ei) V(ip, j-1)
            (log_v[i,j], alignment) = max((np.log(eps + emission_proba(fr_words[j],en_words[i],fr_dict,en_dict,c_emissions)) + np.log(alignment_transition(i, ip, I, fr_corpus, en_corpus, ksi)) + log_v[ip, j-1], ip) for ip in range(I))
            state[i,j] = alignment #a_j-1

    # Compute the Viterbi decoding
    i_best = np.argmax(log_v[:,J-1])
    a[J-1] = i_best
    for j in range(J-1,0,-1):
        a[j-1] = state[int(a[j]),j]

    return a # list of size J
"""
