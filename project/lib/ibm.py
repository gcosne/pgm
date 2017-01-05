import numpy as np
import re
from . import import_corpus as imp
import math

### STATIC
CONVERGENCE_CRITERIA = 1e-2
N_ITER_MAX = 5
##########

###################
###### IBM 1 ######
###################


def IBM1(F, E, D_F, D_E):
    # F is the french corpus
    # E is the english corpus
    # D_F is the list of unique french words
    # D_E is the list of unique english words

    size_dictF = len(D_F)
    size_dictE = len(D_E)
    N = len(F)

    Imax = 0
    phrase_size_F = np.zeros(len(F))
    cpt = 0
    for phrase in F:
        phrase_split = re.split(' |\'', phrase)
        l = len(phrase_split)
        phrase_size_F[cpt] = l
        Imax = max(Imax, l)
        cpt += 1

    Jmax = 0
    phrase_size_E = np.zeros(len(E))
    cpt = 0
    for phrase in E:
        phrase_split = re.split(' |\'', phrase)
        l = len(phrase_split)
        phrase_size_E[cpt] = l
        Jmax = max(Jmax, l)
        cpt += 1

    P = np.ones((size_dictF, size_dictE)) / size_dictF
    diffP = 1

    counter = 0

    while(counter < N_ITER_MAX):  # and diffP > CONVERGENCE_CRITERIA):
        print("IBM1: Iteration %d . Norm difference of P: %d" % (counter, diffP))
        P_prev = P
        counter += 1
        C_align = np.zeros((size_dictF, size_dictE))
        C_word = np.zeros((size_dictE))

        for n in range(N):
            for i in range(int(phrase_size_F[n])):
                Z = 0
                for j in range(int(phrase_size_E[n])):
                    indF = imp.hash(D_F, re.split(' |\'', F[n])[i])
                    indE = imp.hash(D_E, re.split(' |\'', E[n])[j])
                    Z += P[indF, indE]
                for j in range(int(phrase_size_E[n])):
                    indF = imp.hash(D_F, re.split(' |\'', F[n])[i])
                    indE = imp.hash(D_E, re.split(' |\'', E[n])[j])
                    c = P[indF, indE] / Z
                    C_align[indF, indE] += c
                    C_word[indE] += c

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P[i, j] = C_align[i, j] / C_word[j]
        diffP = abs(P - P_prev).max()

    if counter == N_ITER_MAX:
        print("Warning, in IBM1, reached maximum number of iterations")

    return P


###################
###### IBM 2 ######
###################


def h(j, i, J, I):
    return - abs(i / I - j / J)


def b(j, i, J, I, lamb, p_null):
    #  Probability that i is aligned with j
    null_word = p_null != 0
    if null_word and j == 0:
        return p_null  # The first word is the null word
    else:
        Z = 0
        for j1 in range(J):
            Z += math.exp(lamb * h(i, j1, I, J))
        offset = 0
        if null_word:
            offset = 1
        return (1 - p_null) * math.exp(lamb * h(i, j-offset, I, J)) / Z


def IBM2(F, E, D_F, D_E, lamb, p_null):
    # F is the french corpus
    # E is the english corpus
    # D_F is the list of unique french words
    # D_E is the list of unique english words
    # Lambda is a tuning parameter
    # p_null is the probability of alignment with the null word

    size_dictF = len(D_F)
    size_dictE = len(D_E)
    N = len(F)

    Imax = 0
    phrase_size_F = [0 for i in F]
    cpt = 0
    for phrase in F:
        phrase_split = re.split(' |\'', phrase)
        l = len(phrase_split)
        phrase_size_F[cpt] = l
        Imax = max(Imax, l)
        cpt += 1

    Jmax = 0
    phrase_size_E = [0 for i in E]
    cpt = 0

    offset = 0  # For null word
    phrase_split = re.split(' |\'', E[0])
    if phrase_split[0] == 'NULL':
        offset = 1

    for phrase in E:
        phrase_split = re.split(' |\'', phrase)
        l = len(phrase_split)
        phrase_size_E[cpt] = l
        Jmax = max(Jmax, l)
        cpt += 1

    P = np.ones((size_dictF, size_dictE)) / size_dictF
    diffP = 1

    counter = 0

    while(counter < N_ITER_MAX):  # and diffP > CONVERGENCE_CRITERIA):
        print("IBM2: Iteration %d . Norm difference of P: %d" % (counter, diffP))
        P_prev = P
        counter += 1
        C_align = np.zeros((size_dictF, size_dictE))
        C_word = np.zeros((size_dictE))

        for n in range(N):
            I = phrase_size_F[n]
            J = phrase_size_E[n] - offset  # Null word
            for i in range(phrase_size_F[n]):
                indF = imp.hash(D_F, re.split(' |\'', F[n])[i])
                Z = 0
                for j in range(phrase_size_E[n]):
                    indE = imp.hash(D_E, re.split(' |\'', E[n])[j])
                    Z += P[indF, indE] * b(j, i, J, I, lamb, p_null)
                for j in range(phrase_size_E[n]):
                    indE = imp.hash(D_E, re.split(' |\'', E[n])[j])
                    c = P[indF, indE] * b(j, i, J, I, lamb, p_null) / Z
                    C_align[indF, indE] += c
                    C_word[indE] += c

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P[i, j] = C_align[i, j] / C_word[j]
        diffP = abs(P - P_prev).max()

    if counter == N_ITER_MAX:
        print("Warning, in IBM2, reached maximum number of iterations")

    return P
