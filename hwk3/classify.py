##########################################
#### Probabilistic Graphical Models ######
############## Homework 3 ################
##########################################
# Sylvain TRUONG and Matthieu KIRCHMEYER #
##########################################
##########################################

# Call: python classify.py path_to_train path_to_test -n [number_clusters] -question [2,4,5,6,8,9,10,11]
# Several methods can be called at the same time (plots will then appear on the same figure)
# Call example: python classify.py data/EMGaussian.data data/EMGaussian.test -n_clusters 4 -question 2 4 5 6 8 9 10 11


import sys
import itertools
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import operator
from scipy.stats import multivariate_normal as normal_law

# custom imports
from lib import importation as imp
from lib import plotting
from lib import kmeans as km
from lib import GM
from lib import HMM

_legend_size = 10

def parseArguments():
    parser = argparse.ArgumentParser(description="Learn 2D-data-point clustering models and test them out")

    parser.add_argument("training_set",
        help="path_to_training_set")

    parser.add_argument("test_set",
        help="path_to_test_set")

    parser.add_argument('-n', '-n_clusters',
        type=int, help="(optional) Specify the target number of clusters (defaults to 4)")

    parser.add_argument('-q', '-question', nargs='*',
        type=int, help="(optional) Specify the questions you want the answers to (defauts to question 2) Questions are 2 4 5 6 8 9 10 11")

    args = parser.parse_args()
    return args

#########################
### GENERAL FUNCTIONS ###
########################
def design_matrix(data_points):
    result = data_points[:,0:data_points.shape[1]-1]
    return result


##################################################
###################### HWK3 ######################
##################################################


def question2(tr_data,test_data,n_clusters):
    # compute GM
    labels, mus, sigmas, pi_vector, Q = GM.EM_algo(tr_data,n_clusters)
    # define transition matrix
    A = (1.0/6.0)*np.ones([4,4])
    for k in range(len(A)):
        A[k,k] = 0.5
    # initial distribution for the hidden state
    pi = 0.25*np.ones(4)

    # compute messages and compute probabilities (first dimension is the hidden state index / second dimension is time
    probas = HMM.update_r(test_data,sigmas,mus,A,pi)

    # plot probabilities
    plotting.plot_r(probas.T,100,"question2")

def question4(data_points,A,mus,sigmas,pi):
    print "----- PI -----"
    print pi
    print "------ A -----"
    print A

def question5(tr_data,test_data,l_tr,l_test):
    # xaxis = range(len(l_tr))
    # f, axarr = plt.subplots(2, sharex=True, figsize=(8,8))
    # f.suptitle('Log-likelihood evolution (HMM model)', fontsize=15)
    # axarr[0].plot(xaxis,l_tr,'r')
    # axarr[0].set_ylabel('log-likelihood (train)')
    # ymin0, ymax0 = axarr[0].get_ylim()
    # axarr[0].grid(True)
    # axarr[1].plot(xaxis,l_test,'b')
    # axarr[1].set_ylabel('log-likelihood (test)')
    # ymin1, ymax1 = axarr[1].get_ylim()
    # axarr[1].set_xlabel('iterations')
    # axarr[1].grid(True)

    # # set ylim
    # axarr[0].set_ylim([min(ymin0,ymin1),max(ymax0,ymax1)])
    # axarr[1].set_ylim([min(ymin0,ymin1),max(ymax0,ymax1)])
    # #f.tight_layout()

    # plt.gcf().savefig("Report/Figures/question5.eps")
    # plt.close(plt.gcf())
    plotting.plot_log_likelihood_evolution(l_tr,l_test,output_name="question%d"%5)

def question6(tr_data,test_data,n_clusters,A,mus,sigmas,pi):
    labels, mu_vector, sigmas, pi_vector, Q =  GM.EM_algo_isotropic(tr_data,n_clusters)
    result_iso = np.array([GM.log_likelihood(tr_data,mu_vector,sigmas,pi_vector),GM.log_likelihood(test_data,mu_vector,sigmas,pi_vector)])

    labels, mu_vector, sigmas, pi_vector, Q =  GM.EM_algo(tr_data,n_clusters)
    result_gm = np.array([GM.log_likelihood(tr_data,mu_vector,sigmas,pi_vector),GM.log_likelihood(test_data,mu_vector,sigmas,pi_vector)])

    result_hmm = np.array([HMM.log_likelihood_HMM(tr_data,n_clusters,A,sigmas,mus,pi),HMM.log_likelihood_HMM(test_data,n_clusters,A,sigmas,mus,pi)])

    print '---------------'
    print 'LOG LIKELIHOODS'
    print '---------------'
    print 'Isotropic GM'
    print result_iso
    print ''
    print 'General GM'
    print result_gm
    print ''
    print 'Hidden Markov Model'
    print result_hmm
    return

def viterbi(data, sigmas, mus, A, p, K, plot):
    T = len(data) #number of points
    state = np.zeros((K,T))
    out = np.zeros((T))

    # Initialization
    log_v = np.zeros((K, T))
    log_condi_p = np.array([normal_law.logpdf(data[0], mus[i], sigmas[i]) for i in range(K)])
    log_p = np.log(p)

    for k in range(K):
        # V(0,k) = max_q0 P(u0 | q0) P(q1 = k | q0) P(q0)
        (log_v[k,0], etat) = max((log_condi_p[s] + np.log(A[k,s]) + log_p[s], s) for s in range(K))
        state[k,0] = etat

    # Propagation
    for t in range(1, T-1):
        log_condi_p = (np.array([normal_law.logpdf(data[t], mus[i], sigmas[i]) for i in range(K)]))
        for k in range(K):
            # V(t,k) = max_qt P(ut | qt) P(qt+1 = k | qt) V(t-1, qt)
            (log_v[k,t], etat) = max((log_condi_p[s] + np.log(A[k,s]) + log_v[s, t-1], s) for s in range(K))
            state[k,t] = etat

    # Last step
    log_condi_p = (np.array([normal_law.logpdf(data[T-1], mus[i], sigmas[i]) for i in range(K)]))
    for k in range(K):
        # V(T-1,k) = max_qT-1 P(uT-1 | qT-1) V(T-2, qT-1)
        (log_v[k,T-1], etat) = max((log_condi_p[s] + log_v[s, T-2], s) for s in range(K))
        state[k,T-1] = etat

    # Compute the viterbi decoding
    q_best = np.argmax(log_v[:,T-1])
    out[T-1] = q_best
    for t in range(T-2,-1,-1):
        out[int(t)] = state[out[int(t)+1],int(t)]

    return out

##################################################
###################### MAIN ######################
##################################################

def main():
    # plot setttings
    supported_questions = np.array([2,4,5,6,8,9,10,11])

    args = parseArguments()
    if args.q is None:
        questions = [2]
    else:
        questions = args.q

    if args.n is None:
        n_clusters = 4
    else:
        n_clusters = args.n

    (tr_data, test_data) = imp.importdata(args.training_set,args.test_set)

    print "---------------"
    print "IMPORTED DATA SHAPE"
    print "training set : " + str(tr_data.shape)
    print "test set : " +str(test_data.shape)
    print "---------------"

    print "---- First, compute all models for all clustering methods ----"
    # l_train is the sequence of log likelihoods along the iterations of the training data
    # l_test is the sequence of log likelihoods along the iterations of the training data

    _, mu_vector_iso, sigmas_iso, pi_vector_iso,_ =  GM.EM_algo_isotropic(tr_data,n_clusters)
    _, mu_vector_gm, sigmas_gm, pi_vector_gm, _ =  GM.EM_algo(tr_data,n_clusters)

    A, mus, sigmas, pi, l_tr, l_test = HMM.EM_HMM_likelihood(tr_data,test_data,n_clusters,mu_vector_gm,sigmas_gm)
    r_tr = HMM.update_r(tr_data,sigmas,mus,A,pi)
    p_tr = HMM.update_pi0(r_tr)
    r_test = HMM.update_r(test_data,sigmas,mus,A,pi)
    print "------------------ All models computed -----------------------"

    for q in questions:
        if (q==2):
            print '------------------'
            print 'Computing answer 2'
            # define transition matrix
            A_dummy = (1.0/6.0)*np.ones([4,4])
            for k in range(len(A_dummy)):
                A_dummy[k,k] = 0.5
            # initial distribution for the hidden state
            pi_dummy = 0.25*np.ones(4)

            # compute messages and compute probabilities (first dimension is the hidden state index / second dimension is time
            r_dummy = HMM.update_r(test_data,sigmas_gm,mu_vector_gm,A_dummy,pi_dummy)

            number_of_points_to_show = 100
            # plot probabilities
            plotting.plot_r(r_dummy.T,number_of_points_to_show,"question2")
            continue
        if (q==4):
            print '------------------'
            print 'Computing answer 4'
            print "----- PI -----"
            print pi
            print "------ A -----"
            print A
            continue
        if (q==5):
            print '------------------'
            print 'Computing answer 5'
            plotting.plot_log_likelihood_evolution(l_tr,l_test,"question%d" % q)
            continue
        if (q==6):
            print '------------------'
            print 'Computing answer 6'
            result_iso = np.array([GM.log_likelihood(tr_data,mu_vector_iso,sigmas_iso,pi_vector_iso),GM.log_likelihood(test_data,mu_vector_iso,sigmas_iso,pi_vector_iso)])

            result_gm = np.array([GM.log_likelihood(tr_data,mu_vector_gm,sigmas_gm,pi_vector_gm),GM.log_likelihood(test_data,mu_vector_gm,sigmas_gm,pi_vector_gm)])

            result_hmm = np.array([HMM.log_likelihood_HMM(tr_data,n_clusters,A,sigmas,mus,pi),HMM.log_likelihood_HMM(test_data,n_clusters,A,sigmas,mus,pi)])

            print '----------------------------------'
            print 'final LOG LIKELIHOODS [train,test]'
            print '----------------------------------'
            print 'Isotropic GM'
            print result_iso
            print ''
            print 'General GM'
            print result_gm
            print ''
            print 'Hidden Markov Model'
            print result_hmm

            continue
        if (q==8):
            print '------------------'
            print 'Computing answer 8'
            out = viterbi(tr_data, sigmas, mus, A, p_tr, n_clusters, True)
            plotting.plot_labeled_data("Viterbi decoding on training dataset",tr_data,out,mus,"question%d" % q)
            continue
        if (q==9):
            print '------------------'
            print 'Computing answer 9'
            # Plot marginal probability on test set
            plotting.plot_r(r_test.T, 100, "question%d" % q)
            continue
        if (q==10):
            print '------------------'
            print 'Computing answer 10'
            state_10 = np.argmax(r_test, axis = 1)
            state_10 = state_10[0:100]
            print '------------------'
            print 'Computing answer 11'
            out = viterbi(test_data, sigmas, mus, A, p_tr, n_clusters, False)
            state_11 = out[0:100]
            print '------------------'
            print 'Plotting answer 10-11'
            f, axarr = plt.subplots(3, sharex=True, figsize=(8,8))
            f.suptitle('State using marginal decoding, viterbi and differences between the 2', fontsize=15)
            # Plot the most likely state according to marginal probability
            axarr[0].set_title("State using marginal probability")
            axarr[0].set_ylabel("State")
            width = 1/1.5
            x = np.arange(100)
            axarr[0].bar(x[state_10==0], state_10[state_10==0]+1, width, color='r', label="q_t = 0")
            axarr[0].bar(x[state_10==1], state_10[state_10==1], width, color='b', label="q_t = 1")
            axarr[0].bar(x[state_10==2], state_10[state_10==2]-1, width, color='g', label="q_t = 2")
            axarr[0].bar(x[state_10==3], state_10[state_10==3]-2, width, color='k', label="q_t = 3")
            axarr[0].legend(numpoints=1)
            ymin0, ymax0 = axarr[0].get_ylim()
            # Plot the most likely state using Viterbi decoding
            axarr[1].set_title("State using Viterbi decoding")
            axarr[1].set_ylabel("State")
            y = np.arange(100)
            axarr[1].bar(y[state_11==0], state_11[state_11==0]+1, width, color='r', label="q_t = 0")
            axarr[1].bar(y[state_11==1], state_11[state_11==1], width, color='b', label="q_t = 1")
            axarr[1].bar(y[state_11==2], state_11[state_11==2]-1, width, color='g', label="q_t = 2")
            axarr[1].bar(y[state_11==3], state_11[state_11==3]-2, width, color='k', label="q_t = 3")
            axarr[1].legend(numpoints=1)
            ymin1, ymax1 = axarr[1].get_ylim()
            # Plot the difference
            axarr[2].set_title("Differences between the 2 methods")
            axarr[2].set_xlabel("Time")
            axarr[2].set_ylabel("Difference")
            z = np.arange(100)
            labels = np.zeros((100))
            for i in range(100):
                if state_11[i] != state_10[i]:
                    labels[i] = 1
            axarr[2].bar(z, labels, width, color='r', label="Difference")
            axarr[2].legend(numpoints=1)
            axarr[2].set_ylim([min(ymin0,ymin1),max(ymax0,ymax1)])
            # Save figure
            plt.gcf().savefig("Report/Figures/question10-11.eps")
            plt.close(plt.gcf())
            continue

        print "Warning : unsupported question -> %d" % q
        continue

    plt.show()

if __name__ == '__main__':
    main()
