import numpy as np
from scipy.stats import multivariate_normal as normal_law
import kmeans as km
import sys
import GM
import math

CONVERGENCE_THR = 0.01

### expectation step ###

def log_alpha_rec(sigmas, mus, A, previous_log_messages, states, observation):
    result = np.zeros(len(states))
    for state in states:
        tmp_vector = np.log(A[state,:])+previous_log_messages
        result[state] = log_sum(tmp_vector)
        result[state] += normal_law.logpdf(observation,mus[state],sigmas[state])
    return result

def log_beta_rec(sigmas, mus, A, next_log_messages, states, observation):
    result = np.zeros(len(states))
    for state in states:
        tmp_vector = np.log(A[:,state]) + next_log_messages
        for k in range(len(next_log_messages)):
            tmp_vector[k] += normal_law.logpdf(observation,mus[k],sigmas[k])
        result[state] = log_sum(tmp_vector)
    return result

def log_sum(log_values):
    #max_idx = np.argmax(log_values)
    max_val = np.max(log_values)
    result = max_val + math.log(np.sum(np.exp(log_values-max_val)))
    #result = max_val + math.log(1+np.sum(np.exp(log_values-max_val)*(range(len(log_values))!=max_val)))
    return result

def compute_all_alpha(data_points,sigmas,mus,A,pi):
    # compute all logarithms of alpha messages
    # pi is the initial probability distribution over hidden states
    n_states = len(sigmas)
    alphas = np.zeros([n_states,len(data_points)])
    # initialization
    for state in range(n_states):
        alphas[state,0] = math.log(pi[state]) + normal_law.logpdf(data_points[0],mus[state],sigmas[state])
    # log recursion
    for t in range(1,len(data_points)):
        alphas[:,t] = log_alpha_rec(sigmas,mus,A,alphas[:,t-1],range(n_states),data_points[t])
    #alphas = np.exp(alphas)
    return alphas

def compute_all_beta(data_points,sigmas,mus,A):
    # compute all logarithms of beta messages
    n_states = len(sigmas)
    betas = np.zeros([n_states,len(data_points)])
    # initialization
    betas[:,-1] = 0
    # log recursion
    for t in range(len(data_points)-2,-1,-1):
        betas[:,t] = log_beta_rec(sigmas,mus,A,betas[:,t+1],range(n_states),data_points[t+1])
    #betas = np.exp(betas)
    return betas

def cond_proba_unary(log_alphas, log_betas):
    probas = log_alphas + log_betas
    for t in range(probas.shape[1]):
        probas[:,t] = probas[:,t] - log_sum(probas[:,t])
    probas = np.exp(probas)
    return probas

def cond_proba_binary(log_alphas, log_betas, A, observations, mus, sigmas):
    probas = np.zeros([log_alphas.shape[1]-1,A.shape[0],A.shape[1]])
    for t in range(probas.shape[0]):
        for i in range(probas.shape[1]):
            for j in range(probas.shape[2]):
                probas[t,i,j] += log_alphas[j,t] + log_betas[i,t+1]
                probas[t,i,j] += math.log(A[i,j])
                probas[t,i,j] += normal_law.logpdf(observations[t+1],mus[i],sigmas[i])
                probas[t,i,j] -= log_sum(log_alphas[:,t] + log_betas[:,t])
    probas = np.exp(probas)
    return probas

def update_r(data_points,sigmas,mus,A,pi):
    alphas = compute_all_alpha(data_points,sigmas,mus,A,pi)
    betas = compute_all_beta(data_points,sigmas,mus,A)
    probas = cond_proba_unary(alphas,betas)
    return probas.T

def update_s(data_points,sigmas,mus,A,pi):
    alphas = compute_all_alpha(data_points,sigmas,mus,A,pi)
    betas = compute_all_beta(data_points,sigmas,mus,A)
    probas = cond_proba_binary(alphas,betas,A,data_points,mus,sigmas)
    return probas



####################################
############ MAXIMISATION ##########
####################################

def update_mu(data_points,Q):
    mu_vector = np.zeros((Q.shape[1],2))
    for k in range(len(mu_vector)):
        mu_vector_x = np.dot(Q[:,k].T,data_points[:,0])/np.sum(Q[:,k])
        mu_vector_y = np.dot(Q[:,k].T,data_points[:,1])/np.sum(Q[:,k])
        mu_vector[k] = [mu_vector_x,mu_vector_y]
    return mu_vector

def update_sigmas(data_points,Q,mu_vector):
    sigmas_update = np.zeros([len(mu_vector),data_points.shape[1],data_points.shape[1]]);
    for k in range(sigmas_update.shape[0]):
        sigmas_update[k,:,:] = np.dot(Q[:,k]*(data_points - mu_vector[k]).T,(data_points - mu_vector[k])) / (np.sum(Q[:,k]))
    return sigmas_update

def update_A(s): #check the report for definition of s
    A_update = np.zeros([s.shape[-2],s.shape[-1]])
    for k in range(s.shape[-2]):
        for l in range(s.shape[-1]):
            A_update[k,l] = np.sum(s[:,k,l]) / np.sum(s[:,:,l])
    return A_update

def update_pi0(r): #check the report for definition of r
    pi0_update = r[0,:] / np.sum(r[0,:])
    return pi0_update



#####################################
########### EM ######################
#####################################


def EM_HMM(data_points,n_states):

    ##############################
    ####### INITIALISATION #######
    ##############################

    # init with GM
    print "init with GM ..."
    labels, mus, sigmas, pi_vector, Q = GM.EM_algo(data_points,n_states)
    # init transition matrix
    A = (1.0/6.0)*np.ones([4,4])
    for k in range(len(A)):
        A[k,k] = 0.5
    # init initial distribution for the hidden state
    pi = 0.25*np.ones(4)

    ##############################
    #### FIRST ROUND (E-STEP) ####
    ##############################
    print "Computing EM on the HMM ..."
    r = update_r(data_points,sigmas,mus,A,pi)
    s = update_s(data_points,sigmas,mus,A,pi)

    ##############################
    #### WHILE NOT CONVERGED #####
    ##############################
    convergence_thr = CONVERGENCE_THR
    counter = 0
    max_iter = 100

    # while convergence criterion not met
    while counter < max_iter:
        print "\riteration : %d" % counter,
        sys.stdout.flush()

        # Maximization step
        mus = update_mu(data_points,r)
        sigmas = update_sigmas(data_points,r,mus)
        A = update_A(s)
        pi = update_pi0(r)

        # Expectation step
        r_update = update_r(data_points,sigmas,mus,A,pi)
        s_update = update_s(data_points,sigmas,mus,A,pi)

        if (max(np.linalg.norm(r-r_update),np.linalg.norm(s-s_update)) < convergence_thr ):
            break
        counter += 1
        r = r_update
        s = s_update

    if (counter == max_iter):
        print "warning: in EM_algo: counter reached max_iter, convergence is not ensured"

    print ""
    # issue the labels
    #labels = np.zeros(len(data_points))
    #for i in range(len(labels)):
    #   labels[i] = np.argmax(r[i,:])

    return A, mus, sigmas, pi

def log_likelihood_HMM(data_points,n_states,A,sigmas,mus,pi,r=None,s=None):
    if r is None:
        r = update_r(data_points,sigmas,mus,A,pi)
    if s is None:
        s = update_s(data_points,sigmas,mus,A,pi)
    result = 0
    result += np.sum(r[0,:]*np.log(pi))
    for k in range(len(sigmas)):
        for t in range(len(data_points)):
            result += r[t,k]*normal_law.logpdf(data_points[t],mus[k],sigmas[k])
            if (t < len(data_points) - 1):
                for l in range(len(sigmas)):
                    result += s[t,k,l]*math.log(A[k,l])
    return result

def EM_HMM_likelihood(tr_data,test_data,n_states,mus,sigmas): # run the EM on HMM while recording the log likelihoods along the iterations
    log_likelihood_tr = []
    log_likelihood_test = []
    ##############################
    ####### INITIALISATION #######
    ##############################

    # init with GM
    #print "init with GM ..."
    #_, mus, sigmas, _, _ = GM.EM_algo(tr_data,n_states)
    # init transition matrix
    A = (1.0/6.0)*np.ones([4,4])
    for k in range(len(A)):
        A[k,k] = 0.5
    # init initial distribution for the hidden state
    pi = 0.25*np.ones(4)

    ##############################
    #### FIRST ROUND (E-STEP) ####
    ##############################
    print "Computing EM on the HMM ..."
    r = update_r(tr_data,sigmas,mus,A,pi)
    s = update_s(tr_data,sigmas,mus,A,pi)


    ##############################
    #### WHILE NOT CONVERGED #####
    ##############################
    convergence_thr = CONVERGENCE_THR
    counter = 0
    max_iter = 100

    # while convergence criterion not met
    while counter < max_iter:
        print "\riteration : %d" % counter,
        sys.stdout.flush()

        # record log likelihoods
        log_likelihood_tr.append(log_likelihood_HMM(tr_data,n_states,A,sigmas,mus,pi,r,s))
        log_likelihood_test.append(log_likelihood_HMM(test_data,n_states,A,sigmas,mus,pi))

        # Maximization step
        mus = update_mu(tr_data,r)
        sigmas = update_sigmas(tr_data,r,mus)
        A = update_A(s)
        pi = update_pi0(r)
        # Expectation step
        r_update = update_r(tr_data,sigmas,mus,A,pi)
        s_update = update_s(tr_data,sigmas,mus,A,pi)

        if (max(np.linalg.norm(r-r_update),np.linalg.norm(s-s_update)) < convergence_thr ):
            break
        counter += 1
        r = r_update
        s = s_update

    if (counter == max_iter):
        print "warning: in EM_algo: counter reached max_iter, convergence is not ensured"

    print ""
    # issue the labels
    #labels = np.zeros(len(tr_data))
    #for i in range(len(labels)):
    #   labels[i] = np.argmax(r[i,:])

    return A, mus, sigmas, pi , log_likelihood_tr, log_likelihood_test