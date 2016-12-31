import numpy as np
from scipy.stats import multivariate_normal as normal_law
import kmeans as km
import sys
import math

CONVERGENCE_THR = 0.01
###################
##### UPDATES #####
###################

def update_mu(data_points,Q):
    mu_vector = np.zeros((Q.shape[1],2))
    for k in range(len(mu_vector)):
        mu_vector_x = np.dot(Q[:,k].T,data_points[:,0])/np.sum(Q[:,k])
        mu_vector_y = np.dot(Q[:,k].T,data_points[:,1])/np.sum(Q[:,k])
        mu_vector[k] = [mu_vector_x,mu_vector_y]
    return mu_vector

def update_pi(Q):
    pi_vector = np.sum(Q,0) / np.sum(Q)
    return pi_vector

def update_sigmas(data_points,Q,mu_vector):
    sigmas_update = np.zeros([len(mu_vector),data_points.shape[1],data_points.shape[1]]);
    for k in range(sigmas_update.shape[0]):
        sigmas_update[k,:,:] = np.dot(Q[:,k]*(data_points - mu_vector[k]).T,(data_points - mu_vector[k])) / (np.sum(Q[:,k]))
    return sigmas_update

def update_sigmas_isotropic(data_points,Q,mu_vector):
    sigmas_update = np.zeros([len(mu_vector),data_points.shape[1],data_points.shape[1]]);
    for k in range(sigmas_update.shape[0]):
        lam = np.trace(np.dot(np.diag(Q[:,k]),np.dot(data_points-mu_vector[k],(data_points-mu_vector[k]).T)));
        lam /= np.sum(Q[:,k]) * data_points.shape[1]

        sigmas_update[k,:,:] = lam * np.eye(data_points.shape[1])
    return sigmas_update

def update_Q(data_points,mu_vector,sigmas,pi_vector):
    Q_update = np.zeros((len(data_points),len(mu_vector)))

    for i in range(len(data_points)):

        den = 0
        for k in range(len(mu_vector)):
            den += pi_vector[k]*normal_law.pdf(data_points[i],mu_vector[k],sigmas[k])

        for k in range(len(mu_vector)):
            Q_update[i,k] = pi_vector[k]*normal_law.pdf(data_points[i],mu_vector[k],sigmas[k]) / den

    return Q_update



###################
####### EM ########
###################


def EM_algo(data_points,nb_clusters):
    print "Computing Gaussian Mixture with EM"
    # init with k_means
    [labels,mu_vector] = km.kmeans(data_points,nb_clusters)
    sigmas = np.zeros([len(mu_vector),data_points.shape[1],data_points.shape[1]])
    possible_labels = np.unique(labels)

    pi_vector = np.zeros(len(possible_labels))
    for k in range(len(pi_vector)):
        pi_vector[k] = float(np.sum(labels==possible_labels[k])) / float(len(data_points))
    # init scalar covariance matrices
    for k in range(len(sigmas)):
        sigmas[k] = 3*np.eye(sigmas.shape[1])

    # first expectation step
    Q = update_Q(data_points,mu_vector,sigmas,pi_vector)

    convergence_thr = CONVERGENCE_THR
    counter = 0
    max_iter = 100

    # while convergence criterion not met
    while counter < max_iter:
        print "\riteration : %d" % counter,
        sys.stdout.flush()
        # Maximization step
        mu_vector = update_mu(data_points,Q)
        sigmas = update_sigmas(data_points,Q,mu_vector)
        pi_vector = update_pi(Q)

        # Expectation step
        Q_update = update_Q(data_points,mu_vector,sigmas,pi_vector)

        if (np.linalg.norm(Q-Q_update) < convergence_thr):
            break
        counter += 1
        Q = Q_update

    if (counter == max_iter):
        print "warning: in EM_algo: counter reached max_iter, convergence is not ensured"

    print ""
    # issue the labels
    for i in range(len(labels)):
        labels[i] = np.argmax(Q[i,:])

    return labels, mu_vector, sigmas, pi_vector, Q



def EM_algo_isotropic(data_points,nb_clusters):
    print "Computing Gaussian Isotropic Mixture with EM"
    # init with k_means
    [labels,mu_vector] = km.kmeans(data_points,nb_clusters)
    sigmas = np.zeros([len(mu_vector),data_points.shape[1],data_points.shape[1]])
    possible_labels = np.unique(labels)

    pi_vector = np.zeros(len(possible_labels))
    for k in range(len(pi_vector)):
        pi_vector[k] = float(np.sum(labels==possible_labels[k])) / float(len(data_points))
    # init identity covariance matrices
    for k in range(len(sigmas)):
        sigmas[k] = 3*np.eye(sigmas.shape[1])

    # first expectation step
    Q = update_Q(data_points,mu_vector,sigmas,pi_vector)

    convergence_thr = CONVERGENCE_THR
    counter = 0
    max_iter = 100

    # while convergence criterion not met
    while counter < max_iter:
        print "\riteration : %d" % counter,
        sys.stdout.flush()
        # Maximization step
        mu_vector = update_mu(data_points,Q)
        sigmas = update_sigmas_isotropic(data_points,Q,mu_vector)
        pi_vector = update_pi(Q)

        # Expectation step
        Q_update = update_Q(data_points,mu_vector,sigmas,pi_vector)

        if (np.linalg.norm(Q-Q_update) < convergence_thr):
            break
        counter += 1
        Q = Q_update

    if (counter == max_iter):
        print "warning: in EM_algo: counter reached max_iter, convergence is not ensured"

    print ""
    # issue the labels
    for i in range(len(labels)):
        labels[i] = np.argmax(Q[i,:])

    return labels, mu_vector, sigmas, pi_vector, Q



####################
##### REVIEW #######
####################

def log_likelihood(data_points,mu_vector,sigmas,pi_vector):
    Q = update_Q(data_points,mu_vector,sigmas,pi_vector)
    result = 0
    for i in range(len(data_points)):
        for k in range(len(sigmas)):
            result += Q[i,k] * (normal_law.logpdf(data_points[i],mu_vector[k],sigmas[k]) + math.log(pi_vector[k]))
    return result