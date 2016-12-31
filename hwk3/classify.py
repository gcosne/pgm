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
from scipy.stats import multivariate_normal

CONVERGENCE_THR = 0.01
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
        type=int, help="(optional) Specify the questions you want the answers to (defauts to question 2) Questions are 2,4,5,6")

    args = parser.parse_args()
    return args

#########################
### GENERAL FUNCTIONS ###
#########################
def importdata(path_to_train ,path_to_test): 
    train_file = open(path_to_train,'r')
    test_file = open(path_to_test,'r')

    tr_data = np.array([])
    test_data = np.array([])

    for line in train_file:
        data_point_str = line.split()
        data_point_float = []
        for d in data_point_str:
            data_point_float.append(float(d))
        
        # check number of data points in tr_data
        if tr_data.size == 0:
            tr_data = np.array(data_point_float[0:2])
        else:
            tr_data = np.vstack((tr_data,data_point_float[0:2]))
    
    train_file.close()

    for line in test_file:
        data_point_str = line.split()
        data_point_float = []
        for d in data_point_str:
            data_point_float.append(float(d))
        
        # check number of data points in tr_data
        if test_data.size == 0:
            test_data = np.array(data_point_float[0:2])
        else:
            test_data = np.vstack((test_data,data_point_float[0:2]))
    
    test_file.close()
    
    return tr_data, test_data

def plot_data_train_test_labeled(tr_data, test_data, labels_tr, labels_test, title_complement = "",comp1="", comp2=""):
    f, axarr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14.5,8))
    col = plot_labeled_data("Training data " + title_complement + comp1,tr_data,labels_tr,axarr[0])
    axarr[0].legend(prop={'size':_legend_size})
    plot_labeled_data("Test data "+ title_complement + comp2,test_data,labels_test,axarr[1],col)
    axarr[1].legend(prop={'size':_legend_size})
    f.tight_layout()
    return f, axarr, col

def plot_GM_distrib(mu_vector,sigmas, plot=plt, col = []):
    assert len(mu_vector) == len(sigmas), "in plot_GM_distrib: number of clusters not matching"

    color_array = np.random.rand(len(mu_vector),3,1)

    if (len(col) > 1):
        colors = itertools.cycle(col)
    else:
        colors = itertools.cycle(color_array)

    for k in range(len(mu_vector)):
        eigval, eigvec = np.linalg.eig(sigmas[k])
        
        ### plot the center
        plot.scatter(mu_vector[k][0],mu_vector[k][1],marker="o",c="k",s=40)

        ### plot the ellipsoid
        t = np.linspace(0,2*math.pi,100)
        points = np.zeros((len(t),2))
        
        if eigval[0]>eigval[1]:
            maxi=0
            mini=1
        else:
            maxi=1
            mini=0

        angle=-math.atan2(eigvec[maxi][1],eigvec[maxi][0])
        R1 = np.array([math.cos(angle),math.sin(angle)])
        R2 = np.array([-math.sin(angle),math.cos(angle)])
        R = np.vstack((R1,R2))

        for i in range(len(t)):
            ellipse_x_r=math.cos(t[i])*math.sqrt(eigval[maxi]*4.6052) # 90%-ellipse
            ellipse_y_r=math.sin(t[i])*math.sqrt(eigval[mini]*4.6052) # 90%-ellipse
            r_ellipse = np.dot(np.transpose(np.array([ellipse_x_r,ellipse_y_r])),R)
            r_ellipse = np.transpose(r_ellipse)
            points[i,:] = mu_vector[k] + r_ellipse
        
        plot.plot(points[:,0],points[:,1],color=next(colors))


def plot_labeled_data(title,points,labels,plot = plt, col = []):
    if (plot != plt):
        plot.set_title(title)
    list_labels = np.unique(labels)
    color_array = np.random.rand(len(list_labels),3,1)

    if (len(col) > 1):
        colors = itertools.cycle(col)
    else:
        colors = itertools.cycle(color_array)

    for label in list_labels:
        mask_label = (labels == label)
        if (np.sum(mask_label) > 0):
            plot.scatter(points[mask_label,0], points[mask_label,1],label = "Cluster %d" % label,color=next(colors))    
        else:
            next(colors)
    return color_array


def design_matrix(data_points):
    result = data_points[:,0:data_points.shape[1]-1]
    return result


#######################
####### K-MEANS #######
#######################
def assign_label_kmeans(x,centroids):
    # based on euclidean distance
    distances = np.zeros(len(centroids))
    for k in range(len(centroids)):
        distances[k] = np.linalg.norm(x-centroids[k])
    #print distances
    return np.argmin(distances)

def kmeans(data_points,nb_clusters):
## initialize nb_clusters random centroids
    centroids_idx = np.unique(np.floor(len(data_points)*np.random.rand(nb_clusters)))

    safety_counter = 0
    max_safety = 50
    while (len(centroids_idx)<nb_clusters and safety_counter<max_safety):
        centroids_idx = np.unique(np.floor(len(data_points)*np.random.rand(nb_clusters)))

    assert safety_counter<max_safety, "in kmeans : Failed to initialize enough centroids"   

    centroids_idx = centroids_idx.astype(int)
    centroids = data_points[centroids_idx]

    distances = np.zeros(len(centroids))
    for k in range(len(centroids)):
        distances[k] = np.linalg.norm(data_points[0]-centroids[k])

    labels = np.zeros(len(data_points))

    max_iter = 1000
    counter = 0
    convergence_criteria = CONVERGENCE_THR

# begin the k-means iterations
    while(counter<max_iter):
        counter += 1
        
        # assign labels
        for i in range(len(data_points)):
            labels[i] = assign_label_kmeans(data_points[i],centroids)

        # update centroids
        new_centroids = np.zeros(centroids.shape)
        for k in range(len(centroids)):
            ### problem with an empty slice mean
            
            tmp_mean = np.mean(data_points[labels==k],0)
            new_centroids[k,0] = tmp_mean[0]
            new_centroids[k,1] = tmp_mean[1]

        displacement = np.zeros(len(centroids))
        for k in range(len(centroids)):
            displacement[k] = np.linalg.norm(new_centroids[k]-centroids[k])

        if (np.max(displacement)<convergence_criteria):
            break
        else:
            centroids = new_centroids

    return labels, centroids


#######################
## MAXIMISATION STEP ##
#######################
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
        lam /= np.sum(Q[:,k])
        
        sigmas_update[k,:,:] = lam * np.eye(data_points.shape[1])
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

#######################
## EXPECTATION STEP ##
#######################
def normal_law_log(x,mu,sigma):
    # return log of the normal prob
    d = x.shape[0]

    factor = ((2*math.pi)**(float(d)/2.0)*np.sqrt(np.linalg.det(sigma)))**(-1)
    prob = -np.dot(np.dot(x-mu,np.linalg.inv(sigma)),(x-mu))
    return math.log(factor) + prob  

def normal_law(x,mu,sigma):
    d = x.shape[0]

    factor = ((2*math.pi)**(float(d)/2.0)*np.sqrt(np.linalg.det(sigma)))**(-1)
    prob = math.exp(-np.dot(np.dot(x-mu,np.linalg.inv(sigma)),(x-mu)))
    return factor*prob  

def normal_law_2D(x1,y1,mu,sigma):
    assert(x1.shape==y1.shape), "in normal_law_2D : shape mismatch"

    result = np.zeros(x1.shape)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            x = np.array([x1[i,j],y1[i,j]])
            result[i,j] = normal_law(x,mu,sigma)    

    return result   

def update_Q(data_points,mu_vector,sigmas,pi_vector):
    Q_update = np.zeros((len(data_points),len(mu_vector)))
    
    for i in range(len(data_points)):
        
        den = 0
        for k in range(len(mu_vector)):
            den += pi_vector[k]*normal_law(data_points[i],mu_vector[k],sigmas[k])
        
        for k in range(len(mu_vector)):
            Q_update[i,k] = pi_vector[k]*normal_law(data_points[i],mu_vector[k],sigmas[k]) / den
    
    return Q_update


def EM_algo(data_points,nb_clusters):
    print "Computing Gaussian Mixture with EM"
    # init with k_means
    [labels,mu_vector] = kmeans(data_points,nb_clusters)
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
    [labels,mu_vector] = kmeans(data_points,nb_clusters)
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


def assign_label_gm(data_points,mu_vector,sigmas,pi_vector):
    Q = update_Q(data_points,mu_vector,sigmas,pi_vector)
    # issue the labels
    labels = np.zeros(len(Q))
    for i in range(len(data_points)):
        labels[i] = np.argmax(Q[i,:])
    return labels

def log_likelihood(data_points,mu_vector,sigmas,pi_vector):
    Q = update_Q(data_points,mu_vector,sigmas,pi_vector)
    result = 0
    for i in range(len(data_points)):
        for k in range(len(sigmas)):
            result += Q[i,k] * (normal_law_log(data_points[i],mu_vector[k],sigmas[k]) + math.log(pi_vector[k]))
    return result

##################################################
###################### HWK3 ######################
##################################################
def log_alpha_rec(sigmas, mus, A, previous_log_messages, states, observation):
    result = np.zeros(len(states))
    for state in states:
        tmp_vector = np.log(A[state,:])+previous_log_messages
        result[state] = log_sum(tmp_vector) 
        result[state] += normal_law_log(observation,mus[state],sigmas[state])
    return result

def log_beta_rec(sigmas, mus, A, next_log_messages, states, observation):
    result = np.zeros(len(states))
    for state in states:
        tmp_vector = np.log(A[:,state]) + next_log_messages
        for k in range(len(next_log_messages)):
            tmp_vector[k] += normal_law_log(observation,mus[k],sigmas[k])
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
        alphas[state,0] = math.log(pi[state]) + normal_law_log(data_points[0],mus[state],sigmas[state])
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
                probas[t,i,j] += normal_law_log(observations[t+1],mus[i],sigmas[i])
                probas[t,i,j] -= log_sum(log_alphas[:,t] + log_betas[:,t])
    probas = np.exp(probas) 
    return probas

### expectation step ###
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

def EM_HMM(data_points,n_states):

    ##############################
    ####### INITIALISATION #######
    ##############################

    # init with GM
    print "init with GM ..."
    labels, mus, sigmas, pi_vector, Q = EM_algo(data_points,n_states)
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
            result += r[t,k]*normal_law_log(data_points[t],mus[k],sigmas[k]) 
            if (t < len(data_points) - 1):
                for l in range(len(sigmas)):
                    result += s[t,k,l]*math.log(A[k,l])
    return result

def EM_HMM_likelihood(tr_data,test_data,n_states):
    log_likelihood_tr = []
    log_likelihood_test = []
    ##############################
    ####### INITIALISATION #######
    ##############################

    # init with GM
    print "init with GM ..."
    labels, mus, sigmas, pi_vector, Q = EM_algo(tr_data,n_states)
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

    return log_likelihood_tr, log_likelihood_test

def plot_r(probas, N):
    width = 1/1.5
    xaxis = np.arange(N)
    f, axarr = plt.subplots(4, sharex=True, figsize=(12,8))
    
    axarr[0].set_title("Probability r")
    axarr[0].bar(xaxis,probas[0,0:N],width,color='r',label='p(q_t=0|u_1...u_t)')
    axarr[1].bar(xaxis,probas[1,0:N],width,color='b',label='p(q_t=1|u_1...u_t)')
    axarr[2].bar(xaxis,probas[2,0:N],width,color='g',label='p(q_t=2|u_1...u_t)')
    axarr[3].bar(xaxis,probas[3,0:N],width,color='k',label='p(q_t=3|u_1...u_t)')
    axarr[3].set_xlabel('time t')
    
    axarr[0].legend(numpoints=1)
    axarr[1].legend(numpoints=1)
    axarr[2].legend(numpoints=1)
    axarr[3].legend(numpoints=1)

    plt.gcf().savefig("Report/Figures/question9.png")
    plt.close(plt.gcf())

def question2(tr_data,test_data,n_states):
    # compute GM
    labels, mus, sigmas, pi_vector, Q = EM_algo(tr_data,n_states)
    # define transition matrix
    A = (1.0/6.0)*np.ones([4,4])
    for k in range(len(A)):
        A[k,k] = 0.5
    # initial distribution for the hidden state
    pi = 0.25*np.ones(4)
    
    # compute messages and compute probabilities (first dimension is the hidden state index / second dimension is time
    probas = update_r(test_data,sigmas,mus,A,pi)

    # plot probabilities
    plot_r(probas.T,100)

def question4(data_points,n_clusters):
    A, mus, sigmas, pi = EM_HMM(data_points,n_clusters)
    print "----- PI -----"
    print pi
    print "------ A -----"
    print A

def question5(tr_data,test_data,n_clusters):
    l_tr, l_test = EM_HMM_likelihood(tr_data,test_data,n_clusters)
    xaxis = range(len(l_tr))
    f, axarr = plt.subplots(2, sharex=True, figsize=(8,8))
    f.suptitle('Log-likelihood evolution (HMM model)', fontsize=15)
    axarr[0].plot(xaxis,l_tr,'r')
    axarr[0].set_ylabel('log-likelihood (train)')
    ymin0, ymax0 = axarr[0].get_ylim()
    axarr[0].grid(True)
    axarr[1].plot(xaxis,l_test,'b')
    axarr[1].set_ylabel('log-likelihood (test)')
    ymin1, ymax1 = axarr[1].get_ylim()
    axarr[1].set_xlabel('iterations')
    axarr[1].grid(True)
    
    # set ylim
    axarr[0].set_ylim([min(ymin0,ymin1),max(ymax0,ymax1)])
    axarr[1].set_ylim([min(ymin0,ymin1),max(ymax0,ymax1)])
    #f.tight_layout()

def question6(tr_data,test_data,n_states):
    labels, mu_vector, sigmas, pi_vector, Q =  EM_algo_isotropic(tr_data,n_states)
    result_iso = np.array([log_likelihood(tr_data,mu_vector,sigmas,pi_vector),log_likelihood(test_data,mu_vector,sigmas,pi_vector)])

    labels, mu_vector, sigmas, pi_vector, Q =  EM_algo(tr_data,n_states)
    result_gm = np.array([log_likelihood(tr_data,mu_vector,sigmas,pi_vector),log_likelihood(test_data,mu_vector,sigmas,pi_vector)])

    A, mus, sigmas, pi = EM_HMM(tr_data,n_states)
    result_hmm = np.array([log_likelihood_HMM(tr_data,n_states,A,sigmas,mus,pi),log_likelihood_HMM(test_data,n_states,A,sigmas,mus,pi)])

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
    log_condi_p = np.array([multivariate_normal.logpdf(data[0], mus[i], sigmas[i]) for i in range(K)])
    log_p = np.log(p)

    for k in range(K):
        # V(0,k) = max_q0 P(u0 | q0) P(q1 = k | q0) P(q0) 
        (log_v[k,0], etat) = max((log_condi_p[s] + np.log(A[s,k]) + log_p[s], s) for s in range(K))
        state[k,0] = etat

    # Propagation
    for t in range(1, T-1):
        log_condi_p = (np.array([multivariate_normal.logpdf(data[t], mus[i], sigmas[i]) for i in range(K)]))
        for k in range(K):
            # V(t,k) = max_qt P(ut | qt) P(qt+1 = k | qt) V(t-1, qt)
            (log_v[k,t], etat) = max((log_condi_p[s] + np.log(A[s,k]) + log_v[s, t-1], s) for s in range(K))
            state[k,t] = etat

    # Last step
    log_condi_p = (np.array([multivariate_normal.logpdf(data[T-1], mus[i], sigmas[i]) for i in range(K)]))
    for k in range(K):
        # V(T-1,k) = max_qT-1 P(uT-1 | qT-1) V(T-2, qT-1)
        (log_v[k,T-1], etat) = max((log_condi_p[s] + log_v[s, T-2], s) for s in range(K))
        state[k,T-1] = etat

    # Compute the viterbi decoding
    q_best = np.argmax(log_v[:,T-1])
    out[T-1] = q_best
    for t in range(T-2,-1,-1):
        out[t] = state[out[t+1],t]

    if plot:
        # Plot
        plt.figure(figsize=(9,9))
        plt.plot(data[out==0,0], data[out==0,1], 'rx', label="q_t = 0")
        plt.plot(data[out==1,0], data[out==1,1], 'bx', label="q_t = 1")
        plt.plot(data[out==2,0], data[out==2,1], 'gx', label="q_t = 2")
        plt.plot(data[out==3,0], data[out==3,1], 'cx', label="q_t = 3")
        plt.plot(mus[:,0], mus[:,1],'ko', label="Cluster centers")

        plt.axis([-15,15,-15,15])
        plt.title("Viterbi decoding on training dataset")
        plt.legend(numpoints=1)
        plt.gcf().savefig("Report/Figures/question8.png")
        plt.close(plt.gcf())

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

    (tr_data, test_data) = importdata(args.training_set,args.test_set)

    print "---------------"
    print "IMPORTED DATA SHAPE" 
    print "training set : " + str(tr_data.shape)
    print "test set : " +str(test_data.shape)
    print "---------------"

    for q in questions:
        if (q==2):
            print '------------------'
            print 'Computing answer 2'
            question2(tr_data,test_data,n_clusters)
            continue
        if (q==4):
            print '------------------'
            print 'Computing answer 4'
            question4(tr_data,n_clusters)
            continue
        if (q==5):
            print '------------------'
            print 'Computing answer 5'
            question5(tr_data,test_data,n_clusters)
            continue
        if (q==6):
            print '------------------'
            print 'Computing answer 6'
            question6(tr_data,test_data,n_clusters)
            continue
        if (q==8):
            print '------------------'
            print 'Computing answer 8'
            A, mus, sigmas, pi = EM_HMM(tr_data,n_clusters) 
            r_tr = update_r(tr_data,sigmas,mus,A,pi)
            p = update_pi0(r_tr)
            out = viterbi(tr_data, sigmas, mus, A.T, p, n_clusters, True)
            continue
        if (q==9):
            print '------------------'
            print 'Computing answer 9'
            # Plot marginal probability on test set
            A, mus, sigmas, pi = EM_HMM(tr_data,n_clusters) 
            r_test = update_r(test_data,sigmas,mus,A,pi)
            plot_r(r_test.T, 100)
            continue
        if (q==10):
            print '------------------'
            print 'Computing answer 10'
            A, mus, sigmas, pi = EM_HMM(tr_data,n_clusters) 
            r_test = update_r(test_data,sigmas,mus,A,pi)
            state = np.argmax(r_test, axis = 1)
            state = state[0:100]
            # Plot the most likely state according to marginal probability
            plt.title("State using marginal probability")
            plt.xlabel("Time")
            plt.ylabel("State")
            width = 1/1.5
            x = np.arange(100)
            plt.bar(x[state==0], 1+state[state==0], width, color='r', label="q_t = 0")
            plt.bar(x[state==1], 1+state[state==1], width, color='b', label="q_t = 1")
            plt.bar(x[state==2], 1+state[state==2], width, color='g', label="q_t = 2")
            plt.bar(x[state==3], 1+state[state==3], width, color='c', label="q_t = 3")
            plt.legend(numpoints=1)
            plt.gcf().savefig("Report/Figures/question10.png")
            plt.close(plt.gcf())
            continue
        if (q==11):
            print '------------------'      
            print 'Computing answer 11'
            A, mus, sigmas, pi = EM_HMM(tr_data,n_clusters) 
            r = update_r(tr_data,sigmas,mus,A,pi)
            p = update_pi0(r)
            out = viterbi(test_data, sigmas, mus, A.T, p, n_clusters, False)
            state = out[0:100]
            # Plot the most likely state using Viterbi decoding
            plt.title("State using Viterbi decoding")
            plt.xlabel("Time")
            plt.ylabel("State")
            width = 1/1.5
            x = np.arange(100)
            plt.bar(x[state==0], 1+state[state==0], width, color='r', label="q_t = 0")
            plt.bar(x[state==1], 1+state[state==1], width, color='b', label="q_t = 1")
            plt.bar(x[state==2], 1+state[state==2], width, color='g', label="q_t = 2")
            plt.bar(x[state==3], 1+state[state==3], width, color='c', label="q_t = 3")
            plt.legend(numpoints=1)
            plt.gcf().savefig("Report/Figures/question11.png")
            plt.close(plt.gcf())
            continue

        print "Warning : unsupported question -> %d" % q
        continue

    plt.show()

if __name__ == '__main__':
    main()
