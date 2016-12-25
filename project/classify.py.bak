##########################################
#### Probabilistic Graphical Models ######
############## Homework 2 ################
##########################################
# Sylvain TRUONG and Matthieu KIRCHMEYER #
##########################################
##########################################

# Call: python classify.py path_to_train path_to_test -m [1,2,3]
# several methods can be called at the same time (plots will then appear on the same figure)
# Call example: python classify.py path/to/sets/classificationA.train path/to/sets/classificationA.test -m 1 2 3 4

import sys
import itertools
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math

_legend_size = 10

def parseArguments():
	parser = argparse.ArgumentParser(description="Learn 2D-data-point clustering models and test them out")
	
	parser.add_argument("training_set",
		help="path_to_training_set")

	parser.add_argument("test_set",
		help="path_to_test_set")

	parser.add_argument('-n', '-n_clusters', 
		type=int, help="Specify the target number of clusters (defaults to 4)")

	parser.add_argument('-m', '-method', nargs='*', 
		type=int, help="Specify the models to learn: 1/k-means 2/Isotropic Gaussian Mixture 3/General Gaussian Mixture (defaults to k-means)")

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

def plot_data_train_test_labeled(tr_data, test_data, labels_tr, labels_test, title_complement = ""):
	f, axarr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14.5,8))
	col = plot_labeled_data("Training data "+title_complement,tr_data,labels_tr,axarr[0])
	axarr[0].legend(prop={'size':_legend_size})
	plot_labeled_data("Test data "+title_complement,test_data,labels_test,axarr[1],col)
	axarr[1].legend(prop={'size':_legend_size})
	f.tight_layout()
	return f, axarr, col

#def plot_data(title,points,plot = plt):
#	if (plot != plt):
#		plot.set_title(title)
#	plot.scatter(points[:,0], points[:,1],label="Data samples")

def plot_GM_distrib(mu_vector,sigmas, plot=plt, col = []):
	assert len(mu_vector) == len(sigmas), "in plot_GM_distrib: number of clusters not matching"

	color_array = np.random.rand(len(mu_vector),3,1)

	if (len(col) > 1):
		colors = itertools.cycle(col)
	else:
		colors = itertools.cycle(color_array)

	for k in range(len(mu_vector)):
		eigval, eigvec = np.linalg.eig(sigmas[k].T)
		
		### debug
		#print "Cluster %d" % k
		#print eigval
		#print eigvec
		### end debug

		### plot the center
		plot.scatter(mu_vector[k][0],mu_vector[k][1],marker="o",c="k",s=40)


		### plot the ellipsoid
		for i in range(len(eigvec)):
			eigvec[i,:] = eigvec[i]/np.linalg.norm(eigvec[i])
		t = np.linspace(0,2*math.pi,100)
		points = np.zeros((len(t),2))
		for i in range(len(t)):
			points[i,:] = mu_vector[k] + eigvec[1]*math.cos(t[i])*math.sqrt(eigval[1]) - eigvec[0]*math.sin(t[i])*math.sqrt(eigval[0])
		plot.plot(points[:,0],points[:,1],color=next(colors))

#def plot_GM_distrib(mu_vector,sigmas, plot=plt):
#	assert len(mu_vector) == len(sigmas), "in plot_GM_distrib: number of clusters not matching"
#	axes = plot.axis()
#
#	x = np.linspace(axes[0], axes[1], 100)
#	y = np.linspace(axes[2], axes[3], 100)
#
#	x,y = np.meshgrid(x,y)
#
#	for k in range(len(mu_vector)):
#		plot.contour(x,y,normal_law_2D(x,y,mu_vector[k],sigmas[k])-0.9,[0],colors='k')

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
	convergence_criteria = 0.01

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
		lam = (2*math.pi*len(data_points))**(-1)*np.trace(np.dot(data_points-mu_vector[k],(data_points-mu_vector[k]).T))
		sigmas_update[k,:,:] = lam * np.eye(data_points.shape[1])
	return sigmas_update


#######################
## EXPECTATION STEP ##
#######################
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
		
	convergence_thr = 0.001
	counter = 0
	max_iter = 1000

	# while convergence criterion not met
	while counter < max_iter:
		print "\riteration : %d" % counter,
		sys.stdout.flush()
		# Maximization step
		sigmas = update_sigmas(data_points,Q,mu_vector)
		pi_vector = update_pi(Q)
		mu_vector = update_mu(data_points,Q)
		
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

	return labels, mu_vector, sigmas, pi_vector

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
		
	convergence_thr = 0.001
	counter = 0
	max_iter = 1000

	# while convergence criterion not met
	while counter < max_iter:
		print "\riteration : %d" % counter,
		sys.stdout.flush()
		# Maximization step
		sigmas = update_sigmas_isotropic(data_points,Q,mu_vector)
		pi_vector = update_pi(Q)
		mu_vector = update_mu(data_points,Q)
		
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

	return labels, mu_vector, sigmas, pi_vector



def assign_label_gm(data_points,mu_vector,sigmas,pi_vector):
	Q = update_Q(data_points,mu_vector,sigmas,pi_vector)
	# issue the labels
	labels = np.zeros(len(Q))
	for i in range(len(data_points)):
		labels[i] = np.argmax(Q[i,:])
	return labels

##################################################
###################### MAIN ######################
##################################################

def main():
	args = parseArguments()

	(tr_data, test_data) = importdata(args.training_set,args.test_set)

	print "---------------"
	print "IMPORTED DATA SHAPE"	
	print "training set : " + str(tr_data.shape)
	print "test set : " +str(test_data.shape)
	print "---------------"

	#f, axarr = plot_data_train_test(tr_data,test_data)

	if args.m is None:
		methods = [1]
	else:
		methods = args.m

	if args.n 	is None:
		n_clusters = 4
	else:
		n_clusters = args.n
	
	print "Attempting to fit %d clusters" % n_clusters

	methods = list(set(list(methods))) #to remove duplicates

	for method_index in methods:
		assert (method_index > 0 and method_index < 4), "Unsupported method index: %d" % method_index
		# K-means
		if (method_index == 1):
			labels_tr, centroids = kmeans(tr_data,n_clusters)
			
			# Apply on test set
			labels_test = np.zeros(len(test_data))
			for k in range(len(test_data)):
				labels_test[k] = assign_label_kmeans(test_data[k],centroids)

			f,axarr = plot_data_train_test_labeled(tr_data, test_data, labels_tr, labels_test, "(k-means)")
			continue

		# Isotropic Gaussian Mixture
		if (method_index == 2):
			labels_tr, mu_vector, sigmas, pi_vector = EM_algo_isotropic(tr_data,n_clusters)

			# Apply on test set
			labels_test = assign_label_gm(test_data,mu_vector,sigmas,pi_vector)
			
			f,axarr,col = plot_data_train_test_labeled(tr_data, test_data, labels_tr, labels_test, "(Isotropic GM)")
			plot_GM_distrib(mu_vector,sigmas, axarr[0], col)
			plot_GM_distrib(mu_vector,sigmas, axarr[1], col)
			
			continue

		# General Gaussian Mixture
		if (method_index == 3):
			labels_tr, mu_vector, sigmas, pi_vector = EM_algo(tr_data,n_clusters)

			# Apply on test set
			labels_test = assign_label_gm(test_data,mu_vector,sigmas,pi_vector)
			
			f,axarr,col = plot_data_train_test_labeled(tr_data, test_data, labels_tr, labels_test, "(General GM)")
			plot_GM_distrib(mu_vector,sigmas, axarr[0], col)
			plot_GM_distrib(mu_vector,sigmas, axarr[1], col)
			continue


	plt.show()



if __name__ == '__main__':
    main()
