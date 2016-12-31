import sys
import itertools
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
_legend_size = 10

#def plot_data_train_test_labeled(tr_data, test_data, labels_tr, labels_test, title_complement = "",comp1="", comp2=""):
#    f, axarr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14.5,8))
#    col = plot_labeled_data("Training data " + title_complement + comp1,tr_data,labels_tr,axarr[0])
#    axarr[0].legend(prop={'size':_legend_size})
#    plot_labeled_data("Test data "+ title_complement + comp2,test_data,labels_test,axarr[1],col)
#    axarr[1].legend(prop={'size':_legend_size})
#    f.tight_layout()
#    return f, axarr, col

# def plot_GM_distrib(mu_vector,sigmas, plot=plt, col = []): # col for colors, if no colors specified, then chosen randomly
#     assert len(mu_vector) == len(sigmas), "in plot_GM_distrib: number of clusters not matching"

#     color_array = np.random.rand(len(mu_vector),3,1)

#     if (len(col) > 1):
#         colors = itertools.cycle(col)
#     else:
#         colors = itertools.cycle(color_array)

#     for k in range(len(mu_vector)):
#         eigval, eigvec = np.linalg.eig(sigmas[k])

#         ### plot the center
#         plot.scatter(mu_vector[k][0],mu_vector[k][1],marker="o",c="k",s=40)

#         ### plot the ellipsoid
#         t = np.linspace(0,2*math.pi,100)
#         points = np.zeros((len(t),2))

#         if eigval[0]>eigval[1]:
#             maxi=0
#             mini=1
#         else:
#             maxi=1
#             mini=0

#         angle=-math.atan2(eigvec[maxi][1],eigvec[maxi][0])
#         R1 = np.array([math.cos(angle),math.sin(angle)])
#         R2 = np.array([-math.sin(angle),math.cos(angle)])
#         R = np.vstack((R1,R2))

#         for i in range(len(t)):
#             ellipse_x_r=math.cos(t[i])*math.sqrt(eigval[maxi]*4.6052) # 90%-ellipse
#             ellipse_y_r=math.sin(t[i])*math.sqrt(eigval[mini]*4.6052) # 90%-ellipse
#             r_ellipse = np.dot(np.transpose(np.array([ellipse_x_r,ellipse_y_r])),R)
#             r_ellipse = np.transpose(r_ellipse)
#             points[i,:] = mu_vector[k] + r_ellipse

#         plot.plot(points[:,0],points[:,1],color=next(colors))


def plot_labeled_data(title,points,labels,cluster_centers,output_name = "output", col = []):
    plt.title(title)

    list_labels = np.unique(labels)
    color_array = np.random.rand(len(list_labels),3,1)

    if (len(col) > 1):
        colors = itertools.cycle(col)
    else:
        colors = itertools.cycle(color_array)

    for label in list_labels:
        mask_label = (labels == label)
        if (np.sum(mask_label) > 0):
            plt.scatter(points[mask_label,0], points[mask_label,1],label = "Cluster %d" % label,color=next(colors))
        else:
            next(colors)

    plt.plot(cluster_centers[:,0], cluster_centers[:,1],'ko', label="Cluster centers")

    plt.legend(prop={'size':_legend_size})
    plt.gcf().savefig("Report/Figures/%s.eps" % output_name)
    plt.close(plt.gcf())

def plot_log_likelihood_evolution(l_tr,l_test,output_name="output"):
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

    plt.gcf().savefig("Report/Figures/%s.eps" % output_name)
    plt.close(plt.gcf())

def plot_r(probas, N, output_name="output"):
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

    plt.gcf().savefig("Report/Figures/%s.eps" % output_name)
    plt.close(plt.gcf())