import numpy as np

CONVERGENCE_THR = 0.01


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
