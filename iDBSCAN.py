# -*- coding: utf-8 -*-
"""
iDBSCAN: Iterative Density-Based Spatial Clustering of Applications with Noise
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def idbscan(X, iterative = 4, vector_eps = [2.26, 3.5, 2.8, 6], vector_min_samples = [2, 30, 6, 2], cuts = [900, 150], flag_noise = True, flag_plot_noise = 0):
    """
    Parameters
    ----------
    
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
            
    iterative : int, optional
        How many time the DBSCAN will run.
        0 - to run naive DBSCAN using the first parameter on eps and min_samples array
        1 - to look for and save only the 'long' tracks
        2 - to look for and save only the 'medium' tracks
        3 - to look for and save only the 'small' tracks
        4 - to look for and save all the three types
        12 - to look for and save the 'long' and the 'medium' tracks
    
    vector_eps : float array(1x4), optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
        
    vector_min_samples : int array (1x4), optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        
    cuts : int array (1x2), optional
        The min number of sampels that the clusters need to be considerer 
        'long', 'medium' or 'small'.
    
    flag_noise : Boolean
        If it is TRUE the noise removing loop is done, when it is FALSE 
        no noise removing is made.
    
    Returns
    -------
    core_samples : array [n_core_samples]
        Indices of core samples.
    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.
    tag : array [n_samples]
        tag for each point.
            - Noisy samples are given the label '-1'
            - 1st iteration tracks are given the label '1'
            - 2nd iteration tracks are given the label '2'
            - 3rd iteration tracks are given the label '3'

    """

    ## - - - - -
    Index              = np.arange(0,np.shape(X)[0],dtype=int)
    Fcluster           = (-1)+np.zeros(np.shape(X)[0],dtype=int)
    Flabel             = np.empty(np.shape(X)[0],dtype=int)
    Flabel[:]          = -1
    auxClu             = -1
    # - - - - - -
    #vector_eps         = [2.26, 3.5, 2.8, 6]
    #vector_min_samples = [2, 30, 6, 2]
    auxIti             = - 1
    ## - - - - -
    indgood = np.ones(np.shape(X)[0],dtype=bool)
    
    if (iterative >= 0) & (flag_noise == True) :

        auxIti += 1
        db      = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(X)
        labels  = db.labels_
        indgood = db.labels_ != -1

        
        if flag_plot_noise == 1:
            f,ax = plt.subplots(1,2,figsize=(40,20))
            ax[0].scatter(X[:, 1], X[:, 0], alpha = 0.5, s = 10, linewidths = 0)
            ax[0].set_title('Edges after pedestal substraction')
            ax[1].scatter(X[indgood, 1], X[indgood, 0], alpha = 0.5, s = 10, linewidths = 0)
            ax[1].set_title('Edges after removing "noise"')

        ## ----- Salve the clusters and labels
        Fcluster[db.labels_ == -1] = -1
        Flabel[db.labels_   == -1] = -1 # 'n' = noise points

        if iterative == 0:
            ## ----- Salve the clusters and labels
            Fcluster        = labels
            Flabel[indgood] = 3 # 'n' = noise points
    else:
        auxIti += 1

    if iterative >= 1:
        #print('indgood: ', sum(indgood))

        Xnew      = X[indgood,:]
        #print('Xnew: ', Xnew)
        indicenew = np.where(indgood == True)[0]

        auxIti    += 1
        db         = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(Xnew)
        labels     = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Find the Long clusters

        clusters = [Xnew[labels == i] for i in range(n_clusters_)]

        lenClu = np.zeros(n_clusters_,)
        for i in range(0,n_clusters_):
            lenClu[i] = np.size(clusters[i])
        clusterI = (np.where(lenClu > cuts[0]))[0]

        if iterative == 1 or iterative == 4 or iterative == 12: # To salve ONLY the Long Clusters or 4 to all
            ## ----- Salve the clusters and labels
            for i in clusterI:
                auxClu+=1
                indice = Index[indicenew[labels == i]]
                Fcluster[indice] = auxClu
                Flabel[indice] = 1 # 'l' = Long tracks

    if iterative >= 2:

        indgood2 = ~np.in1d(db.labels_, clusterI)
        Xnew2 = Xnew[indgood2,:]
        if np.size(Xnew2) > 1:
            indicenew2 = np.where(indgood2 == True)[0]

            auxIti+=1
            db = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(Xnew2)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            clusters = [Xnew2[labels == i] for i in range(n_clusters_)]

            lenClu = np.zeros(n_clusters_,)
            for i in range(0,n_clusters_):
                lenClu[i] = np.size(clusters[i])
            clusterI = (np.where(lenClu > cuts[1]))[0]

            if iterative == 2 or iterative == 4 or iterative == 12: # To salve ONLY the Medium Clusters or 4 to all
                ## ----- Salve the clusters and labels
                for i in clusterI:
                    auxClu+=1
                    indice = Index[indicenew[indicenew2[labels == i]]]
                    Fcluster[indice] = auxClu
                    Flabel[indice] = 2 # 'c' = Curly tracks


    if iterative >= 3:

        indgood3 = ~np.in1d(db.labels_, clusterI)
        Xnew3 = Xnew2[indgood3,:]
        if np.size(Xnew3) > 1:
            indicenew3 = np.where(indgood3 == True)[0]

            auxIti+=1
            db = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(Xnew3)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            clusters = [Xnew3[labels == i] for i in range(n_clusters_)]

            if iterative == 3 or iterative == 4: # To salve ONLY the Small Clusters or 4 to all
                ## ----- Salve the clusters and labels
                for j in range(0,n_clusters_):
                    auxClu+=1
                    indice = Index[indicenew[indicenew2[indicenew3[labels == j]]]]
                    Fcluster[indice] = auxClu
                    Flabel[indice] = 3 # 'c' = others tracks


    return Fcluster, np.where(Fcluster != -1)[0], Flabel


class iDBSCAN:
    
    def __init__(self, iterative = 4, vector_eps = [2.26, 3.5, 2.8, 6], vector_min_samples = [2, 30, 6, 2], cuts = [900, 150], flag_noise = True, flag_plot_noise = 0):
        self.iterative = iterative
        self.vector_eps = vector_eps
        self.vector_min_samples = vector_min_samples
        self.cuts = cuts
        self.flag_noise = flag_noise
        self.flag_plot_noise = flag_plot_noise

    def fit(self, X):
        
        clust = idbscan(X, self.iterative, self.vector_eps, self.vector_min_samples, self.cuts, self.flag_noise, self.flag_plot_noise)
        self.labels_, self.core_sample_indices_, self.tag_  = clust
        
        return self
