#Libraries
from __future__ import division
cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from sklearn.linear_model import RANSACRegressor
from operator import itemgetter
import time

np.import_array()

#Defining functions
#Ransac function
def ransac_polyfit(np.ndarray[np.npy_int64, ndim=1] x,
                   np.ndarray[np.npy_int64, ndim=1] y,
                   np.npy_intp order, 
                   np.npy_float32 t, 
                   np.npy_float32 n=0.8,
                   np.npy_intp k=100,
                   np.npy_float32 f=0.9):
    
    cdef np.npy_intp kk, i, ransac_control
    cdef np.npy_float64 thiserr, besterr = -1.0
    cdef np.ndarray[np.npy_int64, ndim=1] maybeinliers
    cdef np.ndarray[np.npy_bool, ndim=1] alsoinliers
    cdef np.ndarray[np.npy_float64, ndim=1] bestfit = np.array([0.0]), bestfitderi = np.array([0.0]), maybemodel, res_th, bettermodel
    cdef list polyderi
    
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=int(n*len(x)))
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        polyderi = []
        for i in range(order):
            polyderi.append(maybemodel[i]*(order-i))
        res_th = t / np.cos(np.arctan(np.polyval(np.array(polyderi),x)))
        
        alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < res_th
        if sum(alsoinliers) > len(x)*f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            polyderi = []
            for i in range(order):
                polyderi.append(maybemodel[i]*(order-i))
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
            if (thiserr < besterr) or (besterr == -1.0):
                bestfit = bettermodel
                besterr = thiserr
                bestfitderi = np.array(polyderi)
    
    if (besterr == -1.0):
        ransac_control = 0
        return ransac_control, bestfit, bestfitderi
    else:
        ransac_control = 1
        return ransac_control, bestfit, bestfitderi


#ddbscan_inner function    
def ddbscaninner(np.ndarray[np.npy_intp, ndim=2] data,
                   np.ndarray[np.uint8_t, ndim=1] is_core,
                   np.ndarray[object, ndim=1] neighborhoods,
                   np.ndarray[object, ndim=1] neighborhoods2,
                   np.ndarray[np.npy_intp, ndim=1] labels):
    
    #Parameters of the ddbscan_inner
    
    cdef np.npy_float32 acc_th = 0.80    #Accuracy of the RANSAC to save one point of the cluster for the directional search
    cdef np.npy_int32 points_th = 70   #Minimum number of points to test the ransac
    cdef np.npy_float32 t = 5  #The thickness of the track
    cdef np.npy_float32 time_threshold = 600 #Maximum ammount of time that the directional search is enabled for each cluster (marked as infinite to see the results)
    
    #Beginning of the algorithm - DBSCAN check part
    
    cdef np.npy_intp i, label_num = 0, v, l1, ransac_control, control, j
    cdef np.npy_int64 pts0, pts1
    cdef np.npy_float32 accuracy, t1, t2
    cdef np.ndarray[np.npy_intp, ndim=1] neighb, moment_lab, inliers
    cdef np.ndarray[np.npy_intp, ndim=2] la_aux = np.zeros([labels.shape[0], 2], dtype = np.intp)
    cdef list acc = [], clu_stra = [], length = [], auxiliar_points = [], clu_coordinates, stack = []
    cdef np.ndarray[np.npy_int64, ndim=2] uniques
    cdef np.ndarray[np.npy_int64, ndim=1] x, y
    cdef np.ndarray[np.npy_float64, ndim = 2] vet_aux
    cdef np.ndarray[np.npy_float64, ndim = 1] fit_model, fit_deri, res_th
    cdef np.ndarray[np.npy_bool, ndim=1] lt, inliers_bool
    cdef object ransac


    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1:
                            stack.append(v)

            if len(stack) == 0:
                break
            i = stack[len(stack)-1]
            del(stack[len(stack)-1])
        #Ransac part
        if sum(labels==label_num) > points_th:
            x = data[labels==label_num][:,0]
            y = data[labels==label_num][:,1]
            if (np.median(np.abs(y - np.median(y))) == 0):
                ransac = RANSACRegressor(min_samples=0.8, residual_threshold = 0.1)
                ransac.fit(np.expand_dims(x, axis=1), y)
            else:
                ransac = RANSACRegressor(min_samples=0.8)
                ransac.fit(np.expand_dims(x, axis=1), y)
            accuracy = sum(ransac.inlier_mask_)/len(y)
            if accuracy > acc_th:
                clu_stra.append(label_num)
                acc.append(accuracy)
                length.append(sum(labels==label_num))
        
        
        label_num += 1

    
    #End of DBSCAN loop - check if directional part is viable
    print("Clusters found in DBSCAN: %d" %(len(set(labels)) - (1 if -1 in labels else 0)))
    if (len(clu_stra) == 0):
        #If no cluster has a good fit model, the output will be the same of the DBSCAN
        la_aux[:,0] = np.copy(labels)
        return la_aux
    else:
    #If any cluster has a good fit model, it'll be marked from the worst fitted cluster to the best, each of them respecting the accuracy threshold
        vet_aux = np.zeros([len(clu_stra),3])
        vet_aux[:,0] = np.asarray(clu_stra)
        vet_aux[:,1] = np.asarray(acc)
        vet_aux[:,2] = np.asarray(length)
        vet_aux = np.asarray(sorted(vet_aux,key=itemgetter(1),reverse=1))
        if (sum(vet_aux[:,1]==1) > 1):
            l1 = sum(vet_aux[:,1]==1)
            vet_aux[0:l1,:] = np.asarray(sorted(vet_aux[0:l1,:],key=itemgetter(2),reverse=1))
        for u in range(len(clu_stra)):
            lt = (labels==vet_aux[u][0])*is_core
            auxiliar_points.append(np.where(lt)[0][0])
            print("The point %d has been assigned as part of a good fit" %(np.where(lt)[0][0]))
        
        
        #Now the clusterization will begin from zero with directionality enabled for the clusters that have a good fit model
        label_num = 0
        labels = np.full(data.shape[0], -1, dtype=np.intp)
        stack = []
        for i in auxiliar_points:
            if labels[i] != -1 or not is_core[i]:
                continue
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if is_core[i]:
                        neighb = neighborhoods[i]
                        for i in range(neighb.shape[0]):
                            v = neighb[i]
                            if labels[v] == -1:
                                stack.append(v)

                if len(stack) == 0:
                    break
                i = stack[len(stack)-1]
                del(stack[len(stack)-1])
            
            #Now that the provisional cluster has been found, directional search begins
            if sum(labels==label_num) > points_th:
                
                #Taking unique points to use on the ransac
                clu_coordinates = [tuple(row) for row in data[labels==label_num]] 
                uniques = np.unique(clu_coordinates,axis=0)
                x = uniques[:,0]
                y = uniques[:,1]
                
                
                #RANSAC fit
                ransac_control, fit_model, fit_deri = ransac_polyfit(x,y,order=1, t = t)
                counter = 1
                #Adding new points to the cluster (If the fit_model output is None, then no model was found)
                if ransac_control == 1:
                    control = 1
                    pts1 = 0
                    t1 = time.time()
                    while True:

                        #Filling stack list with possible new points to be added (start point)
                        pts0 = pts1
                        moment_lab = np.where(labels==label_num)[0]
                        stack = []
                        for j in moment_lab:
                            #if is_core[j]:
                            neig2 = neighborhoods2[j]
                            for k in neig2:
                                if labels[k] != label_num:
                                #if (labels[k] != label_num) or 1:
                                    stack.append(k)
                        stack = np.unique(stack).tolist()
                        if len(stack) == 0:
                            break
                        
                        res_th = t / np.cos(np.arctan(np.polyval(fit_deri,data[:,0])))
                        inliers_bool = np.abs(np.polyval(fit_model, data[:,0])-data[:,1]) < res_th
                        inliers = np.where(inliers_bool)[0]
                            

                        i = stack[len(stack)-1]
                        #Adding the inliers points from stack list and filling stack with more possible points
                        while True:
                            if i in inliers and (labels[i] != label_num):
                            #if i in inliers:
                                labels[i] = label_num
                                #if is_core[i]:
                                neig2 = neighborhoods2[i]
                                for i in range(neig2.shape[0]):
                                    v = neig2[i]
                                    if labels[v] != label_num:
                                        stack.append(v)

                            if len(stack) == 0:
                                break
                            i = stack[len(stack)-1]
                            del(stack[len(stack)-1])

                        #Checking current cluster for possible fit model update

                        clu_coordinates = [tuple(row) for row in data[labels==label_num]] 
                        uniques = np.unique(clu_coordinates,axis=0)
                        x = uniques[:,0]
                        y = uniques[:,1]


                        #Updating the ransac model
                        if control == 1:
                            ransac_control, fit_model, fit_deri = ransac_polyfit(x,y,order=1, t = t)
                        else:
                            ransac_control, fit_model, fit_deri = ransac_polyfit(x,y,order=3, t = t)
                        pts1 = sum(labels==label_num)
                        #Stop criteria - time
                        t2 = time.time()
                        if (t2 - t1) > time_threshold:
                            break
                        #Stop criteria - When there is no more point to be added or if the fit is not good anymore
                        counter = counter + 1
                        if (pts1 == pts0) or (ransac_control != 1):
                            if control == 0:
                                print('The cluster %d' %(label_num) + ' needed %d attempts' %(counter))
                                break
                            else:
                                ransac_control, fit_model, fit_deri = ransac_polyfit(x,y,order=3, t = t)
                                control = 0
                                if ransac_control != 1:
                                    break
                
                
            #label_num += 1
            if sum(labels==label_num) > 29:
                label_num += 1
            else:
                labels[labels==label_num] = len(data)
            

        #Now that the clusters with good fit models were found, the rest of the data will be clustered with the standard DBSCAN logic
        for i in range(labels.shape[0]):
            if labels[i] != -1 or not is_core[i]:  
                continue
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if is_core[i]:     #Only core points are expanded
                        neighb = neighborhoods[i]
                        for i in range(neighb.shape[0]):
                            v = neighb[i]
                            if labels[v] == -1:
                                stack.append(v)

                if len(stack) == 0:
                    break
                i = stack[len(stack)-1]
                del(stack[len(stack)-1])

            if sum(labels==label_num) > 29:
                label_num += 1
            else:
                labels[labels==label_num] = len(data)
        #False clusters remotion
        labels[labels==len(data)] = -1

        #Clusterization has finished, now the clusters found with ransac fit model will be marked
        la_aux[:,0] = np.copy(labels)
        la_aux[auxiliar_points,1] = 1

        return la_aux
                
        
        