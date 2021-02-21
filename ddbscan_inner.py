
from __future__ import division
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import time

def ransac_polyfit(x,y,order, t, n=0.8,k=100,f=0.9):
    
    besterr = np.inf
    bestfit = np.array([None])
    bestfitderi = np.array([None])
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
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                bestfitderi = np.array(polyderi)
                     
    return bestfit, bestfitderi

#Parameters of the new ransac function:
# x, y - x and y coordinates 
# order - Order of the polynomial
# t - Thickness of the track
# n - Random fraction of the data used to find the polyfit
# k - Number of tries 
# f - Accuracy of the RANSAC to consider the fit a good one
    
def ddbscaninner(data, is_core, neighborhoods, neighborhoods2, labels):
    #Definitions
    
    acc_th = 0.80    #Accuracy of the RANSAC to save one point of the cluster for the directional search
    points_th = 70   #Minimum number of points to test the ransac
    t = 5.5  #The thickness of the track
    time_threshold = np.inf #Maximum ammount of time that the directional search is enabled for each cluster (marked as infinite to see the results)
    
    #Beginning of the algorithm - DBSCAN check part
    label_num = 0
    stack = []
    clu_stra = []
    acc = []
    length = []
    
    #Loop 
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
    if len(clu_stra) == 0:
        #If no cluster has a good fit model, the output will be the same of the DBSCAN
        la_aux = np.copy(labels)
        labels = np.zeros([la_aux.shape[0],2], dtype=np.intp)
        labels[:,0] = la_aux
        return labels
    else:
        #If any cluster has a good fit model, it'll be marked from the worst fitted cluster to the best, each of them respecting the accuracy threshold
        auxiliar_points = []
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
                fit_model, fit_deri = ransac_polyfit(x,y,order=1, t = t)
                counter = 1
                #Adding new points to the cluster (If the fit_model output is None, then no model was found)
                if sum(fit_model == None) == 0:
                    control = 1
                    pts1 = 0
                    t1 = time.time()
                    while True:

                        #Filling stack list with possible new points to be added (start point)
                        pts0 = pts1
                        moment_lab = np.where(labels==label_num)[0]
                        stack = []
                        for j in moment_lab:
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
                            fit_model, fit_deri = ransac_polyfit(x,y,order=1, t = t)
                        else:
                            fit_model, fit_deri = ransac_polyfit(x,y,order=3, t = t)
                        pts1 = sum(labels==label_num)
                        #Stop criteria - time
                        t2 = time.time()
                        if (t2 - t1) > time_threshold:
                            break
                        #Stop criteria - When there is no more point to be added or if the fit is not good anymore
                        counter = counter + 1
                        if (pts1 == pts0) or (sum(fit_model == None) != 0):
                            if control == 0:
                                print('The cluster %d' %(label_num) + ' needed %d attempts' %(counter))
                                break
                            else:
                                fit_model, fit_deri = ransac_polyfit(x,y,order=3, t = t)
                                control = 0
                                if sum(fit_model == None) != 0:
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
         
            #label_num += 1
            if sum(labels==label_num) > 29:
                label_num += 1
            else:
                labels[labels==label_num] = len(data)
        
        #False clusters remotion
        labels[labels==len(data)] = -1
        
        #Clusterization has finished, now the clusters found with ransac fit model will be marked
        la_aux = np.copy(labels)
        labels = np.zeros([la_aux.shape[0],2], dtype=np.intp)
        labels[:,0] = la_aux
        labels[auxiliar_points,1] = 1
        
        return labels
        
            
            
        
        
        
        
        
            