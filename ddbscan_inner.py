
from __future__ import division
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from operator import itemgetter

def ransac_polyfit(x,y,order, t, n=0.8,k=400,f=0.9):
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

    
def ddbscaninner(data, is_core, neighborhoods, neighborhoods2, labels):
    #Definitions
    loss = 0
    if loss == 0:
      loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
    elif loss == 1:
      loss_function = lambda y_true, y_pred: np.abs((y_true - y_pred)) ** 2
    
    acc_th = 0.80
    points_th = 70
    t = 5
    
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
        #If any cluster has a good fit model, it'll be marked from the best fitted cluster to the worst, each of them respecting the accuracy threshold
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
        
        #Now the clusterization will begin from zero with directionality enabled first for the clusters that have a good fit model
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

                clu_coordinates = [tuple(row) for row in data[labels==label_num]] 
                uniques = np.unique(clu_coordinates,axis=0)
                x = uniques[:,0]
                y = uniques[:,1]
                
                
                

                
                #RANSAC fit
                fit_model, fit_deri = ransac_polyfit(x,y,order=1, t = t)
                #Adding new points to the cluster
                if sum(fit_model == None) == 0:
                    control = 1
                    pts1 = 0
                    while True:

                        #Filling stack list with possible new points to be added (start point)
                        pts0 = pts1
                        moment_lab = np.where(labels==label_num)[0]
                        stack = []
                        for j in moment_lab:
                            neig2 = neighborhoods2[j]
                            for k in neig2:
                                stack.append(k)
                        stack = np.unique(stack).tolist()
                        
                        res_th = t / np.cos(np.arctan(np.polyval(fit_deri,data[:,0])))
                        inliers_bool = np.abs(np.polyval(fit_model, data[:,0])-data[:,1]) < res_th
                        inliers = np.where(inliers_bool)[0]
                            

                        i = stack[len(stack)-1]
                        #Adding the inliers points from stack list and filling stack with more possible points
                        while True:
                            if i in inliers:
                                labels[i] = label_num
                                if is_core[i]:
                                    neig2 = neighborhoods2[i]
                                    for i in range(neig2.shape[0]):
                                      v = neig2[i]
                                      if labels[v] == -1:
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
                        #Stop criteria
                        if (pts1 == pts0) or (sum(fit_model == None) != 0):
                            if control == 0:
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
        
            
            
        
        
        
        
            