import numpy as np

def getContours(xbox,ybox):
    
    ymi = []
    yma = []
    xmi = []
    xma = []
    xu = np.unique(xbox)
    for i in xu:
        if ybox[xbox == i].size > 0:
            mi = np.min(ybox[xbox == i])
            ymi.append(mi)
            xmi.append(i)
            ma = np.max(ybox[xbox == i])
            yma.append(ma)
            xma.append(i)
            
    xri = np.concatenate((xmi, xma[::-1],[xmi[0]]))
    yri = np.concatenate((ymi, yma[::-1], [ymi[0]]))
    return xri,yri

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def inputFile(numrun,filedir,formattype):

    ## Loading image for analysis
    # otherwise use 'mid'
    if formattype == 'h5':
        imagename     = 'histogram_Run'
    else:
        imagename     = 'histograms_Run'

    filename = '%s%s%s.root' % (filedir,imagename,numrun)
    return filename

def findedges(ybox,xbox,rebin):
    from skimage.measure import find_contours
    from numpy import zeros
    from scipy.ndimage import uniform_filter
    
    rescale = int(2048/rebin)
    mm = zeros([rescale,rescale],dtype=int)
    mm[ybox,xbox]=10000
    mm = uniform_filter(mm, size=5)
    contours = find_contours(mm, 0.9)
    return contours

def noisereductor(edges,rescale,meancut=0.35):
    tpx = 10

    for i in range(rescale):
        for j in range(2):
            edges[j,i]=0
            edges[i,j]=0
            edges[rescale-1-j,i]=0
            edges[i,rescale-1-j]=0
    
    for i in range(1,rescale-2):
        for j in range(1,rescale-2):
            spx = edges[i,j]
            frame = edges[i-1:i+2,j-1:j+2]
            mpx = (np.sum(frame)-spx)/8.
            # put very noisy pixels at the average value of the frame around
            if np.abs(spx - mpx) > tpx :
                edges[i,j] = mpx
            # filter the pixels with no sufficient energy around
            if (mpx < 0.35):
                edges[i,j] = 0
            # require at least two neighbors above threshold
            neighbors = len(frame[frame>0])
            if neighbors<3:
                edges[i,j] = 0
    return edges
