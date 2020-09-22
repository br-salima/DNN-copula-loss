import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import dtcwt
from copulas.multivariate import GaussianMultivariate
from scipy import stats
from copulas.visualization import hist_1d
import pandas as pd

from copulas.univariate import GammaUnivariate


##############################RR######################################
# Read RGB image 
for line1 in open("./data_NSS_test/img_namesNSS_l.txt", "r"):
    path= '/mnt/e/documents/NSSADNN_IQA-master/data_NSS_test/distortedimage_R/'+ line1
    path = path.strip("\n")
    #print(path)
    lenaR = cv2.imread(path) 
    #print(lena.shape)
    lena = cv2.imread(path ,0)
    #print(lena)
    x=1
    lena = lena.astype('float32')
    # Parse the lena file and rescale to be in the range (0,1]
    lena = lena /225
    lena -= np.mean(lena, axis=0)
    lena /= np.std(lena, axis=0)
    #s=np.shape(lena)
    #print (lena.shape)
    nlevels=3
    orientation=6
    transform = dtcwt.Transform2d('near_sym_b','qshift_b')
    t = transform.forward(lena, nlevels)

    p = []
    for i in range(nlevels) :
        for slice_idx in range(orientation):

            p.append(np.abs(t.highpasses[i][:,:,slice_idx]))
            #print(np.abs(t.highpasses[i][:,:,slice_idx]).shape)
    XR = np.array(p)
    #print (XL[23].shape)   


    ########################################RL###################################################################
    pathr=('/mnt/e/documents/NSSADNN_IQA-master/data_NSS_test/distortedimage_L/'+line1)
    pathr=pathr.strip("\n")
    lenaR = cv2.imread(pathr,0) 
    #print(lenaR.shape)
    x=1
    lenaR = lenaR.astype('float32')
    # Parse the lena file and rescale to be in the range (0,1]
    lenaR = lenaR /225
    lenaR -= np.mean(lenaR, axis=0)
    lenaR /= np.std(lenaR, axis=0)
    #s=np.shape(lena)
    #print (lena.shape)
    
    transform = dtcwt.Transform2d('near_sym_b','qshift_b')
    t = transform.forward(lenaR, nlevels)

    #############################abs#####################################

    pR = []
    for i in range(nlevels) :
        for slice_idx in range(orientation):

            pR.append(np.abs(t.highpasses[i][:,:,slice_idx]))
            
            #print(np.abs(t.highpasses[i][:,:,slice_idx]).shape)
    XL = np.array(pR)

    #print(type(XR))
    ########################################copula#########################################

    conv = []
    scales = []
    As = []

    for i in range(len(XL)) :
        sbl=XL[i].flatten()
        sbr=XR[i].flatten()
        conc = np.empty((sbl.shape[0],2))
        conc[:,0] = sbr[:]
        conc[:,1] = sbl[:]
        #print(conc.shape)
        copula = GaussianMultivariate(distribution=GammaUnivariate)
        copula.fit(conc)
        XX = np.array(copula.to_dict()['covariance'])
        xx= XX.flatten()
        conv.append(xx) 
        UNI=copula.to_dict()['univariates'][0] #avec les sbr de droite
        scales.append(UNI["scale"])
        As.append(UNI["a"])
    A = np.array(As)
    B = np.array(scales)
    C = np.array(conv)
    distribution = np.empty((A.shape[0],6))
    distribution[:,0] = A
    distribution[:,1] = B
    distribution[:,2:6] = C    # distribution[shape scale c[0] c[1] c[2] c[3] ] pour chaque sb 
    #dis=np.reshape(distribution, -1)
    #distribution.shape
    #print(str(distribution))
    print(str(distribution.shape[0]))
    print(str(distribution.shape[1]))
    with open('./data_NSS_test/parametre_NSS.txt', 'a+') as f:
        f.seek(1)
        for j in range(distribution.shape[1]) :
            for i in range(distribution.shape[0]) :
                f.write("%s\n" % distribution[i,j])
                print(distribution[i,j])

                
                #print('x=1%d',x=x+1)
        #print(distribution[j,i])
        f.close()