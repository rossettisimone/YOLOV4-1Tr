# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:19:11 2021

@author: Utente
"""

import pandas as pd
import re
import numpy as np

## Load the dataset and replace the zeroes with faked vals
## Mimicking slow human motion between 0 and 1 and computing the centroid
## accordingly

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
# load and build dataset as a array
def make_dataset(df, num_subject):
    step = 6
    mstep = 4
    sC = 2
    datax = np.zeros((len(df),num_subject * mstep))
    dataC = np.zeros((len(datax),num_subject * sC))   
    dataset = np.zeros((len(datax), num_subject * step)) 
    cols = df.columns
    for kk in range(len(df)):
        # print(kk)
        ss = 0
        uu = 0
        tt = 0
        
        for jj in range(3):
              U = df[cols[jj]].iloc[kk]
              # U0 = U[0].split('[')
              # U1 = U0[-1].split(']')[0]
              U = U.replace('[','')
              U = U.replace(']','')
              U = U.replace(',','')
              U = U.split(' ')
              U = [i for i in U if is_number(i)]
              
              outList  = [np.float(x) for x in U]
              datax[kk,ss:ss+4] = outList
              dataC[kk,uu:uu+2] = (np.array(outList[0:2]) + np.array(outList[2:4]))*0.5
              dataset[kk,tt:tt+6] = np.concatenate([datax[kk,ss:ss+4],dataC[kk,uu:uu+2]],axis = 0)
              ss = ss + 4
              uu = uu + 2
              tt = tt + 6
    return dataset, datax, dataC




## Build a faked motion sequence + centroids
def make_faked_sequence(lengthX):
    step = 6
    mstep = 4
    xyzw = np.random.rand(1,mstep)
    faked_vals = np.zeros((lengthX,6))
    faked_vals[0,0:mstep] = xyzw
    faked_vals[0,mstep:step] = (faked_vals[0, 0:2] + faked_vals[0, 2:mstep])*0.5
    for ii in range(1,lengthX):
        u = np.random.rand(1,mstep)/(step*np.log(lengthX))
        if  np.sum(xyzw + u < [1,1,1,1]) == mstep  & np.sum((xyzw + u > [0, 0, 0, 0])) == mstep:
            faked_vals[ii,0:mstep] =  xyzw + u
            xyzw = xyzw +u
        else:  
            faked_vals[ii,0:mstep] =  xyzw - u
            xyzw = xyzw - u
        cc = (faked_vals[ii, 0:2] + faked_vals[ii, 2:mstep])*0.5
        faked_vals[ii,mstep:step] = cc 
    return faked_vals
        
    # 
    
### replace all zeros with  faked vectors of structure (x0, y0, x1, y1, cx, cy )   
def replace_0(dataset, datax, dataC, num_subject):
    row, cols = dataset.shape
    step  = 6
    mstep = 4
    sC  = 2
    aspect = np.arange(0,cols, step)
    for rr, jj in enumerate(aspect):
        
        X = dataset[:,jj : jj+step].copy()
        ttx = np.argwhere(X[:,0]==0)
        if len(ttx) > 0:
            ttx = ttx[:, 0]
            first = np.array([ttx[0]], dtype = np.int)
            last = np.array([ttx[-1]], dtype = np.int)
        
            # Compute middle regions if there are any
            dt = np.diff(ttx)
            wdt  = np.argwhere(dt > 1)
            if wdt.shape[0] > 0:
                middle = np.zeros(2*len(wdt), dtype = np.int)
                kk = 0 
                for val in wdt:
                    # print(val[0])
                    ww1 = ttx[val[0]]
                    ww2 = ttx[val[0]+1]
                    middle[kk] = ww1
                    middle[kk+1] = ww2
                    kk = kk+2
                    
            else:
                middle = np.array([], dtype = np.int)
            
            regions = np.concatenate([first, middle, last], axis = 0)
            s = 0
            for ii in range( len(regions)//2):
      
               lengthFake = (regions[s+1]-regions[s])+1
               fakedVector = make_faked_sequence(lengthFake)
               X[regions[s]: regions[s+1]+1, :] = fakedVector
               s = s+2
               
            # print('jj', jj)        
            
            dataset[:,jj : jj + step ]  = X   
            
            kj = rr * mstep
            datax[:,kj : kj + mstep] = X[:, 0:mstep]
            
            kj = rr * sC
            dataC[:, kj: kj +sC] = X[:,-2: step]
    return dataset, datax, dataC

def loadD():
     df = pd.read_csv('subjects.csv') 
     num_subject = 4
     dataset, datax, dataC = make_dataset(df, num_subject)
     dataset, datax, dataC = replace_0(dataset, datax, dataC, num_subject )
     return dataset, datax, dataC
    

 
    
 