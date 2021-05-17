# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:22:10 2021

@author: Fiora
"""
############# bipartite G
import networkx as nx
from   networkx.algorithms import bipartite
import itertools
import numpy as np
import scipy.optimize
from scipy import sparse


def bipartite_matching(list_corr):

    n = len(list_corr)
    H = list_corr.copy()
    W = np.zeros((n+1,n+1))
    W[1:n+1, 1:n+1] = H
    
    W = 1-W
    labels = np.arange(1, 2*n+1, 1)    
    
    # Initialise graph
    Gg = nx.Graph()
    top_nodes = [x for x in labels[:2*n//2]]
    bottom_nodes =  [x for x in labels[-2*n//2:]]
    Gg.add_nodes_from(top_nodes, bipartite=0)
    Gg.add_nodes_from(bottom_nodes, bipartite=1)
    #####
    #Assignment
    M = np.zeros((n+1, n+1))
   
    for ii, nt in enumerate(top_nodes):
        ii = ii+1
        for jj, nb in enumerate(bottom_nodes):
            jj = jj+1
            w = W[ii,jj]
            Gg.add_edge(nt, nb, weight = w)
            M[ii,jj] = w
            
            
    Mi = M[1:M.shape[0],1:M.shape[1]]
    
            
    # print("Edges in Gg: ", Gg.edges(data=True))
    # print("Nodes in Gg: ", Gg.nodes(data=True))
    
    min_match =bipartite.matching.minimum_weight_full_matching(Gg, top_nodes, "weight")
    Gt = np.zeros((n,n))
    for key, val in min_match.items():
        # print(key,val)
        if key <= n:
           Gt[key-1,val-n-1] = 1
    
    
    return Gt, Mi
