# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sparse
import itertools
import pandas as pd

class CMIM:
    def __init__(self):
        return
    def entropy(self,n_elements,*X):
        """calculates joint entropy between len(*X) random variables"""
        return(np.sum(-p * np.log2(p) if p > 0 else 0 for p in
            (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
                for classes in itertools.product(*[(0,1) for x in range(0,n_elements)]))))
    def cond_inf(self,Mat):
        """calculates conditional mutual information. Could probably stand a rework to make more general"""
        if isinstance(Mat,sparse.csc_matrix):
            Mat=Mat.todense()
        if Mat.shape[1]==2:
            return(-self.entropy(2,Mat[:,0],Mat[:,1])+self.entropy(1,Mat[:,1])+self.entropy(1,Mat[:,0]))
        else:
            return(self.entropy(2,Mat[:,0],Mat[:,2])-self.entropy(1,Mat[:,2])-self.entropy(3,Mat[:,0],Mat[:,2],Mat[:,1])+self.entropy(2,Mat[:,1],Mat[:,2]))
        
    def fit(self,X,y):
        """Sequentially selects explanatory variables until no variables add information conditional on the already chosen features.  Only works for binary random variables"""
        if isinstance(X,pd.DataFrame):
            X=np.array(X)
        if isinstance(y,pd.Series):
            y=np.array(y)
        #I wish numpy handled (n,) arrays more gracefully
        y=y.reshape(len(y),1)
        #concatenate target/feature vectors
        if isinstance(X,sparse.csc_matrix):
            Data=sparse.hstack([sparse.csc_matrix(y),X],format="csc")
        else:
            Data=np.hstack([y,X])
    
    #computes I(Y;Xn) for every feature n
        featuredict={x:[self.cond_inf(Data[:,(0,x)]),0] for x in range(1,Data.shape[1])}
    #initialize feature selection
        selected_features={}
        #begin feature selection;
        for K in range(1,X.shape[1]+1):
            smin=0
            for feature in range(1,X.shape[1]):
                while featuredict[feature][0]>smin and featuredict[feature][1]<(K-1):
                    featuredict[feature][1]+=1
                    featuredict[feature][0]=min(featuredict[feature][0],self.cond_inf(Data[:,(0,feature,selected_features[featuredict[feature][1]])]))
                
                if featuredict[feature][0]>smin:
               
                    smin=featuredict[feature][0]
                    #make sure algorithm terminates properly when additional variables add no new information.  Recheck to make sure this isn't causing algorithm to terminate early
                    if feature in selected_features.values():
                        self.selected_index=selected_features.values()
                        return
                    selected_features[K]=feature
            
        self.selected_index=selected_features.values()
        return
    def transform(self,X):
        """Remove uninformative variables"""
        if isinstance(X,pd.DataFrame):
            X=np.array(X)
        return(X[:,self.selected_index])
    def fit_transform(self,X,y):
        """Identify and remove uninformative variables"""
        if isinstance(X,pd.DataFrame):
            X=np.array(X)
        if isinstance(y,pd.Series):
            y=np.array(y)
        self.fit(X,y)
        return(X[:,self.selected_index])