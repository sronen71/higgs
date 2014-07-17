
import inspect
from scipy.optimize import minimize
import os
import sys
# add path of xgboost python module
code_path = os.path.join(
            os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../python")
sys.path.append(code_path)

import time
import csv
import sys
import numpy as np
import scipy as sp
import xgboost as xgb
import sklearn.cross_validation as cv
from scipy.optimize import minimize_scalar

def ratio_score(ratio,rank,w,target):
    ncut=int(ratio*len(rank))
    s=rank[-ncut:]
    y_pred=np.array([0]*len(rank))
    y_pred[s]=1
    truePos, falsePos = get_rates(y_pred, target, w)
    return  AMS(truePos, falsePos)


def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))


def get_rates(prediction, solution, weights):
    '''
    Returns the true and false positive rates.
    This assumes that:
        label 's' corresponds to 1 (int)
        label 'b' corresponds to 0 (int)
    '''
    assert prediction.size == solution.size
    assert prediction.size == weights.size

    # Compute sum of weights for true and false positives
    truePos  = sum(weights[(solution == 1) * (prediction == 1)])
    falsePos = sum(weights[(solution == 0) * (prediction == 1)])

    return truePos, falsePos


def get_training_data(training_file):
    '''
    Loads training data.
    '''
    data = list(csv.reader(open(training_file, "rb"), delimiter=','))
    X       = np.array([map(float, row[1:-2]) for row in data[1:]])
    labels  = np.array([int(row[-1] == 's') for row in data[1:]])
    weights = np.array([float(row[-2]) for row in data[1:]])
    sub=len(labels)
    #sub=25000
    X=X[1:sub,:]
    labels=labels[1:sub]
    weights=weights[1:sub]
    return X, labels, weights


def estimate_performance_xgboost(x0, X,labels,weights,num_round, folds):
    '''
    Cross validation for XGBoost performance 
    '''

    kf = cv.KFold(labels.size, n_folds=folds)
    param={}
    param['objective'] = 'binary:logitraw'
    param['bst:eta'] = x0[0]
    param['bst:gamma']= x0[1]
    param['bst:max_depth'] = 6
    param['bst:min_child_weightd'] = x0[2]
    param['bst:subsample']=x0[3]    
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 2
    param['disp']=True    
    print param

    all_AMS=[]

    time_start=time.time()
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]

        # Rescale weights so that their sum is the same as for the entire training set
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))
      
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])

        param['scale_pos_weight'] = sum_wneg / sum_wpos
            # you can directly throw param in, though we want to watch multiple metrics here 
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)

        best_AMS=0
        rank=np.argsort(y_out)
     
        cuts=np.arange(0.05,0.4,0.005)
        for ratio in cuts:
            AMS = ratio_score(ratio,rank,w_test,y_test)
            if AMS>best_AMS:
                best_AMS=AMS
                best_ratio=ratio
        print "Best AMS =", best_AMS,best_ratio
        all_AMS.append(best_AMS)

    #print "------------------------------------------------------"
    time_end=time.time()
    
    mean_AMS=sp.mean(all_AMS)
    print "Mean AMS = ",mean_AMS," std ",sp.std(all_AMS)," time ",time_end-time_start
               
    #print "------------------------------------------------------"
    return -mean_AMS 

def main():

    num_round = 120 # Number of boosted trees
    folds = 3 # Folds for CV
    training_file="data/training.csv"
    X, labels, weights = get_training_data(training_file)
    #estimate_performance_xgboost(param,X,labels,weights, num_round, folds)

    #x0=(0.1,0.02,1,0.5) # eta,gamma,min_child_weight,subsample,
    #x0=(0.1,0.1,0.98,0.7)
    res = minimize(estimate_performance_xgboost, x0, args=(X,labels,weights,num_round,folds),method='nelder-mead',  options={'ftol':1e-2,'xtol': 1e-2 ,'maxdev': 1000,'disp': True})
    print "Best:",res,res.x

if __name__ == "__main__":
    main()
