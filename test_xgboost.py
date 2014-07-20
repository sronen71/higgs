import matplotlib.pyplot as plt
import inspect
import os
import sys
import reduce_features_sym as rfs
from sklearn.metrics import auc,roc_curve,roc_auc_score
import sklearn.metrics as metrics
from sklearn.svm import SVC
import sklearn as sk
from scipy.interpolate import interp1d

from sklearn.decomposition import PCA
from sklearn.lda import LDA
import math


# add path of xgboost python module
code_path = os.path.join(
            os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../python")
sys.path.append(code_path)


import csv
import sys
import numpy as np
import scipy as sp
import xgboost as xgb
import sklearn.cross_validation as cv
import reduce_features_forum as rff

def fullWeight():
    return 411691.8 

def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10
    #print "wegihts:",s,b,(s+b)
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


def get_training_data(data,num_jets,has_mass):
    '''
    Loads training data.
    '''
    select=data['PRI_jet_num']==num_jets
    if has_mass:
        select=select & (data['DER_mass_MMC']!=-999)
    else:
        select= select &  (data['DER_mass_MMC']==-999) 
    
    print "#jets, has_mass,#selected",num_jets,has_mass,sum(select)
    data=data[select]
    data=rfs.reduce_angles(data)
    data=np.array(data.tolist())

    X       = np.array([map(float, row[1:-2]) for row in data])
    labels  = np.array([int(row[-1] == 1) for row in data])
    weights = np.array([float(row[-2]) for row in data])

    none=(X==-999).nonzero()
    col=np.unique(none[1])
    print col
    for c in col:
        print sum(X[:,c]==-999),X.shape
    X=np.delete(X,col,1)
    print X.shape    
    #X=X[:,0:13] # Include only first 13 features (all DER)    
    return X, labels, weights


def doPCA(X,labels):
    target_names=["background","signal"]  
    y=labels      
    pc=PCA(n_components=3)
    pc.fit(X)
    #print pc.components_[0:5,:]
    print pc.explained_variance_ratio_
    print("PCA explained: ",sum(pc.explained_variance_ratio_))
    X_r=pc.transform(X)
    
    plt.figure()
    for c, i, target_name in zip("rg", [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('PCA of Higgs data')

    #plt.figure()
    #for i in range(3):
    #    plt.plot(range(len(pc.components_[i,:])), pc.components_[i,:],label=str(i) )   
    #plt.legend()


    plt.show()    


def boots(bst,X_test,y_test,w_test,cutrange):
    a=np.zeros(len(cutrange))
    for k in range(10):
        boot=np.random.choice(len(y_test),len(y_test))
        by_test=y_test[boot]
        bw_test=w_test[boot]
        xgmat_test = xgb.DMatrix(X_test[boot,:], missing=-999.0)
        y_out = bst.predict(xgmat_test)
        ysort=np.sort(y_out)

        eval_metric=1-roc_auc_score(by_test,y_out,sample_weight=bw_test,average=None)
        fpr,tpr,thresholds=roc_curve(by_test,y_out,sample_weight=bw_test)
        # Explore changing threshold_ratio and compute AMS
        sum_wpos_test = sum(w_test[y_test == 1])
        sum_wneg_test = sum(w_test[y_test == 0])
        
        for i,cut in enumerate(cutrange):
            ntop=round(len(ysort)*cut)
            th=ysort[-ntop]
            k=np.argmax(thresholds[::-1]>=th)
            k=len(thresholds)-1-k
            a[i]+=AMS(tpr[k]*sum_wpos_test,fpr[k]*sum_wneg_test)
    a=a/10.0
    """    
    plt.figure()
    plt.plot(cutrange,a)
    plt.xlabel('cut')
    plt.ylabel('ams')
    plt.show()
    """
    max_ams=max(a)
    pos=np.where(a==max_ams)[0]
    popratio=cutrange[pos]
 

    print "ratio,best_Ams,metric",popratio,max_ams
    return a 

def estimate_performance_xgboost(data, param, num_round, folds,jets,has_mass):
    # Load training data

    X, labels, weights = get_training_data(data,jets,has_mass)
 
    sum_wpos = sum(weights[labels == 1])
    sum_wneg = sum(weights[labels == 0])
    print "initial s,b",sum_wpos,sum_wneg
    #doPCA(X,labels)    
       
    #print labels.size
    # Cross validate
    kf = cv.StratifiedKFold(labels, n_folds=folds,shuffle=True,random_state=5) #1
    npoints  = 20
    # Dictionary to store all the AMSs
    all_AMS = {}
    for curr in range(npoints):
        all_AMS[curr] = []
    cutoffs  = np.arange(0.002,0.3,0.002)
    ascore=np.zeros(len(cutoffs))
    test_dic={}
    fold=0
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]
        
        # Rescale weights s  that their sum is the same as for the entire training set
        w_train *= (fullWeight() / sum(w_train))
        worg_test=w_test.copy()

        w_test  *= (fullWeight() / sum(w_test))

        sum_pos=sum(y_train)
        sum_neg=sum(1-y_train)
        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])
        param['scale_pos_weight'] = sum_wneg / sum_wpos

        plst = param.items()#+[('eval_metric', 'ams@0.15')]
        watchlist = []#[(xgmat, 'train')]
        
        # xgboost

        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)
        bst = xgb.train(plst, xgmat, num_round, watchlist)
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)
     
        # SVM
        #sc=sk.preprocessing.StandardScaler()
        #X_train=sc.fit_transform(X_train)
        #clf=SVC(verbose=True)
        #clf.fit(X_train,y_train,sample_weight=w_train)
        #X_test=sc.transform(X_test)
        #y_out=clf.decision_function(X_test)
        


        ###########
        eval_metric=1-roc_auc_score(y_test,y_out,sample_weight=w_test,average=None)
        fpr,tpr,thresholds=roc_curve(y_test,y_out,sample_weight=w_test)
        # Explore changing threshold_ratio and compute AMS
        sum_wpos_test = sum(w_test[y_test == 1])
        sum_wneg_test = sum(w_test[y_test == 0])
        a=np.zeros((len(thresholds)))
        b=np.zeros((len(thresholds)))
        s=np.zeros((len(thresholds)))
        for i in range(len(thresholds)):
            s[i]=sum_wpos_test*tpr[i]
            b[i]=sum_wneg_test*fpr[i]
            a[i]=AMS(s[i],b[i])
       
        """
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.figure()
        plt.plot(thresholds,a)
        plt.xlabel('threshold')
        plt.ylabel('ams')
        plt.show()
        """
        max_ams=max(a)
        pos=np.where(a==max_ams)[0]
        popratio=sum(y_out>=thresholds[pos])/float(len(y_out))
        
        print "th,best_Ams,s,b,metric",popratio,max_ams,s[pos],b[pos],eval_metric
        ind=np.argsort(y_out)
        ind=ind[::-1]
        
        test_dic[fold]={'y':y_test[ind],'w':worg_test[ind],'popratio':popratio}
        fold+=1
        score=boots(bst,X_test,y_test,w_test,cutoffs)
        ascore+=score
    print "------------------------------------------------------"
    ascore=ascore/len(kf) 

    return test_dic

def main():
    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logistic'
    param['bst:eta'] = 0.1
    param['bst:max_depth'] =6  # 6 (oK; 4,5)
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 2

    num_round = 150 # Number of boosted trees
    folds = 3 # Folds for CV

    data = np.genfromtxt('data/training.csv',delimiter=',',names=True,converters={32: lambda x:int(x=='s'.encode('utf-8'))})


    cases={}
    nn=0
    totalWeight=[0]*folds
    categories=[(0,True),(1,True),(2,True),(2,False),(3,True)]
    for categ in categories:
        jets=categ[0]
        has_mass=categ[1]
        test_dic=estimate_performance_xgboost(data, param, num_round, folds,jets,has_mass)
        for fold in range(folds):
          target=test_dic[fold]['y']
          weights=test_dic[fold]['w']
          ratio_guess=test_dic[fold]['popratio']
          totalWeight[fold]+=sum(weights)
          cases[(nn,fold)]={'target':target,'weights':weights,'ratio_guess':ratio_guess} # sorted, most signal like first
        nn+=1
    for key  in cases:
        cases[key]['weights']=cases[key]['weights']*fullWeight()/totalWeight[key[1]]
    print "optimizing mix..."       
    #guess= [0.156,0.15,0.425,0.125,0.253]
    #guess= [0.141,0.149,0.409,0.145,0.173]
    #guess=[0.111,0.140,0.377,0.087,0.209]
    guess=[0.106,0.127,0.375,0.036,0.235]
    for fold in range(folds):
        fcases={}
        for key in cases:
            if key[1]==fold:
                fcases[key[0]]=cases[key]
        print "guess AMS",-evalcomb(guess,fcases)
        res=sp.optimize.basinhopping(evalcomb,x0=guess,T=0.5,
            stepsize=0.05,minimizer_kwargs={"args": (fcases,)})
        print "optimized:",res.x,-res.fun
    
def evalcomb(cuts,fcases):
    b=0
    s=0
    for i in range(len(fcases)):
        x=len(fcases[i]['weights'])*cuts[i]
        if x>0:
            top=math.floor(x)
            b+=sum(fcases[i]['weights'][:top]*(1-fcases[i]['target'][:top]))
            s+=sum(fcases[i]['weights'][:top]*fcases[i]['target'][:top])
            if top<len(fcases[i]['weights']):
                b+=fcases[i]['weights'][top]*(1-fcases[i]['target'][top])*(x-top)
                s+=fcases[i]['weights'][top]*fcases[i]['target'][top]*(x-top)

    return -AMS(s,b)

if __name__ == "__main__":
    main()
