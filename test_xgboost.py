import matplotlib.pyplot as plt
import inspect
import os
import sys
import reduce_features_sym as rfs
from sklearn.metrics import auc,roc_curve,roc_auc_score
import sklearn.metrics as metrics
from sklearn.svm import SVC


from sklearn.decomposition import PCA
from sklearn.lda import LDA
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


def get_training_data(training_file):
    '''
    Loads training data.
    '''
    #data = list(csv.reader(open(training_file, "rb"), delimiter=','))
    
    data = np.genfromtxt('data/training.csv',delimiter=',',names=True,converters={32: lambda x:int(x=='s'.encode('utf-8'))})

    select=(data['PRI_jet_num']==2) &  (data['DER_mass_MMC']!=-999) # 2 jets with DER_mass_MMC    
        
    data=data[select]
    data=rfs.reduce_angles(data)
    data=np.array(data.tolist())

    X       = np.array([map(float, row[1:-2]) for row in data])
    labels  = np.array([int(row[-1] == 1) for row in data])
    weights = np.array([float(row[-2]) for row in data])
    weights=weights/sum(weights)*fullWeight() # scale to full training set
    #X=X[:,0:13] # Include only first 13 features (all DER)    
    return X, labels, weights


def doPCA(X,lables):
    target_names=["background","signal"]  
    y=lables      
    pc=PCA(n_components=3)
    pc.fit(X)
    #print pc.components_[0:5,:]
    print pc.explained_variance_ratio_
    print("PCA explained: ",sum(pc.explained_variance_ratio_))
    X_r=pc.transform(X)
    
    plt.figure()
    for c, i, target_name in zip("rg", [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    #plt.legend()
    plt.title('PCA of Higgs data')

    plt.figure()
    for i in range(3):
        plt.plot(range(len(pc.components_[i,:])), pc.components_[i,:],label=str(i) )   
    plt.legend()


    #plt.show()    
def estimate_performance_xgboost(training_file, param, num_round, folds):
    '''
    Cross validation for XGBoost performance 
    '''
    # Load training data
    X, labels, weights = get_training_data(training_file)
    doPCA(X,labels)    
   
    print labels.size
    # Cross validate
    kf = cv.StratifiedKFold(labels, n_folds=folds,shuffle=True,random_state=1)
    npoints  = 20
    # Dictionary to store all the AMSs
    all_AMS = {}
    for curr in range(npoints):
        all_AMS[curr] = []
    # These are the cutoffs used for the XGBoost predictions
    cutoffs  = sp.linspace(0.05, 0.30, npoints)
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]

        # Rescale weights s  that their sum is the same as for the entire training set
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))

        sum_pos=sum(y_train)
        sum_neg=sum(1-y_train)
        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])

        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)

        # scale weight of positive examples
        param['scale_pos_weight'] = sum_wneg / sum_wpos
        # you can directly throw param in, though we want to watch multiple metrics here 
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)
        #clf=SVC(probability=True)
        #clf.fit(X_train,y_train,sample_weight=w_train)
        #y_out=clf.predict_proba(X_test)[:,1]
        
        #print y_out[1:30]

        # Construct matrix for test set
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1


        eval_metric=1-roc_auc_score(y_test,y_out,sample_weight=w_test,average=None)
        
    
        # Explore changing threshold_ratio and compute AMS
        best_AMS = -1.
        cut=0
        for curr, threshold_ratio in enumerate(cutoffs):
            y_pred = sp.zeros(len(y_out))
            ntop = int(threshold_ratio * len(rorder))
            for k, v in res:
                if rorder[k] <= ntop:
                    y_pred[k] = 1

            truePos, falsePos = get_rates(y_pred, y_test, w_test)
            this_AMS = AMS(truePos, falsePos)
            all_AMS[curr].append(this_AMS)
            if this_AMS > best_AMS:
                best_AMS = this_AMS
                cut=threshold_ratio
        print "s,b,ws,wb",sum_pos,sum_neg,sum_wpos,sum_wneg,"cut,Best AMS =", cut,best_AMS,"metric",eval_metric
    print "------------------------------------------------------"
    #for curr, cut in enumerate(cutoffs):
    #    print "Thresh = %.2f: AMS = %.4f, std = %.4f" % \
    #        (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr]))
    #print "------------------------------------------------------"


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
    estimate_performance_xgboost("data/training.csv", param, num_round, folds)


if __name__ == "__main__":
    main()
