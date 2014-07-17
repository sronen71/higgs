#!/usr/bin/python
# this is the example script to use xgboost to train 
import inspect
import os
import sys
import numpy as np
import random
import reduce_features_forum as rff

def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10.
    ams=np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))
    return ams


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
    t=sum((solution==1)*(prediction==1))
    f=sum((solution==0)*(prediction==1))
    truePos  = sum(weights[(solution == 1) * (prediction == 1)])
    falsePos = sum(weights[(solution == 0) * (prediction == 1)])

    print "t,f,tw,fw,s/b",t,f,truePos,falsePos,truePos/falsePos
    return truePos, falsePos

def  ratio_score(ratio,rank,w,target):
    ncut=int(ratio*len(rank))
    s=rank[-ncut:]
    y_pred=np.array([0]*len(rank))
    y_pred[s]=1
    truePos, falsePos = get_rates(y_pred, target, w)
    ams=AMS(truePos, falsePos)
    return ams    

# add path of xgboost python module
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../python")

sys.path.append(code_path)

import xgboost as xgb

test_size = 550000

# path to where the data lies
dpath = 'data'

# load in training data, directly use numpy
#dtrain = np.loadtxt( dpath+'/training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
dtrain=np.genfromtxt(dpath+'/training.csv',delimiter=',',names=True,converters={32: lambda x:int(x=='s'.encode('utf-8'))})
print ('finish loading from csv ')
dtrain=rff.reduce_angles(dtrain)

label  = dtrain[:,-1]
data   = dtrain[:,1:-2]

print data.shape
#samp=random.sample(xrange(len(data)),25000)

samp=range(len(data))
data=data[samp,:]
label=label[samp]
# rescale weight to make it same as test set
#weight = dtrain[samp,31] * float(test_size) / len(label)
weight=dtrain[samp,-2]
#weight*=sum(dtrain[:,31])/sum(weight)

sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

# print weight statistics 
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix( data, label=label, missing = -999.0, weight=weight )

# setup parameters for xgboost
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['bst:eta'] = 0.1  # 0.1
#param['bst:gamma']=0.0203
param['bst:max_depth'] = 6
param['eval_metric'] = 'auc'
param['silent'] = 1
#param['subsample']=0.5203 # none 


#param['best_min_child_weight']=0.9888 #1
param['nthread'] = 2

# you can directly throw param in, though we want to watch multiple metrics here 
plst = list(param.items())+[('eval_metric', 'ams@0.15')]

watchlist = [ (xgmat,'train') ]
# boost num_round tres
num_round = 120
print ('loading data end, start to boost trees')
bst = xgb.train( plst, xgmat, num_round, watchlist );
# save out model
xgmat_in = xgb.DMatrix(data, missing=-999.0)

ypred=bst.predict( xgmat_in )
rank=np.argsort(ypred)


cuts=np.arange(0.05,0.25,0.002)
best_AMS=0
best_ratio=0
for ratio in cuts:
    score = ratio_score(ratio,rank,weight,label)
    print ratio,score
    if score>best_AMS:
        best_AMS=score
        best_ratio=ratio

bst.save_model('higgs.model')
print "best:",best_ratio,best_AMS
print 'finish training'
