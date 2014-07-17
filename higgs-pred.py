#!/usr/bin/python
# make prediction 
import sys
import numpy as np
import reduce_features_forum as rff
# add path of xgboost python module
sys.path.append('../../python/')
import xgboost as xgb

# path to where the data lies
dpath = 'data'

modelfile = 'higgs.model'
outfile = 'higgs.pred.csv'
# make top 15% as positive 
threshold_ratio = 0.142
# load in training data, directly use numpy
#dtest = np.loadtxt( dpath+'/test.csv', delimiter=',', skiprows=1 )
dtest = np.genfromtxt( dpath+'/test.csv', delimiter=',', names=True )
dtest=rff.reduce_angles(dtest)

data   = dtest[:,1:]
idx = dtest[:,0]

print ('finish loading from csv ')
xgmat = xgb.DMatrix( data, missing = -999.0 )
bst = xgb.Booster({'nthread':16})
bst.load_model( modelfile )
res = bst.predict( xgmat )

rank=np.argsort(res)

# write out predictions
ntop = int( threshold_ratio * len(res ) )


fo = open(outfile, 'w')
fo.write('EventId,RankOrder,Class\n')
s=rank[-ntop:]

y_pred=np.array([0]*len(rank))
y_pred[s]=1
print "ntop:",ntop
reorder={}
lb={}
for k in range(len(rank)):
    ind=rank[k]
    v=y_pred[ind]
    if v==1:
        lb[ind] = 's'
    else:
        lb[ind] = 'b'
    reorder[ind]=k
for ind in range(len(reorder)):
    fo.write('%d,%d,%s\n' % ( idx[ind],reorder[ind]+1, lb[ind] ) )
fo.close()

print ('finished writing into prediction file')



