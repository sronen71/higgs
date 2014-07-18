import math
import numpy as np

def angle_invert(angle) :
    return (
        -999 * (angle == -999) +
        (angle != -999) * (-angle) 
    )

def reduce_angles(X) :
    """ This function works in-place!"""
    
    delta_angle = 'PRI_tau_phi'
    for angle in ['PRI_lep_phi', 'PRI_met_phi'] :
        X[angle ] = X[angle]- X[delta_angle]
    for angle in ['PRI_jet_leading_phi', 'PRI_jet_subleading_phi'] :
        X[angle] = (X[angle]- X[delta_angle]) * (X[angle] != -999) +(-999) * (X[angle] == -999)
        
   
    #for angle in ['PRI_tau_eta', 'PRI_lep_eta', 'PRI_jet_leading_eta', 'PRI_jet_subleading_eta'] :
    #    X[angle] = angle_invert(X[angle], invert_mask)
   
    #for angle in ['PRI_lep_phi', 'PRI_met_phi','PRI_jet_leading_phi',
    #    'PRI_jet_subleading_phi']:
    #    X[angle] = angle_invert(X[angle], invert_mask)
  
    nam=X.dtype.names
    print X.shape
    for k in range(len(nam)):
        if nam[k]==delta_angle:
            kd=k
    
    #Xp1=X
    #for angle in ['PRI_tau_eta', 'PRI_lep_eta', 'PRI_jet_leading_eta', 'PRI_jet_subleading_eta'] :
    #    Xp1[angle] = -X[angle] # z symmetry together with roational symmetry implies parity in x-y as well
    #X=np.concatenate((X,Xp1))

    print X.shape
    Xadd=np.zeros((len(X),4))
    ii=0
    for angle in ['PRI_lep_phi', 'PRI_met_phi','PRI_jet_leading_phi',
        'PRI_jet_subleading_phi']: 
        Xadd[:,ii]=(X[angle]!=-999)*np.cos(X[angle])-999*(X[angle]==-999)
        X[angle]=(X[angle]!=-999)*np.sin(X[angle])-999*(X[angle]==-999)
        ii+=1



    X=np.array(X.tolist())

    X=np.delete(X,kd,1)
    X=np.hstack((X[:,0:1],Xadd,X[:,1:]))


    #X=X[1:5000,:]
    print X.shape
    return X

def main():
    dpath='data'
    dtrain=np.genfromtxt(dpath+'/training.csv',delimiter=',',names=True)
    dtest=np.genfromtxt(dpath+'/test.csv',delimiter=',',names=True)
    np.savetxt(dpath+'/training_red.csv',dtrain,delimiter=',')
    np.savetxt(dpath+'/test_red.csv',dtest,delimiter=',')


if __name__ == "__main__":
    main()
