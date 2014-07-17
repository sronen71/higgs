import math
import numpy as np

def delta_angle_norm(a, b) :
    delta = (a - b)
    delta += (delta < -math.pi) * 2 * math.pi
    delta -= (delta > math.pi) * 2 * math.pi
    return delta

def angle_invert(angle, invert_mask) :
    return (
        -999 * (angle == -999) +
        (angle != -999) * (
            angle * (invert_mask == False) +
            (-angle) * (invert_mask == True)
        )
    )

def reduce_angles(X) :
    """ This function works in-place!"""
    
    delta_angle = 'PRI_tau_phi'
    for angle in ['PRI_lep_phi', 'PRI_met_phi'] :
        X['%s' % angle ] = delta_angle_norm(X[angle], X[delta_angle])
    for angle in ['PRI_jet_leading_phi', 'PRI_jet_subleading_phi'] :
        X['%s'% angle] = (
            delta_angle_norm(X[angle], X[delta_angle]) * (X[angle] != -999) +
            (-999) * (X[angle] == -999)
        )
   
    invert_mask = X['PRI_tau_eta'] < 0
    for angle in ['PRI_tau_eta', 'PRI_lep_eta', 'PRI_jet_leading_eta', 'PRI_jet_subleading_eta'] :
        X[angle] = angle_invert(X[angle], invert_mask)
   
    invert_mask = X['PRI_lep_phi'] < 0
    for angle in ['PRI_lep_phi', 'PRI_met_phi','PRI_jet_leading_phi',
        'PRI_jet_subleading_phi']:
        X[angle] = angle_invert(X[angle], invert_mask)
  
    nam=X.dtype.names
    print X.shape
    for k in range(len(nam)):
        if nam[k]==delta_angle:
            kd=k
    X=np.array(X.tolist())
    X=np.delete(X,kd,1)
    return X

def main():
    dpath='data'
    dtrain=np.genfromtxt(dpath+'/training.csv',delimiter=',',names=True)
    dtest=np.genfromtxt(dpath+'/test.csv',delimiter=',',names=True)
    np.savetxt(dpath+'/training_red.csv',dtrain,delimiter=',')
    np.savetxt(dpath+'/test_red.csv',dtest,delimiter=',')


if __name__ == "__main__":
    main()
