import numpy as np
from sklearn.metrics import roc_auc_score

def statistical_parity(preds, X, protected_class):
    '''
    ref:
    Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012, January). 
    Fairness through awareness. In Proceedings of the 3rd innovations in theoretical 
    computer science conference (pp. 214-226).
    '''
    sp = preds[X[protected_class]==1].sum()/preds[X[protected_class]==1].shape[0] - preds[~(X[protected_class]==1)].sum()/preds[~(X[protected_class]==1)].shape[0]
    return sp

def equalized_odds(preds, X, y, protected_class):
    '''
    ref:
    Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. 
    Advances in neural information processing systems, 29.
    '''
    pos1 = preds[(X[protected_class]==1) & (y==1)].sum()/preds[(X[protected_class]==1) & (y==1)].shape[0]
    neg1 = preds[((X[protected_class]==0)) & (y==1)].sum()/preds[((X[protected_class]==0)) & (y==1)].shape[0]
    pos0 = preds[(X[protected_class]==1) & (y==0)].sum()/preds[(X[protected_class]==1) & (y==0)].shape[0]
    neg0 = preds[((X[protected_class]==0)) & (y==0)].sum()/preds[((X[protected_class]==0)) & (y==0)].shape[0]
    eo = abs(pos1 - neg1) + abs(pos0 - neg0)
    return eo
