import numpy as np
from sklearn.metrics import roc_auc_score

def statistical_parity(preds, X, protected_class):
    sp = preds[X[protected_class]==1].sum()/preds[X[protected_class]==1].shape[0] - preds[~(X[protected_class]==1)].sum()/preds[~(X[protected_class]==1)].shape[0]
    return sp

def equalized_odds(preds, X, y, protected_class):
    pos1 = preds[(X[protected_class]==1) & (y==1)].sum()/preds[(X[protected_class]==1) & (y==1)].shape[0]
    neg1 = preds[((X[protected_class]==0)) & (y==1)].sum()/preds[((X[protected_class]==0)) & (y==1)].shape[0]
    pos0 = preds[(X[protected_class]==1) & (y==0)].sum()/preds[(X[protected_class]==1) & (y==0)].shape[0]
    neg0 = preds[((X[protected_class]==0)) & (y==0)].sum()/preds[((X[protected_class]==0)) & (y==0)].shape[0]
    eo = abs(pos1 - neg1) + abs(pos0 - neg0)
    return eo

def average_odds_difference(preds, X, y, protected_class):
    tpr_u = preds[(X[protected_class]==0) & (y==1)].sum()/(y==1).sum() # u is unprivileged
    tpr_p = preds[(X[protected_class]==1) & (y==1)].sum()/(y==1).sum() # p is privileged
    fpr_u = preds[(X[protected_class]==0) & (y==0)].sum()/(y==0).sum()
    fpr_p = preds[(X[protected_class]==1) & (y==0)].sum()/(y==0).sum()
    aod = ((fpr_u - fpr_p) + (tpr_u - tpr_p)) * 0.5
    return aod

def equal_opportunity_difference(preds, X, y, protected_class):
    tpr_u = preds[(X[protected_class]==0) & (y==1)].sum()/(y==1).sum() # u is unprivileged
    tpr_p = preds[(X[protected_class]==1) & (y==1)].sum()/(y==1).sum() # p is privileged
    eod = tpr_u - tpr_p
    return eod

def disparate_impact(preds, X, protected_class):
    num = preds[X[protected_class]==0].sum()/(X[protected_class]==0).sum()
    den = preds[X[protected_class]==1].sum()/(X[protected_class]==1).sum()           
    if den == 0:
        return np.nan
    di = num / den
    return di

def abroca(proba_preds, X, y, protected_class):
    priviled_proba_preds = proba_preds[X[protected_class]==1]
    non_priviled_proba_preds = proba_preds[X[protected_class]==0]
    privileged_y = y[X[protected_class]==1]
    non_privileged_y = y[X[protected_class]==0]
    privileged_auc = roc_auc_score(privileged_y, priviled_proba_preds)
    non_privileged_auc = roc_auc_score(non_privileged_y, non_priviled_proba_preds)
    return abs(privileged_auc - non_privileged_auc)
