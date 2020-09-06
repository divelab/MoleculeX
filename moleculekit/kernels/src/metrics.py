from sklearn.metrics import auc,  precision_recall_curve, roc_auc_score

def RMSE(y, pred):
    mse = ((y-pred)**2).mean()
    return mse**0.5

def MAE(y, pred):
    mae = (((y-pred)**2)**0.5).mean()
    return mae

def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def PRC(y_true, y_pred):
    prc = prc_auc(y_true, y_pred)
    return prc

def ROC(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    return roc