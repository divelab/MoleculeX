from kernels import Kernel
from utils import save_model, load_model
import logging
import numpy as np
from sklearn import svm

def train_eval(config, args, X_train, Y_train, X_test=None, Y_test=None):
    
    seed = str(args['seed']) if not args['split_ready'] else ''
    model_path = "%s%s_%s.pkl"%(args['model_path'],args['dataset'],seed)
    ker = Kernel(config, args['kernel_type'])
    logging.info('Training on dataset %s...'%args['dataset'])
    logging.info('\tComputing %s kernel.'%args['kernel_type'])
    K_train = ker.fit_transform(X_train)
    
    lins = []
    nans = []
    for col in range(Y_train.shape[1]):
        Y_train_all = Y_train[:, col]
        K_train_notnan = K_train[~np.isnan(Y_train_all)][:,~np.isnan(Y_train_all)]
        Y_train_notnan = Y_train_all[~np.isnan(Y_train_all)]
        nans.append(np.isnan(Y_train_all))

        if args['metric'] in ['ROC', 'PRC']:
            logging.info('\tTraining classifier on task %d.'%(col+1))
            lin = svm.SVC(kernel='precomputed', C=10, probability=True)
            lin.fit(K_train_notnan, Y_train_notnan)
        else:
            logging.info('\tTraining regressor on task %d.'%(col+1))
            lin = svm.SVR(kernel='precomputed', C=10)
            lin.fit(K_train_notnan, Y_train_notnan)
        lins.append(lin)

    model = {'kernel':ker, 'linear':lins, 'nans':nans}
    save_model(model, model_path)
    logging.info('\tTrained model saved to \"%s\".'%(model_path.split('/')[-1]))

    if X_test is not None and Y_test is not None:
        score = evaluate(args, X_test, Y_test)
        logging.info('\tAll tasks averaged score (%s): %.6f.'%(args['metric'],score))
        return score


def evaluate(args, X_test, Y_test):
    pred_test = predict(args, X_test)
    eval_metric = args['eval_fn']
    assert len(pred_test)==Y_test.shape[1]
    scores = []
    for pred, true in zip(pred_test, Y_test.T):
        pred = pred[~np.isnan(true)]
        true = true[~np.isnan(true)]
        score = eval_metric(true, pred)
        scores.append(score)
    return np.array(scores).mean()


def predict(args, X_test, save=True):
    
    seed = str(args['seed']) if not args['split_ready'] else ''
    model_path = "%s%s_%s.pkl"%(args['model_path'],args['dataset'],seed)
    model = load_model(model_path)
    assert model is not None
    ker = model['kernel']
    lins = model['linear']
    nans = model['nans']
    assert len(lins)==len(nans)
    logging.info('Predicting on dataset %s...'%args['dataset'])
    logging.info('\tModel loaded.')

    K_test = ker.transform(X_test)
    preds = []
    for nan_idx, lin in zip(nans, lins):
        K_test_notnan = K_test[:,~nan_idx]
        if args['metric'] in ['ROC', 'PRC']:
            pred_test = lin.predict_proba(K_test_notnan)[:,1]
        else:
            pred_test = lin.predict(K_test_notnan)
        preds.append(pred_test)

    if save:
        prediction_path = '%s%s_seed_%s.npy'%(args['prediction_path'],
                                    args['dataset'], seed)
        np.save(prediction_path, np.array(preds).T)
        logging.info('\tPredictions saved to \"%s\".'%(prediction_path.split('/')[-1]))

    return np.array(preds)

