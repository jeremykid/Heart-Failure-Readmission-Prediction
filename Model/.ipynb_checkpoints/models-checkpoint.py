from catboost import CatBoostClassifier, Pool
from helps import *
import numpy as np

def run_CatBoost_model(X, y, X_val, y_val, X_train, y_train, X_test, y_test, verbose=True):
    '''
    CatBoost Model
    Return Result Dict 
    y_test
    y_pred
    y_score
    creport
    impfeat_x
    impfeat_y
    '''
    cat_feats = list(X.select_dtypes(exclude=['float64','int64']).columns)

    #scale pos wt
    spw = dict()
    for out in y: spw[out] =  (len(y_train[out]) - sum(y_train[out]))/sum(y_train[out])
    if verbose:
        print('scale pos wt :', spw)
        print()
    
    return_result = {}
    #generate dataset for catboost
    train_data = dict()
    eval_data = dict()
    for out in y:
        train_data[out] = Pool(data=X_train[out],label=y_train[out].astype(int), weight=np.ones(len(y_train[out])), cat_features = cat_feats)
        eval_data[out] = Pool(data=X_val[out],label=y_val[out].astype(int), weight=np.ones(len(y_val[out])), cat_features = cat_feats)

    #fit the model
    mod = dict ()
    for out in y:
        mod[out] = CatBoostClassifier(iterations=2000,
                                   #depth=3,
                                   #learning_rate=0.01,
                                  loss_function='Logloss',
                                 custom_metric = ['Logloss','F1'],
                                 eval_metric = 'AUC',#'F1', #'F1:use_weights=true'
                                 scale_pos_weight = spw[out], 
                                 #one_hot_max_size = 200,
                                 verbose=verbose, 
                                task_type = "GPU", 
                                 devices='3',
                                  metric_period = 100)

        if verbose: print('Model Performance for: ' + out)
        #mod[out].fit(train_data[out], cat_features = cat_feats, eval_set=eval_data[out], early_stopping_rounds = 50, plot = True, use_best_model=True)
        mod[out].fit(train_data[out], eval_set=eval_data[out], early_stopping_rounds = 100, plot = verbose, use_best_model=True)
        print()

    #evaluate the model and get important features
    y_pred = dict()
    y_score = dict()
    creport = dict()
    impfeat = dict()
    impfeat_x = dict()
    impfeat_y = dict()

#     print('Model Performance for: observations, ' +  str(ndays) + ' ' +label_type + ' days outcome with ' + feat_type + ' features, evaluated by ' + evalsplit + ' split.')
    for out in y:
        print('Model Performance for: ' +  out + ' outcome.')
        y_pred[out] = mod[out].predict(X_test[out])
        y_score[out] = mod[out].predict_proba(X_test[out])[:,1]
        creport[out] = class_report(y_test[out],y_pred[out],y_score[out])
#         evalplots(y_test[out],y_score[out],y_pred[out],[0, 1])
        impfeat[out] = get_feat_imp(X_train[out],mod[out])
        impfeat_x[out] = list(impfeat[out].values())[:100]
        impfeat_y[out] =  list(impfeat[out].keys())[:100]

        print('______________________________________________')
        print()
    return_result['catboost'] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "y_score": y_score,
            "creport": creport,
            "impfeat_x": impfeat_x,
            "impfeat_y": impfeat_y,
            "impfeat": impfeat
        }
    return return_result


from sklearn.linear_model import LogisticRegression
import shap

def run_Logistic_Regression_model(X, y, X_val, y_val, X_train, y_train, X_test, y_test, verbose=False):
    clf = dict()
    pred = dict()
    pred_prob = dict()
    score = dict()
    log_creport = dict()
    shap_values = dict()

    for out in y:
        print(out)
        print('Logistic Regression Model Performance for: ' +  out + ' outcome.')
        clf[out] = LogisticRegression(penalty='l2',C =1,random_state=56, class_weight='balanced', max_iter=1000,solver='liblinear').fit(X_train_log[out], y_train[out])
        pred[out] = clf[out].predict(X_test_log[out])
        pred_prob[out] = clf[out].predict_proba(X_test_log[out])[:,1]
        log_creport[out] = class_report(y_test[out],pred[out],pred_prob[out])
        explainer = shap.LinearExplainer(clf[out], X_train_log[out], feature_perturbation="interventional")
        shap_values[out]  = explainer.shap_values(X_test_log[out])
        shap.summary_plot(shap_values[out], X_test_log[out])
        print('______________________________________________')
        print()

    return_result['logisReg'] = {
        "y_test": y_test,
        "pred": pred,
        "pred_prob": pred_prob,
        "shap_values": shap_values,
        "log_creport": log_creport,
    }
    return return_result

from sklearn.ensemble import RandomForestClassifier

def run_Random_Forest_model(X, y, X_val, y_val, X_train, y_train, X_test, y_test, verbose=False):
    clf = dict()
    pred = dict()
    pred_prob = dict()
    score = dict()
    log_creport = dict()
    shap_values = dict()

    for out in y:
        print(out)
        print('Random Forest Model Performance for: ' +  out + ' outcome.')
        clf[out] = RandomForestClassifier(random_state=0, class_weight="balanced").fit(X_train_log[out], y_train[out])
        pred[out] = clf[out].predict(X_test_log[out])
        pred_prob[out] = clf[out].predict_proba(X_test_log[out])[:,1]
        log_creport[out] = class_report(y_test[out],pred[out],pred_prob[out])


    return_result['logisRF'] = {
        "y_test": y_test,
        "pred": pred,
        "pred_prob": pred_prob,
        "shap_values": shap_values,
        "RF_creport": log_creport,
    }
    return return_result

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.pipeline import make_pipeline

def run_SVM_model(X, y, X_val, y_val, X_train, y_train, X_test, y_test, verbose=False):
    clf = dict()
    pred = dict()
    pred_prob = dict()
    score = dict()
    log_creport = dict()
    shap_values = dict()

    for out in y:
        print(out)
        print('SVM Model Performance for: ' +  out + ' outcome.')
        clf[out] = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True)).fit(X_train_log[out], y_train[out])
#         clf[out] = make_pipeline(StandardScaler(), SVC(kernel='auto', probability=True)).fit(X_train_log[out], y_train[out])
#         clf[out] = svm.SVC(kernel='linear', probability=True).fit(X_train_log[out], y_train[out])

        pred[out] = clf[out].predict(X_test_log[out])
        pred_prob[out] = clf[out].predict_proba(X_test_log[out])[:,1]
        log_creport[out] = class_report(y_test[out],pred[out],pred_prob[out])

    return_result['SVM'] = {
        "y_test": y_test,
        "pred": pred,
        "pred_prob": pred_prob,
        "shap_values": shap_values,
        "SVM_creport": log_creport,
    }
    return return_result