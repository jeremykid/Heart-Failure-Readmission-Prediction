from catboost import CatBoostClassifier, Pool
from utils import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
    
from torch.utils.data import DataLoader, TensorDataset
from generate_report import get_optimal_cutoff, get_pred_report, evalplots, class_report

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

def run_Logistic_Regression_model(X_log, y, X_val_log, y_val, X_train_log, y_train, X_test_log, y_test, verbose=False):
    clf = dict()
    pred = dict()
    pred_prob = dict()
    score = dict()
    log_creport = dict()
    shap_values = dict()
    return_result = {}
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

def run_Random_Forest_model(X_log, y, X_val_log, y_val, X_train_log, y_train, X_test_log, y_test, verbose=False):
    clf = dict()
    pred = dict()
    pred_prob = dict()
    score = dict()
    log_creport = dict()
    shap_values = dict()
    return_result = {}

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

def run_SVM_model(X_log, y, X_val_log, y_val, X_train_log, y_train, X_test_log, y_test, verbose=False):
    clf = dict()
    pred = dict()
    pred_prob = dict()
    score = dict()
    log_creport = dict()
    shap_values = dict()
    return_result = {}

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

# Define the neural network model
class DeepModel(nn.Module):
    def __init__(self, input_size):
        super(DeepModel, self).__init__()
        self.input_size = input_size
        # Input layer to first hidden layer (200 -> 330)
        self.fc1 = nn.Linear(self.input_size, 330)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(330, 433)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(433, 511)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(511, 449)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(449, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
    


# out = 'readm_death'
def generate_torch_loader(X,Y,batch_size=32):
    X_tensor = torch.tensor(X.values.astype(np.float32))
    Y_tensor = torch.tensor(Y.values.astype(np.float32))
    dataset = TensorDataset(X_tensor, Y_tensor)
    torch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return torch_loader


def get_pre_label(model, data_loader):
    model.eval()
    lossfun = nn.BCELoss()
    pred_list = []
    label_list = []
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for X_batch, y_batch in data_loader:
                y_pred = model(X_batch)
                loss = lossfun(y_pred, y_batch)
                pred_list.append(y_pred)
                label_list.append(y_batch)
                pbar.update(1)
    pred_list = [pred.cpu().detach().numpy() for pred in pred_list]
    label_list = [label.cpu().detach().numpy() for label in label_list]

    labels = np.concatenate(label_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    labels = labels.reshape(labels.shape[0], 1)
    pred = pred.reshape(pred.shape[0], 1)
    return labels, pred

def train_torch_model(train_loader, val_loader, model_path, input_size, pos_weight = None):
    model = DeepModel(input_size=input_size)
    # Loss function and optimizer
    
    if pos_weight == None:
        criterion = nn.BCELoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and Validation
    n_epochs = 50  # Number of epochs; you can adjust this value

    min_val_loss = 999999999999999999
    early_stop_count, early_stop_epoch = 0, 9
    for epoch in range(n_epochs):
        if early_stop_count >= early_stop_epoch:
            break
        model.train()  # Set model to training mode
        train_loss = 0.0
        # Training loop
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # balance class weight in pytorch
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}")

        model.eval()  # Set model to evaluation mode

        # Validation loop
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.view(-1, 1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print ('save_model:', val_loss)
            torch.save(model, model_path)
            early_stop_count = 0
        else:
            early_stop_count += 1

        print(f"Epoch {epoch+1}/{n_epochs} - Validation Loss: {val_loss}")
    return model
 

def run_NN_model(X, y, X_val, y_val, X_train, y_train, X_test, y_test):
    train_loader = generate_torch_loader(X_train, y_train)
    val_loader = generate_torch_loader(X_val, y_val)
    test_loader = generate_torch_loader(X_test, y_test)

    model = train_torch_model(train_loader, val_loader, model_path, input_size)
    
    train_labels, train_pred = get_pre_label(model, train_loader)
    test_labels, test_pred = get_pre_label(model, test_loader)

    # Use the train_y to generate the optimal cutoff point
    _,_, _, _, roc_j_thr_dict = get_optimal_cutoff(train_labels,train_pred, ['out'])
    # Use the test_y to generate report
    class_df_dict, creport_dict = get_pred_report(test_labels,test_pred,['out'],roc_j_thr_dict, verbose=False) # verbose = False
    
    return creport_dict
    