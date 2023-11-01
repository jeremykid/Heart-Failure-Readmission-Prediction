#function to plot results    
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix, accuracy_score,roc_curve, auc
from collections import OrderedDict
import numpy as np

def evalplots(y_test,y_score,y_pred,labels, thrplot = False):

    precision, recall, thr = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    f1score = f1_score(y_test, y_pred)
    f1vec = [hmean([precision[i],recall[i]]) for i in range(sum(recall!=0))]
    
    #plt.plot([i/len(f1vec) for i in range(len(f1vec))],f1vec,color='r',alpha=0.2)
    plt.figure(figsize = (15,7))
    plt.subplot(1, 2, 1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}, F1={1:0.2f}'.format(average_precision,f1score))
    #plt.show()
    
    # Compute ROC curve
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.show()

    if thrplot:
        plt.step( thr[recall[:-1]!=0],f1vec,color='r',alpha=0.2,where='post')
        plt.fill_between(thr[recall[:-1]!=0],f1vec,step='post', alpha=0.2,color='r')
        plt.xlabel('Threshold')
        plt.ylabel('Estimated F1-Scores')
        plt.ylim([0.0, 1.0])
        plt.axvline(x=0.5,color ='r')
        plt.title('Threshold Vs F1-Score: Max F1 ={0:0.2f}, Reported F1={1:0.2f}'.format(np.max(f1vec),f1score))
        plt.show()        

        #plt.rcParams["figure.figsize"] = (5, 5)
        plt.step(precision[:-1], thr, color='b', alpha=0.2, where='post')
        plt.fill_between(precision[:-1], thr, alpha=0.2, color='b', step='post')
        plt.xlabel('precision')
        plt.ylabel('Threshold')
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.grid()
        plt.show()

        #plt.rcParams["figure.figsize"] = (5, 5)
        plt.step(recall[:-1], thr, color='b', alpha=0.2, where='post')
        plt.fill_between(recall[:-1], thr, alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Threshold')
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.grid()
        plt.show()
    
    cm = confusion_matrix(y_test, y_pred,labels)
    print('Recall: {0:0.2f}'.format(recall_score(y_test, y_pred)))
    print('Precision: {0:0.2f}'.format(precision_score(y_test, y_pred)))
    display(pd.DataFrame(cm,columns = ['Negative','Positive'], index = ['Negative','Positive']))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap='hot')
    print('\n')
    plt.title('Confusion matrix : Acc={0:0.2f}'.format(accuracy_score(y_test, y_pred)))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print('--------------------------------------------------------')
    
#plot percentage of missing values
def plotnaperc(f_perc): 
    plt.figure(figsize = (20,10))
    plt.plot([sum(f_perc<i/100) for i in range(0,101)])
    plt.xlabel('percent missing')
    plt.ylabel('number')
    plt.xticks(np.linspace(0, 100,51))
    plt.grid()
    plt.show
    
#classification report
def class_report(y_test, y_pred,y_score):
    acc = (y_pred == y_test).mean()
    roc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred, average='binary')
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) 
    aprec = average_precision_score(y_test, y_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn+fp)

    print("acc score: {0:,.4f}".format(acc))
    print("roc score: {0:,.4f}".format(roc))
    print("f1 score: {0:,.4f}".format(f1))
    print("Precision score: {0:,.4f}".format(prec))
    print("Specificity score: {0:,.4f}".format(spec))
    print("Recall score: {0:,.4f}".format(rec))
    print('Average precision-recall score: {0:,.4f}'.format(aprec))
    return(acc,roc,f1,prec,rec,spec,aprec,tn,fp,fn,tp)

from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve

# def k_fold_cross_validation(X, y, model='catboost', k=5):
    
#     cat_feats = list(X.select_dtypes(exclude=['float64','int64']).columns)
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
#     X = X.reset_index(drop=True)
#     y = y.reset_index(drop=True)
    
#     result = {}
#     index = 1
    
#     # Todo Veify the test index
    
#     for train_index, test_index in tqdm(kf.split(X, y)):
        
#         X_train = X.loc[train_index].reset_index(drop=True)
#         X_test = X.loc[test_index].reset_index(drop=True)
#         y_train = y.loc[train_index].reset_index(drop=True)
#         y_test = y.loc[test_index].reset_index(drop=True)

#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
#                                                       random_state=40, shuffle=True)
#         print ('----- cv5 -----', model)
#         print (train_index[:10])
#         print (test_index[:10])
#         print (len(train_index))
#         print (len(test_index))
#         print ('----- end of cv5 -----')
#         break
#         if model == 'catboost':
#             train_data = Pool(data=X_train,label=y_train.astype(int), weight=np.ones(len(y_train)), cat_features = cat_feats)
#             eval_data = Pool(data=X_val,label=y_val.astype(int), weight=np.ones(len(y_val)), cat_features = cat_feats)
#             spw =  (len(y_train) - sum(y_train))/sum(y_train)
#             mod = CatBoostClassifier(iterations=2000,
#                                    #depth=3,
#                                    #learning_rate=0.01,
#                                   loss_function='Logloss',
#                                  custom_metric = ['Logloss','F1'],
#                                  eval_metric = 'AUC',#'F1', #'F1:use_weights=true'
#                                  scale_pos_weight = spw, 
#                                  #one_hot_max_size = 200,
#                                   task_type = "GPU", 
#                                   devices='3',
#                                   metric_period = 100)

#             mod.fit(train_data, eval_set=eval_data, early_stopping_rounds = 100, use_best_model=True)
#             y_pred = mod.predict(X_test)
#             y_score = mod.predict_proba(X_test)[:,1]
#             creport = class_report(y_test,y_pred,y_score)
#         elif model == 'random_forest'
#             clf = RandomForestRegressor(criterion='squared_error', random_state=56, max_features='sqrt', n_estimators=1000).fit(X_train, y_train)
#             pred = clf.predict(X_test)
#             pred_prob = clf.predict_proba(X_test)[:,1]
#             creport = class_report(y_test,pred,pred_prob)
#         else: #logistic regression
            
#             clf = LogisticRegression(penalty='l2',C =1,random_state=56, class_weight='balanced', max_iter=1000,solver='liblinear').fit(X_train, y_train)
#             pred = clf.predict(X_test)
#             pred_prob = clf.predict_proba(X_test)[:,1]
#             creport = class_report(y_test,pred,pred_prob)


    
#         result[str(index)+'_fold'] = creport
#         index+=1
#     return result
    

def plot_stacked_bar(data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, x_label=None, 
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    x_label         -- Label for x-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))
    axes = []
    cum_size = np.zeros(ny)
    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)
            
    if x_label:
        plt.xlabel(x_label)

    plt.legend(loc='upper left')

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")

#get_feature_importance
def get_feat_imp(X,mod,num=25):
    imp_features = dict(zip(X.columns, mod.get_feature_importance()))
    imp_features_sort = OrderedDict(sorted(imp_features.items(), key=lambda kv: kv[1], reverse=True))
    print(np.array(list(imp_features_sort.keys()))[:num])
    return imp_features_sort

#function to return correlated features
def getcorrfeat_frommat(corr_matrix, threshold, verbose = False):
    col_corr = set() # Set of all the names of to be discarded columns to be returned
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                if verbose:
                    print(str(i) + ' ' + str(j) + ' are i and j ')
                    print(corr_matrix.columns[i] + ' and ' + corr_matrix.columns[j] + ' are correlated above ' + str(threshold))
                    print('Discarding ' + corr_matrix.columns[i])
                    print()
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                #if colname in dataset.columns:
                #    del dataset[colname] # deleting the column from the dataset
    return(list(col_corr))