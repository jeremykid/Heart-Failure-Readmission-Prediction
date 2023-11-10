from sklearn.metrics import roc_auc_score,classification_report, det_curve,precision_recall_fscore_support, brier_score_loss
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, roc_curve, auc, RocCurveDisplay 
from sklearn.metrics import DetCurveDisplay, f1_score, ConfusionMatrixDisplay, average_precision_score, precision_recall_curve
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve

# import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns
from scipy.stats import hmean
import numpy as np
import pandas as pd

from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))
    
#function to plot results    
def evalplots(y_test,y_score,y_pred,labels, creport_dict=None, thrplot = False):
    '''
    
    '''
    precision, recall, thr = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    f1score = f1_score(y_test, y_pred)
    f1vec = [hmean([precision[i],recall[i]]) for i in range(sum(recall!=0))]
    
    #plt.plot([i/len(f1vec) for i in range(len(f1vec))],f1vec,color='r',alpha=0.2)
    plt.figure(figsize = (15,7))
    plt.subplot(1, 2, 1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    if creport_dict:
        plt.plot([creport_dict['rec'], 0], [creport_dict['prec'], creport_dict['prec']], color = 'blue', linestyle='--')
        plt.plot([creport_dict['rec'], creport_dict['rec']], [creport_dict['prec'], 0], color = 'blue', linestyle='--')
        
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
    # print ('creport_dict', creport_dict)
    if creport_dict:
        tfrate = creport_dict['tp']/(creport_dict['tp']+creport_dict['fn'])
        fprate = creport_dict['fp']/(creport_dict['fp']+creport_dict['tn'])
        plt.plot([fprate, 0], [tfrate, tfrate], color = 'blue', linestyle='--')
        plt.plot([fprate, fprate], [0, tfrate], color = 'blue', linestyle='--')
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

    '''
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
    '''
       
#classification report
def class_report(y_test, y_pred, y_score, verbose=True):
    acc = (y_pred == y_test).mean()
    roc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred, average='binary')
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) 
    
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = metrics.auc(lr_recall, lr_precision)
    
    aprec = average_precision_score(y_test, y_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    br_score = brier_score_loss(y_test, y_score, pos_label=y_test.max())
    spec = tn / (tn+fp)
    sensitivity = tp / (tp + fn)

    if (verbose):
        print("AUROC score: {0:,.4f}".format(roc))
        print("AUPRC score: {0:,.4f}".format(auprc))     #TODO AUPRC
        print('Average precision-recall score: {0:,.4f}'.format(aprec))
        print("Accuracy score: {0:,.4f}".format(acc))
        print("Sensitivity / Recall score: {0:,.4f}".format(rec))
        print("Specificity score: {0:,.4f}".format(spec))
        print("Positive predictive value / Precision score: {0:,.4f}".format(prec))
        print("f1 score: {0:,.4f}".format(f1))
        print("Brier score: {0:,.4f}".format(br_score))
        print()

    return {
        'accuracy': acc,
        'auroc': roc,
        'auprc': auprc,
        'f1_score': f1,
        'prec': prec,
        'rec': rec,
        'sensitivity': sensitivity,
        'spec': spec,
        'aprec': aprec,
        'br_score': br_score,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    } #acc, roc, auprc, f1,prec,rec,spec,aprec,br_score,tn,fp,fn,tp

#plot the calibration curve
def plot_calib_curve(y_test,y_score, nbins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_score, n_bins = nbins)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "o-", label="DL Model", linewidth=3)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Model", linewidth=1.5)
    plt.title('Calibration Plot')
    plt.xlabel('Mean Predicted Score')
    plt.ylabel('Fraction of True Positives')
    plt.legend()
    plt.show()
    
    
def get_optimal_cutoff(y_test,y_prob,text_labels):
    # get optimal cutoff based on tpr - fpr should be max : Youden's J-Score
    n_classes = len(text_labels)
    y_test = y_test
    y_score = y_prob

    # Compute ROC curve and ROC area for each class
    roc = dict()
    roc_auc = dict()
    roc['tpr'] = dict()
    roc['fpr'] = dict()
    roc['thr'] = dict()
    roc_df = dict()
    roc_tf_thr = dict()
    roc_j_thr = dict()
    roc_j_thr_dict = dict()
        
    for i in range(n_classes):
        roc['fpr'][i], roc['tpr'][i], roc['thr'][i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[text_labels[i]] = auc(roc['fpr'][i], roc['tpr'][i])
        j = np.arange(len(roc['tpr'][i])) 
        roc_df[text_labels[i]] = pd.DataFrame({'tf' : pd.Series(roc['tpr'][i]-(1- roc['fpr'][i]), index=j),'j_score' : pd.Series((roc['tpr'][i]- roc['fpr'][i]), index=j), 'threshold' : pd.Series(roc['thr'][i], index=j)})
        roc_tf_thr[text_labels[i]] = roc_df[text_labels[i]].iloc[(roc_df[text_labels[i]].tf-0).abs().argsort()[:1]]
        roc_j_thr[text_labels[i]] = roc_df[text_labels[i]].iloc[(roc_df[text_labels[i]].j_score).argsort()[-1:]]
        roc_j_thr_dict[text_labels[i]] =  roc_j_thr[text_labels[i]]['threshold'].values[0]
    return roc_auc,roc_df, roc_tf_thr, roc_j_thr, roc_j_thr_dict


def get_pred_report(y_test,y_prob,text_labels,roc_j_thr_dict, verbose = True):
    n_classes = len(text_labels)
    y_test = y_test
    y_score = y_prob
    class_df_dict = dict()
    creport_dict = dict()

    for i in range(n_classes):
        j = np.arange(len(y_score[:, i])) 
        class_df_dict[text_labels[i]] = pd.DataFrame({'true' : pd.Series(y_test[:, i], index=j),'pred_proba' : pd.Series(y_score[:, i], index=j)})
        class_df_dict[text_labels[i]]['pred'] = class_df_dict[text_labels[i]]['pred_proba'].map(lambda x: 1 if x > roc_j_thr_dict[text_labels[i]] else 0)
        #print(confusion_matrix(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred']))
        if verbose:
            printmd('**' + text_labels[i] + '**')
            print('Cutoff Probability based on Training ROC: ', roc_j_thr_dict[text_labels[i]])
            print (classification_report(y_test[:, i], class_df_dict[text_labels[i]]['pred'], target_names=['0','1']))

        creport_dict[text_labels[i]] = class_report(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred'], class_df_dict[text_labels[i]]['pred_proba'], verbose)
        if verbose:
            cm = confusion_matrix(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6,6))
            disp.plot(ax=ax) 
            evalplots(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred_proba'], 
                      class_df_dict[text_labels[i]]['pred'], [0, 1], creport_dict[text_labels[i]])
            plot_calib_curve(class_df_dict[text_labels[i]]['true'],class_df_dict[text_labels[i]]['pred_proba'], nbins=10)

            print('----------------------------------------')
    
    return class_df_dict, creport_dict


# import sklearn.metrics as metrics

# import matplotlib.pyplot as plt
# from sklearn.metrics import DetCurveDisplay, f1_score, ConfusionMatrixDisplay, average_precision_score, precision_recall_curve


# columns = train_y_true.columns.tolist()
# fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(24, 10))

# for i in range(len(columns)):
#     y = test_y_true[columns[i]]
#     pred = test_y_prob[columns[i]]
#     fpr, tpr, thresholds = metrics.roc_curve(y, pred)
#     roc_auc = metrics.auc(fpr, tpr)

#     ax1.plot(fpr, tpr, 
#              label='AUROC curve of class '+ columns[i] +' (area =  '+str(roc_auc.round(4))+')')

#     ax1.legend(loc="lower right")

#     ax1.plot([0, 1], [0, 1], 'k--')
#     ax1.set_xlim([0.0, 1.0])
#     ax1.set_ylim([0.0, 1.05])
#     ax1.set_xlabel('False Positive Rate')
#     ax1.set_ylabel('True Positive Rate')
    
#     lr_precision, lr_recall, _ = precision_recall_curve(y, pred)
    
#     ax2.plot(lr_recall, lr_precision, marker='.', 
#              label='AUPRC curve of class '+ columns[i] +' (area =  '+str(roc_auc.round(4))+')')
#     # axis labels
#     ax2.set_xlabel('Recall')
#     ax2.set_ylabel('Precision')
#     # show the legend
#     ax2.legend()

    
# plt.show()
