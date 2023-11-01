import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import fiscalyear
fiscalyear.START_MONTH = 4 #fiscal year start in April
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ignore some warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def preprocess_HFclassifier(lb_obs_id, label_type, feat_type, ndays, evalsplit, 
                            verbose = True): #random_forest
    #function to select observations, feature groups; create feature matrix; learn catboost model and evaluate for HF readmission/death.
    #lb_obs_id -> observation type. options: 1ylb_fix_MR, 1ylb_fix_inp_MR
    #label_type -> label type. options: 'MR','f5'; 'MR' produces 'only death' and 'readm + death' models, when 'f5' is selected 'only death' model is not generated 
    #feat_type -> group name of feature. options: 'no_select','all_feats', 'dx_feats', 'demo_feats', 'visit_feats', 'intrv_feats', 'dish_feats', 'cihi_feats' ('no_select' = no feature selection)
    #ndays -> number of days till event to model options: any integer, usually 30 or 365
    #evalsplit -> type of data splitting for train, validation and test folds. options:'fiscal','random' 
    #logr -> boolean to indicate whether to perform logistic regression based on top features from catboost results
    
    random.seed(14)
    mindays = 365 #minimum number of days in observation window : any integer, default 365 days for 1yrlb_fix
    proc_datadir = '/home/padmalab/heart_failure/data/processed/'
    label_fname = proc_datadir + 'label_4_tables_2002_death_inp_pop_added.pickle'
    sel_feats_dict_fname = proc_datadir + 'sel_grp_feats_dict_0216_' + lb_obs_id + '_200920.pickle'
    ftdata_allsub_fname = proc_datadir + 'feats_allsub_0216_' + lb_obs_id + '_040920.pickle'
    lb_rcpt_obs_id = 'rcpt_' + lb_obs_id +'_obs_id'
    obstype = lb_rcpt_obs_id[14:-7]
    return_result = {
        'report': {},
    }
    
    #load labels for particular label type
    with open(label_fname, 'rb') as handle: label = pickle.load(handle)[obstype]  
    if verbose: print('Shape of Label dataframe: ', label.shape)

    #load features for all obs
    with open(ftdata_allsub_fname, 'rb') as handle: X_allsub = pickle.load(handle)
    if verbose: print('Shape of feature dataframe for all observations: ', X_allsub.shape)


    #load feature name grouping
    with open(sel_feats_dict_fname, 'rb') as handle: sel_feats_dict = pickle.load(handle)
    if verbose: print('Feature groups: ', sel_feats_dict.keys())

    #define label vars
    days_till_readm = 'days_till_'+ obstype +'_epiend_' + obstype + '_'+ 'nextReadm'
    days_till_f5_readm = 'days_till_'+ obstype +'_epiend_' + obstype.replace('MR','f5') + '_'+ 'nextReadm'
    days_till_death = 'days_till_'+ obstype +'_epiend_' + 'death'
    rt_censor = obstype + '_rt_censor'
    lt_censor = obstype + '_lt_censor'
    obs_id = obstype + '_obs_id'
    death_bool = 'death_bool'

    #create rcpt_obs_id and set as index
    rcpt_obs_id = 'rcpt_'+ obs_id
    label[rcpt_obs_id] = label.Rcpt_Anon_ID.str.decode('utf-8') + '_' + label[obs_id].astype(int).astype(str)
    label = label.set_index(rcpt_obs_id)

    # if interested in f5_readm
    if label_type == 'f5' : days_till_readm = days_till_f5_readm 

    if verbose:
        print('Censored death percentage :', (~label[death_bool]).mean()) 
        print('Total censored percentage :',((~label[death_bool]) & (label[rt_censor]).astype(bool)).mean()) 
        print()

    ###Exclusions of instances
    #Exclude if not resident of Alberta - already done
    #Exclude the instances without pop record in the year matching the episode start date - already done
    #epi_end_date is same as obs_end_date
    #If rt_censor is False: days_till_readm  = next_episode_start_date - epi_end_date
    #If rt_censor is True and death_bool is False: days_till_readm = study_end_date (2016-03-31) - epi_end_date
    #If rt_censor is True and death_bool is True: days_till_readm = death_date - epi_end_date (when <= ndays, exclude for readm only predict, but retain for death or death + readm prediction)
    #If death_bool is False: days_till_death = study_end_date (2016-03-31) - epi_end_date
    #If death_bool is True: days_till_death = death_date - epi_end_date

    #Exclude if death occurs within same or next day
    bl_dth = label[days_till_death] <= 1
    if verbose: print('Number, percent of death in hosp (or next day): '+ str(bl_dth.sum()),str(bl_dth.mean()))

    #exclude observations with right censored with less than n days in the prediction window
    #When there is no nextReadm event and the patient survives till study end, the last study date is used for calculation -- *not used*
    bl_rt_cen = (label[rt_censor]).astype(bool) & (label[days_till_readm] <= ndays) 
    if verbose: print('Number, percent of right censored readm at less than ' + str(ndays) + ' days (considering death as end of study; for readm only prediction): '+ str(bl_rt_cen.sum()),str(bl_rt_cen.mean()))

    #exclude observations where patinet is not dead and not readmitted by end of the study with less than n days in the prediction window
    bl_rt_dth_cen = ((~label[death_bool]) & (label[rt_censor]).astype(bool)) & (label[days_till_readm] <= ndays) 
    if verbose: print('Number, percent of right censored readm or death with less than ' + str(ndays) + ' days prediction window: '+ str(bl_rt_dth_cen.sum()),str(bl_rt_dth_cen.mean()))
            
    #exclude all episodes that ended after ndays of obs end date ie with partial prediction window (eg: 2015-03-31 for 1 yr lb)
    obs_study_end = label['epi_end_date'].max() - np.timedelta64(ndays, 'D')
    if verbose: print("End date for episodes to be considered :", obs_study_end)
    bl_partial_pred = label['epi_end_date'] >= obs_study_end
    if verbose: print('Number, percent of observations with less than ' + str(ndays) + ' days prediction window: '+ str(bl_partial_pred.sum()),str(bl_partial_pred.mean()))

    #exclude all episodes that started before mindays of lookback period (eg: 2003-03-31 for 1 yr lb)
    obs_study_start = label['epi_start_date'].min() + np.timedelta64(mindays, 'D')
    if verbose: print("Start date for episodes to be considered :", obs_study_start)
    bl_partial_obs = label['epi_start_date'] <= obs_study_start
    if verbose: print('Number, percent of observations with less than ' + str(mindays) + ' days observation window: '+ str(bl_partial_obs.sum()),str(bl_partial_obs.mean()))

    #exclude the instances without features because pop record in the year matching the episode start date was missing
    bl_sub = np.array([i not in X_allsub.index for i in label.index])
    if verbose: print('Number, percent of observations without pop record for episode start year: '+ str(bl_sub.sum()),str(bl_sub.mean()))

    #merge
    #exl_bl = (bl_dth | bl_rt_cen | bl_partial_obs|bl_sub) # use if prediction is only for readm, not considering death
    #exl_bl = (bl_dth | bl_rt_dth_cen | bl_partial_obs|bl_sub) #uncensored events with partial prediction window are included
    exl_bl = (bl_dth | bl_partial_pred | bl_partial_obs|bl_sub) #all events with partial prediction window are excluded
    if verbose: print('Total number, percent of observations to exclude: ' + str(exl_bl.sum()),str(exl_bl.mean()))
    print()

    #list of instances selected
    sel_obs_id = label[~exl_bl].index
    return_result['report']['number_patients'] = len(set([s.split('_')[0] for s in sel_obs_id]))
    return_result['report']['number_instances'] = len(sel_obs_id)
    return_result['report']['number_features'] = X_allsub.shape[1]
    if verbose:
        print("Number of patients : " + str(len(set([s.split('_')[0] for s in sel_obs_id]))))
        print("Number of selected instances : " + str(len(sel_obs_id)))
        print('Censored death percentage with selected instances :', (~label.loc[sel_obs_id][death_bool]).mean()) 
        print('Total censored percentage with selected instances ',((~label.loc[sel_obs_id][death_bool]) & (label.loc[sel_obs_id][rt_censor]).astype(bool)).mean()) 
        print()
        
    #Select obs id for the X
    X = X_allsub.loc[sel_obs_id]

    #define ys
    y = dict()
    #y['readm'] = label.loc[sel_obs_id,days_till_readm] <= ndays #readm only ; consider bl_rt_cen instead of bl_rt_dth_cen
    y['readm_death'] = ((label.loc[sel_obs_id,days_till_readm] <= ndays) | (label.loc[sel_obs_id,days_till_death] <= ndays)) #readm or death
    if label_type == 'MR' : y['death'] = label.loc[sel_obs_id,days_till_death] <= ndays #death only
    for out in y:  
        return_result['report']['outcome_percentage '+out] = y[out].mean()
        if verbose: 
            print(out, len(y[out]),y[out].sum(),y[out].mean(),1-y[out].mean())
    print()
    
    #check if obs_year match the epi start date
#     assert all(X['obs_year'] == label.loc[X['obs_year'].index,'epi_start_date'].dt.year)
    if ~(all(X['obs_year'] == label.loc[X['obs_year'].index,'epi_start_date'].dt.year)):
        print(sum(X['obs_year'] - label.loc[X['obs_year'].index,'epi_start_date'].dt.year),' mismatches found between obs_year and epi_start_date!')

    #change 'obs_year' to fiscal year
    epi_start_day = label.loc[X['obs_year'].index,'epi_start_date'].dt.day
    epi_start_mon = label.loc[X['obs_year'].index,'epi_start_date'].dt.month
    epi_start_year = label.loc[X['obs_year'].index,'epi_start_date'].dt.year
    X['obs_year'] = [fiscalyear.FiscalDate(epi_start_year[oid], epi_start_mon[oid], epi_start_day[oid]).fiscal_year for oid in X.index]
    
    #group hospitalizations by fiscal year and plot events
    #if 365 days predict, 2016 is not used
    obs_gp = X.groupby('obs_year')
    fisc_yrs = list(obs_gp.groups.keys())
    if verbose: print('fiscal years :',obs_gp.groups.keys())

    readm_fis_yr ={}
    death_fis_yr ={}
    for yr in fisc_yrs:
        death_fis_yr[yr] = label.loc[obs_gp.groups[yr],days_till_death] <= ndays #death only
        readm_fis_yr[yr] = label.loc[obs_gp.groups[yr],days_till_readm] <= ndays #readm only
           
    event_count = {}
    event_count['Death'] = {}
    event_count['Alive'] = {}
    event_count['Readmisson'] = {}
    event_count['No Readmisson'] = {}

    for yr in fisc_yrs:
        event_count['Death'][yr] = death_fis_yr[yr].sum()
        event_count['Alive'][yr] = len(death_fis_yr[yr]) - death_fis_yr[yr].sum()
        event_count['Readmisson'][yr] = readm_fis_yr[yr].sum()
        event_count['No Readmisson'][yr] = len(readm_fis_yr[yr]) - readm_fis_yr[yr].sum()
        if verbose:
            print('Fiscal year', yr)
            print('Death', len(death_fis_yr[yr]),death_fis_yr[yr].sum(),death_fis_yr[yr].mean())
            print('Readm', len(readm_fis_yr[yr]),readm_fis_yr[yr].sum(),readm_fis_yr[yr].mean())
            print()
    
    return_result['report']['event_count'] = event_count
    ## plot event counts per fiscal year
    ## NOTE: Death in hospital or day following the discharge are not considered
    ## NOTE: Last month of 2016 fiscal year is not considered for 30 days, whole of 2016 is not considered for 365 days

    series_list = [['No Readmisson', 'Readmisson'],['Alive', 'Death']]
    if verbose:
        for series_labels in series_list:
            plt.figure(figsize=(15, 6))
            category_labels = list(event_count[series_labels[0]].keys())
            dat = [[event_count[sr][cat] for cat in category_labels] for sr in series_labels]

            plot_stacked_bar(
                dat, 
                series_labels, 
                category_labels=category_labels, 
                show_values=True, 
                grid = False,
                value_format="{:.0f}",
                colors=['tab:blue', 'tab:red'],
                y_label="Number of Hospitalizations",
                x_label="Fiscal Year"
            )

            plt.show()
    
    
    #Keep the right set of features as selected
    if feat_type != 'no_select': X = X.loc[:,sel_feats_dict[feat_type]]
    if verbose: 
        print('Feature group :', feat_type)
        print("Shape of feature matrix with selected obs:", X.shape)
    
    #convert nan in categorical features to empty strings
    obj_feats = list(X.select_dtypes(include=['O']).columns) #categorical object features
    #obj_feats = list(X.select_dtypes(exclude=['float64','int64']).columns)
    X[obj_feats] = X[obj_feats].fillna('')
    cat_feats = list(X.select_dtypes(exclude=['float64','int64']).columns)
        
    if evalsplit == 'fiscal':
        #split the data based on the fiscal year
        if verbose: print('Performing data splits by fiscal year :')
        yrs = {}
        yrs['train'] = [2005, 2006, 2007, 2008, 2009, 2011,2012]
        yrs['validation'] = [2004, 2010]
        if ndays == 30: yrs['test'] = [2013,2014,2015,2016]
        if ndays == 365: yrs['test'] = [2013,2014,2015]

        if verbose:
            for sp in yrs:
                print ('Fiscal years used for :', sp, yrs[sp])
            print()

        sp_obs_id = {}
        X_sp = {}
        y_sp = {}

        for sp in yrs:
            sp_obs_id[sp] = [obs_gp.groups[yr] for yr in yrs[sp]]
            sp_obs_id[sp] = [j for i in  sp_obs_id[sp] for j in i]
            X_sp[sp] = {}
            y_sp[sp] = {}
            for out in y:
                X_sp[sp][out] = X.loc[sp_obs_id[sp]].copy()
                y_sp[sp][out] = y[out].loc[sp_obs_id[sp]].copy()

        X_train = X_sp['train']
        y_train = y_sp['train']
        X_test = X_sp['test']
        y_test = y_sp['test']
        X_val = X_sp['validation']
        y_val = y_sp['validation']

    if evalsplit == 'random':
        #get random test trian val splits #stratify by class 
        if verbose: print('Performing random stratified splits :')
        X_train = dict()
        y_train = dict()
        X_test = dict()
        y_test = dict()
        X_val = dict()
        y_val = dict()
        for out in y:
            X_train[out], X_test[out], y_train[out], y_test[out] = train_test_split(X, y[out], test_size=0.20, random_state=42,shuffle = True, stratify = y[out])
            print (out, '-train:', X_train[out].shape)
            print (out, '-test:', X_test[out].shape)

            X_train[out], X_val[out], y_train[out], y_val[out]  = train_test_split(X_train[out], y_train[out], test_size=0.20,random_state=24, shuffle = True, stratify = y_train[out])
            
    if verbose:
        for out in y:
            print(out)
            return_result['report']['Train Set'+out] = len(y_train[out])
            return_result['report']['Val Set'+out] = len(y_val[out])
            return_result['report']['Test Set'+out] = len(y_test[out])
            print('Train Set N, PosN, PosPerc: ', len(y_train[out]), y_train[out].sum(),y_train[out].mean())
            print('Val Set N, PosN, PosPerc: ',len(y_val[out]), y_val[out].sum(),y_val[out].mean())
            print('Test Set N, PosN, PosPerc: ',len(y_test[out]), y_test[out].sum(),y_test[out].mean())
            print()
    for out in y:
        print(out)
        return_result['report']['Train Set'+out] = len(y_train[out])
        return_result['report']['Val Set'+out] = len(y_val[out])
        return_result['report']['Test Set'+out] = len(y_test[out])     

    #convert nan in categorical features to empty strings
    obj_feats = list(X.select_dtypes(include=['O']).columns) #categorical object features
    #obj_feats = list(X.select_dtypes(exclude=['float64','int64']).columns)
    X[obj_feats] = X[obj_feats].fillna('')
    cat_feats = list(X.select_dtypes(exclude=['float64','int64']).columns)
    
    return_result['number_of_features'] = X_allsub.shape[1]
    
    return return_result, X, y, X_val, y_val, X_train, y_train, X_test, y_test


def get_top_features(X, y, X_val, y_val, X_train, y_train, X_test, y_test, impfeat, verbose = False, n_log_feat = 25):
    #n_log_feat number of top catboost features to consider
    X_train_log = dict()
    X_val_log = dict()
    X_test_log = dict()
    one_thr = 0.05 #threshold for min frequency for one hot encoded featres
    corr_thresh = 0.5 #threshold for max correlation between features
    miss_bl = False #boolean to indicate whether to include missingness indicators
    train_val_comb_bl = False #boolean to indicate whether to include validation set in training
    featmap = {}

    for out in y:
        if verbose:
            print()
            print('Collecting features for logistic regression, for the outcome :', out)
        log_feats = np.array(list(impfeat[out].keys()))[:n_log_feat]
        cat_log_feats = list(X[log_feats].select_dtypes(exclude=['float64','int64']).columns)
        num_log_feats = list(set(log_feats) - set(cat_log_feats))
        
        #one hot enocoding
        cat_df=pd.get_dummies(X.loc[:,cat_log_feats], prefix = cat_log_feats, prefix_sep = ' = ', columns = cat_log_feats, dummy_na=True)
        print ('one hot enocoding')
        #append missingness indicator from numerical, and assert no nan
        if miss_bl:
            na_log_df = X[num_log_feats].isna()
            na_log_df = na_log_df.rename(columns=dict(zip(na_log_df.columns,[c+' = nan'for c in na_log_df.columns])))
            na_log_df = na_log_df.astype(int)
            cat_df = pd.concat([cat_df,na_log_df],axis =1)

        assert cat_df.isna().sum().sum() == 0
        #keep the features with atleast threshold percent of ones
        cat_thr_feats = cat_df.columns[(cat_df==1).mean() >= one_thr]
        print ('keep the features with atleast threshold percent of ones')

        #concat the numerical and one hot features
        X_log = pd.concat([X[num_log_feats],cat_df[cat_thr_feats]], axis =1)
        print ('concat the numerical and one hot features')

        #impute nan in numerical and create indicator
        imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
        scaler = StandardScaler()
        X_log = pd.DataFrame(imputer.fit_transform(X_log),index = X_log.index, columns = X_log.columns)
        print ('impute nan in numerical and create indicator')

        #remove highly correlated features
        #Rearrange the feature columns, so that less perferred  among strongly correlated features can be removed
        #Sort alphabetically Put STD and TREND features in the end
        X_log = X_log.reindex(sorted(X_log.columns), axis=1) 
        print ('remove highly correlated features')

        #Sort by length which puts shorter features in the front
        charcount = X_log.columns.str.count('.')
        char_sort_index = np.argsort(charcount, kind = 'stable')
        X_log = X_log.reindex(X_log.columns[char_sort_index], axis=1)
        print ('Sort by length which puts shorter features in the front')

        #Sort based on number of '.', which implies the level of agregation
        dotcount = X_log.columns.str.count('\.')
        dot_sort_index = np.argsort(dotcount, kind = 'stable')
        X_log = X_log.reindex(X_log.columns[dot_sort_index], axis=1)
        print ("Sort based on number of '.', which implies the level of agregation")

        #generate correlation matrix
        X_log_corr_matrix = X_log.corr()

        #subset relevant features from corr matrix
        X_corr_feats = getcorrfeat_frommat(X_log_corr_matrix,corr_thresh)
        X_log = X_log.drop(columns = X_corr_feats)

        featmap[out] = X_log.shape[1]

        #make splits same as catboost
        print ('make splits same as catboost')

        #check whether to include validation instances in training, default = no
        if train_val_comb_bl:
            X_train_log[out] = X_log.loc[list(X_train[out].index) + list(X_val[out].index),:]
            y_train[out] = y[out].loc[list(y_train[out].index) + list(y_val[out].index)]
        else:
            X_train_log[out] = X_log.loc[X_train[out].index,:]
            X_val_log[out] = X_log.loc[X_val[out].index,:]

        X_test_log[out] = X_log.loc[X_test[out].index,:]
    return X_log, y, X_val_log, y_val, X_train_log, y_train, X_test_log, y_test
