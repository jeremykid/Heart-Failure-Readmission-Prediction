import featuretools as ft

def feature_engineer(entityset, target_entity = 'observations',
    agg_prim = ['mean','n_most_common','count','percent_true','last','median','max','num_unique','min','std','entropy','trend','all','any','first','skew'],
    trans_prim = ['is_null', 'year', 'is_weekend'],
    where_prim = ['count','median'],
    n_jobs = 4,
    features_only = True,
    verbose = True):
    '''
    Extract Feature useing Deep feature synthesis
    '''
    features = ft.dfs(entityset=entityset, 
                      target_entity=target_entity, 
                      agg_primitives= agg_prim, 
                      trans_primitives= trans_prim, 
                      where_primitives= where_prim,
                      n_jobs=n_jobs,
                      features_only= features_only,
                      verbose=verbose)
    return features