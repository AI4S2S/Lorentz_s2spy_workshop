import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Union
import itertools



def corrcoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]

def r2_score(y_true, y_pred, multioutput='variance_weighted'):
    return metrics.r2_score(y_true, y_pred, multioutput=multioutput)

class ErrorSkillScore:
    def __init__(self, constant_bench: float=False, squared=False):
        '''
        Parameters
        ----------
        y_true : pd.DataFrame or pd.Series or np.ndarray
        y_pred : pd.DataFrame or pd.Series or np.ndarray
        benchmark : float, optional
            DESCRIPTION. The default is None.
        squared : boolean value, optional (default = True)
            If True returns MSE value, if False returns RMSE value

        Returns
        -------
        RMSE (Skill Score).

        '''
        if type(constant_bench) in [float, int, np.float_]:
            self.benchmark = float(constant_bench)
        elif type(constant_bench) in [np.ndarray, pd.Series, pd.DataFrame]:
            self.benchmark = np.array(constant_bench, dtype=float)
        else:
            print('benchmark is set to False')
            self.benchmark = False

        self.squared = squared
        # if type(self.benchmark) is not None:


    def RMSE(self, y_true, y_pred):
        self.RMSE_score = metrics.mean_squared_error(y_true, y_pred,
                                              squared=self.squared)
        if self.benchmark is False:
            return self.RMSE_score
        elif type(self.benchmark) is float:
            b_ = np.zeros(y_true.size) ; b_[:] = self.benchmark
        elif type(self.benchmark) is np.ndarray:
            b_  = self.benchmark
        self.RMSE_bench = metrics.mean_squared_error(y_true,
                                           b_,
                                           squared=self.squared)
        return (self.RMSE_bench - self.RMSE_score) / self.RMSE_bench

    def MAE(self, y_true, y_pred):
        fc_score = metrics.mean_absolute_error(y_true, y_pred)
        if self.benchmark is False:
            return fc_score
        elif type(self.benchmark) is float:
            b_ = np.zeros(y_true.size) ; b_[:] = self.benchmark
        elif type(self.benchmark) is np.ndarray:
            b_  = self.benchmark
        self.MAE_bench = metrics.mean_absolute_error(y_true, b_)
        return (self.MAE_bench - fc_score) / self.MAE_bench

    def BSS(self, y_true, y_pred):
        self.brier_score = metrics.brier_score_loss(y_true, y_pred)
        if self.benchmark is False:
            return self.brier_score
        elif type(self.benchmark) is float:
            self.b_ = np.zeros(y_true.size) ; self.b_[:] = self.benchmark
        elif type(self.benchmark) is np.ndarray:
            self.b_  = self.benchmark
        self.BS_bench = metrics.brier_score_loss(y_true, self.b_)
        return (self.BS_bench - self.brier_score) / self.BS_bench

class binary_score:
    def __init__(self, threshold: float=0.5):
        self.threshold = threshold

    def precision(self, y_true, y_pred):
        y_pred_b = y_pred > self.threshold
        return round(metrics.precision_score(y_true, y_pred_b)*100,0)

    def accuracy(self, y_true, y_pred):
        #  P(class=0) * P(prediction=0) + P(class=1) * P(prediction=1)
        y_pred_b = y_pred > self.threshold
        return round(metrics.accuracy_score(y_true, y_pred_b)*100,0)


def AUC_SS(y_true, y_pred):
    # from http://bibliotheek.knmi.nl/knmipubIR/IR2018-01.pdf eq. 1
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    auc_bench = .5
    return (auc_score - auc_bench) / (1-auc_bench)

class CRPSS_vs_constant_bench:
    def __init__(self, constant_bench: float=False, return_mean=True,
                 weights: np.ndarray=None):
        '''
        Parameters
        ----------
        y_true : pd.DataFrame or pd.Series or np.ndarray
        y_pred : pd.DataFrame or pd.Series or np.ndarray
        benchmark : float, optional
            DESCRIPTION. The default is None.
        return_mean: boolean value, optional (default = True)
            If True mean CRPSS instead of array of size len(y_true)
        weights : array_like, optional
            If provided, the CRPS is calculated exactly with the assigned
            probability weights to each forecast. Weights should be positive, but
            do not need to be normalized. By default, each forecast is weighted
            equally.
        Returns
        -------
        if return_mean == False (default):
            mean CRPSS versus benchmark
        if return_mean:
            mean CRPSS versus benchmark and continuous evaluation of forecasts

        '''
        self.benchmark = constant_bench
        self.return_mean = return_mean
        self.weights = weights
        # if type(self.benchmark) is not None:

        # return metrics.mean_squared_error(y_true, y_pred, squared=root
    def CRPSS(self, y_true, y_pred):
        fc_score = ps.crps_ensemble(y_true, y_pred,
                                    weights=self.weights)
        if self.return_mean:
            fc_score = fc_score.mean()
        if self.benchmark is False:
            return fc_score
        elif type(self.benchmark) in [float, int]:
            b_ = np.zeros_like(y_true) ; b_[:] = self.benchmark
            bench = ps.crps_ensemble(y_true, b_,
                                    weights=self.weights)
            if self.return_mean:
                bench = bench.mean()
            return (bench - fc_score) / bench


def get_scores(prediction, df_splits: pd.DataFrame=None, score_func_list: list=None,
               score_per_test=False, n_boot: int=1, blocksize: int=1,
               rng_seed=1):
    '''


    Parameters
    ----------
    prediction : TYPE
        DESCRIPTION.
    df_splits : pd.DataFrame, optional
        DESCRIPTION. The default is None.
    score_func_list : list, optional
        DESCRIPTION. The default is None.
    score_per_test : TYPE, optional
        DESCRIPTION. The default is True.
    n_boot : int, optional
        DESCRIPTION. The default is 1.
    blocksize : int, optional
        DESCRIPTION. The default is 1.
    rng_seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    pd.DataFrames format:
    index [opt. splits]
    Multi-index columns [lag, metric name]
    df_trains, df_test_s, df_tests, df_boots.

    '''
    #%%
    if df_splits is None and 'TrainIsTrue' not in prediction.columns:
        # assuming all is test data
        TrainIsTrue = np.zeros((prediction.index.size, 1))
        RV_mask  = np.ones((prediction.index.size, 1))
        df_splits = pd.DataFrame(np.concatenate([TrainIsTrue,RV_mask], axis=1),
                                   index=prediction.index,
                                   dtype=bool,
                                   columns=['TrainIsTrue', 'RV_mask'])
    elif df_splits is None and 'TrainIsTrue' in prediction.columns:
        # TrainIsTrue columns are part of prediction
        df_splits = prediction[['TrainIsTrue', 'RV_mask']]



    # add empty multi-index to maintain same data format
    if hasattr(df_splits.index, 'levels')==False:
        df_splits = pd.concat([df_splits], keys=[0])

    if hasattr(prediction.index, 'levels')==False:
        prediction = pd.concat([prediction], keys=[0])

    columns = [c for c in prediction.columns[:] if c not in ['TrainIsTrue', 'RV_mask']]
    if 'TrainIsTrue' not in prediction.columns:
        pred = prediction.merge(df_splits,
                                left_index=True,
                                right_index=True)
    else:
        pred = prediction


    # score on train and per test split
    if score_func_list is None:
        score_func_list = [metrics.mean_squared_error, corrcoef]
    splits = pred.index.levels[0]
    columns = np.array(columns[1:])
    df_trains = np.zeros( (columns.size), dtype=object)
    df_tests_s = np.zeros( (columns.size), dtype=object)
    for c, col in enumerate(columns):
        df_train = pd.DataFrame(np.zeros( (splits.size, len(score_func_list))),
                            columns=[f.__name__ for f in score_func_list])
        df_test_s = pd.DataFrame(np.zeros( (splits.size, len(score_func_list))),
                            columns=[f.__name__ for f in score_func_list])
        for s in splits:
            sp = pred.loc[s]
            not_constant = True
            if np.unique(sp.iloc[:,0]).size == 1:
                not_constant = False
            trainRV = np.logical_and(sp['TrainIsTrue']==1, sp['RV_mask']==True)
            testRV  = np.logical_and(sp['TrainIsTrue']==0, sp['RV_mask']==True)
            for f in score_func_list:
                name = f.__name__
                if (~trainRV).all()==False and not_constant: # training data exists
                    train_score = f(sp[trainRV].iloc[:,0], sp[trainRV].loc[:,col])
                else:
                    train_score  = np.nan
                if score_per_test and testRV.any() and not_constant:
                    test_score = f(sp[testRV].iloc[:,0], sp[testRV].loc[:,col])
                else:
                    test_score = np.nan

                df_train.loc[s,name] = train_score
                df_test_s.loc[s,name] = test_score
        df_trains[c] = df_train
        df_tests_s[c]  = df_test_s
    df_trains = pd.concat(df_trains, keys=columns, axis=1)
    df_tests_s = pd.concat(df_tests_s, keys=columns, axis=1)


    # score on complete test
    df_tests = np.zeros( (columns.size), dtype=object)
    pred_test = get_df_test(pred).iloc[:,:-2]
    if pred_test.size != 0 : # ensure test data is available
        for c, col in enumerate(columns):
            df_test = pd.DataFrame(np.zeros( (1,len(score_func_list))),
                                    columns=[f.__name__ for f in score_func_list])
            for f in score_func_list:
                name = f.__name__
                y_true = pred_test.iloc[:,0]
                y_pred = pred_test.loc[:,col]
                if np.unique(y_true).size >= 2:
                    df_test[name] = f(y_true, y_pred)
                else:
                    if c == 0:
                        print('Warning: y_true is constant. Returning NaN.')
                    df_test[name] = np.nan
            df_tests[c]  = df_test
        df_tests = pd.concat(df_tests, keys=columns, axis=1)


    # Bootstrapping with replacement
    df_boots = np.zeros( (columns.size), dtype=object)
    if pred_test.size != 0: # ensure test data is available
        for c, col in enumerate(columns):
            old_index = range(0,len(y_true),1)
            n_bl = blocksize
            chunks = [old_index[n_bl*i:n_bl*(i+1)] for i in range(int(len(old_index)/n_bl))]
            if np.unique(y_true).size > 1 or n_boot==0:
                score_list = _bootstrap(pred_test.iloc[:,[0,c+1]], n_boot,
                                        chunks, score_func_list,
                                        rng_seed=rng_seed)
            else:
                score_list  = np.repeat(np.nan,
                                        n_boot*len(score_func_list)).reshape(n_boot, -1)

            df_boot = pd.DataFrame(score_list,
                                   columns=[f.__name__ for f in score_func_list])
            df_boots[c] = df_boot
        df_boots = pd.concat(df_boots, keys=columns, axis=1)

    out = (df_trains, df_tests_s, df_tests, df_boots)

#%%
    return out

def _bootstrap(pred_test, n_boot_sub, chunks, score_func_list, rng_seed: int=1):

    y_true = pred_test.iloc[:,0]
    y_pred = pred_test.iloc[:,1]
    score_l = []
    rng = np.random.RandomState(rng_seed) ; i = 0 ; r = 0
    while i != n_boot_sub:
        i += 1 # loop untill n_boot
        # bootstrap by sampling with replacement on the prediction indices
        ran_ind = rng.randint(0, len(chunks) - 1, len(chunks))
        ran_blok = [chunks[i] for i in ran_ind] # random selection of blocks
        indices = list(itertools.chain.from_iterable(ran_blok)) #blocks to list

        if len(np.unique(y_true[indices])) < 2:
            i -= 1 ; r += 1 # resample and track # of resamples with r
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if r <= 100:
            score_l.append([f(y_true[indices],
                              y_pred[indices]) for f in score_func_list])
        else: # after 100 resamples, plug in NaNs
            score_l.append([np.nan for i in range(len(score_func_list))])
            if i == n_boot_sub:
                print(f'Too many ({r}) resample attempts to get both negative '
                      'and positive samples of truth, returning NaNs')

    return score_l

def cond_fc_verif(df_predict: pd.DataFrame,
                  df_forcing: pd.DataFrame,
                  df_splits: pd.DataFrame,
                  score_func_list: list=None,
                  quantiles:list =[.25],
                  n_boot: int=0):
    ''' Calculate metrics on seperate time indices. Split in time indices is
    determined by anomalous states of the df_forcing timeseries. The quantiles
    determine 'how anomalous' the seperation is.

    Parameters
    ----------
    df_predict : pd.DataFrame
        Out of sample prediction with multi-index [split, time].
    df_forcing : pd.DataFrame
        Out of sample forcing timeseries with multi-index [split, time].
        Calculates an equal weighted mean over columns to get 1-d timeseries
    df_splits : pd.DataFrame
        Train-test split masks with multi-index [split, time].
    score_func_list : list, optional
        list with scoring metrics. The default is None.
    quantiles : list, optional
        list with quantiles (q) to split the time indices.
        e.g., when q=0.25, time indices will be split based on df_forcing
        being below the 0.25q and above 0.75q, i.e. anomalous.
        The default is [.25].
    n_boot : int, optional
        n times bootstrapping skill metrics.
        The default is 0.

    Returns
    -------
    pd.DataFrame, metric names are the index (rows) and columns are the strong
    and weak quantile subsets. For example, [strong 50%, weak 50%] for q=.25.

    '''
    #%%
    df_forctest = get_df_test(df_forcing.mean(axis=1),
                                           df_splits=df_splits)

    df_test = get_df_test(df_predict,
                                       df_splits=df_splits)

    metrics = [s.__name__ for s in score_func_list]
    if n_boot > 0:
        cond_df = np.zeros((len(metrics), len(quantiles)*2, n_boot))
    else:
        cond_df = np.zeros((len(metrics), len(quantiles)*2))
    stepsize = 1 if len(quantiles)==1 else len(quantiles)*2
    for i, met in enumerate(metrics):
        for k, l in enumerate(range(0,stepsize,2)):
            q = quantiles[k]

            # =============================================================
            # Strong forcing
            # =============================================================
            # extrapolate quantile values based on training data
            q_low = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':q})
            # Extract out-of-sample quantile values
            q_low = get_df_test(q_low,
                                               df_splits=df_splits)

            q_high = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':1-q})
            q_high = get_df_test(q_high,
                                               df_splits=df_splits)

            low = df_forctest < q_low.values.ravel()
            high = df_forctest > q_high.values.ravel()
            mask_anomalous = np.logical_or(low, high)
            # anomalous Boundary forcing
            condfc = df_test[mask_anomalous.values]
            # condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
            cond_verif_tuple = get_scores(condfc,
                                                   score_func_list=score_func_list,
                                                   n_boot=n_boot,
                                                   score_per_test=False,
                                                   blocksize=1,
                                                   rng_seed=1)

            df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
            cond_verif_tuple  = cond_verif_tuple
            if n_boot == 0:
                cond_df[i, l] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            else:
                cond_df[i, l, :] = df_boot[df_boot.columns[0][0]][met]
            # =============================================================
            # Weak forcing
            # =============================================================
            q_higher_low = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':.5-q})
            q_higher_low = get_df_test(q_higher_low,
                                               df_splits=df_splits)


            q_lower_high = functions_pp.get_df_train(df_forcing.mean(axis=1),
                                     df_splits=df_splits, s='extrapolate',
                                     function='quantile', kwrgs={'q':.5+q})
            q_lower_high = get_df_test(q_lower_high,
                                               df_splits=df_splits)

            higher_low = df_forctest > q_higher_low.values.ravel()
            lower_high = df_forctest < q_lower_high.values.ravel()

            mask_anomalous = np.logical_and(higher_low, lower_high)

            condfc = df_test[mask_anomalous.values]

            cond_verif_tuple = get_scores(condfc,
                                                   score_func_list=score_func_list,
                                                   n_boot=n_boot,
                                                   score_per_test=False,
                                                   blocksize=1,
                                                   rng_seed=1)
            df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
            if n_boot == 0:
                cond_df[i, l+1] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            else:
                cond_df[i, l+1, :] = df_boot[df_boot.columns[0][0]][met]
    columns = [[f'strong {int(q*200)}%', f'weak {int(q*200)}%'] for q in quantiles]
    columns = functions_pp.flatten(columns)
    if n_boot > 0:
        columns = pd.MultiIndex.from_product([columns, list(range(n_boot))])

    df_cond_fc = pd.DataFrame(cond_df.reshape((len(metrics), -1)),
                              index=list(metrics),
                              columns=columns)
    #%%
    return df_cond_fc

def get_df_test(df, cols: list=None, df_splits: pd.DataFrame=None):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        df with train-test splits on the multi-index.
    cols : list, optional
        return sub df based on columns. The default is None.
    df_splits : pd.DataFrame
        seperate df with TrainIsTrue column specifying the train-test data


    Returns
    -------
    Returns only the data at which TrainIsTrue==False.

    '''
    if df_splits is None:
        splits = df.index.levels[0]
        TrainIsTrue = df['TrainIsTrue']
    else:
        splits = df_splits.index.levels[0]
        TrainIsTrue = df_splits['TrainIsTrue']
    list_test = []
    for s in range(splits.size):
        TestIsTrue = (TrainIsTrue[s]==0.).values
        # get test values
        try: # normal
            test_vals = df.loc[s][TestIsTrue]
        except: # only RV_mask (for predictions)
            TestIsTrue = TestIsTrue[df_splits.loc[s]['RV_mask']]
            test_vals = df.loc[s][TestIsTrue]
        list_test.append(test_vals)


    df = pd.concat(list_test).sort_index()
    if cols is not None:
        df = df[cols]
    return df

def get_df_train(df, cols: list=None, df_splits: pd.DataFrame=None, s=0,
                 function='mean', kwrgs: dict={}):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        df with train-test splits on the multi-index.
    cols : list, optional
        return sub df based on columns. The default is None.
    df_splits : pd.DataFrame
        seperate df with TrainIsTrue column specifying the train-test data
    s : int or str
        if int, it will select the training sample s.
        if s == 'squeeze', the function (e.g. mean) is calculated across
        training samples.
        if s == 'extrapolate', the function computes the mean/quantile/std of
        each training sample and extrapolates (fills all timesteps).

    function : str
        Call attribute of pd.DataFrame (e.g. mean, quantile, std)


    Returns
    -------
    Returns only the data at which TrainIsTrue.

    '''
    if cols is not None:
        df = df[cols]
    if df_splits is None:
        TrainIsTrue = df['TrainIsTrue']
    else:
        TrainIsTrue = df_splits['TrainIsTrue']
    if type(s) is int:
        df_train = df.loc[s][TrainIsTrue.loc[s].values==1]
    elif s == 'squeeze' or s == 'extrapolate': # mean over all training data
        if type(df) is pd.Series:
            df = pd.DataFrame(df)
        df_trains = []
        for col in df.columns:
            splits = TrainIsTrue.index.levels[0]
            l_dfs = [df[col].loc[s][TrainIsTrue.loc[s].values==1] for s in splits]

            if s == 'squeeze':
                df_coltrain = pd.concat(l_dfs, axis=1)
                df_coltrain = df_coltrain.groupby(by=df_coltrain.columns,
                                                  axis=1)
                df_coltrain = getattr(df_coltrain, function)(**kwrgs)
                df_coltrain = pd.DataFrame(df_coltrain, columns=[col])
            elif s == 'extrapolate':
                df_coltrain = [getattr(d, function)(**kwrgs) for d in l_dfs]
                df_coltrain = np.repeat(np.array(df_coltrain).reshape(splits.size,1),
                                        TrainIsTrue.loc[0].index.size, axis=0)
                df_coltrain = pd.DataFrame(df_coltrain.reshape(-1,1), index=df.index,
                                           columns=[str(col)+'_'+function])
            df_trains.append(df_coltrain)
        df_train = pd.concat(df_trains, axis=1)
    return df_train