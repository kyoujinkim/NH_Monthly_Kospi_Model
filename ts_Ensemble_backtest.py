import multiprocessing
import os
import pickle
from glob import glob

import pandas as pd
import numpy as np
import datetime as dt

from model.parameter_setting import learning_setting
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate

import warnings

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def ensemble_learning(var_path_each):

    results = []

    #set column name of label(weekly return of Kospi)
    level = 'y'

    #calculate monthly return by product returns of 4 weeks
    steps = 4

    #Load hyperparameter setting
    parameters = learning_setting(var_path_each)


    #read PCA data from path
    with open(var_path_each, 'rb') as f:
        data = pickle.load(f)

    #drop unnecessary data
    data = data.drop('KospiDeT', axis=1)
    #resample data
    X = pd.concat([data, w_rtn], axis=1) \
        .resample('W-Fri').last().dropna()

    #train Ridge Regressor
    forecaster = ForecasterAutoregMultiVariate(
        regressor=Ridge(
                        random_state =123,
                        alpha        =parameters['RidgeSetting']['alpha'],
                        fit_intercept=parameters['RidgeSetting']['fit_intercept']
        ),
        level    =level,
        lags     =parameters['RidgeSetting']['Lags'],
        steps    =steps
    )
    forecaster.fit(series=X)
    prediction = forecaster.predict(steps=steps)
    result = ((1 + prediction).product() - 1).values * 100
    results.extend(result)

    #train LGBM Regressor
    forecaster = ForecasterAutoregMultiVariate(
        regressor=LGBMRegressor(random_state     =123,
                                boosting_type    =parameters['LGBMSetting']['boosting_type'],
                                colsample_bytree =parameters['LGBMSetting']['colsample_bytree'],
                                learning_rate    =0.1,
                                max_depth        =parameters['LGBMSetting']['max_depth'],
                                min_child_samples=parameters['LGBMSetting']['min_child_samples'],
                                objective        ='regression'),
        level    =level,
        lags     =parameters['LGBMSetting']['Lags'],
        steps    =steps
    )
    forecaster.fit(series=X)
    prediction = forecaster.predict(steps=steps)
    result = ((1 + prediction).product() - 1).values * 100
    results.extend(result)

    #train RandomForest Regressor
    forecaster = ForecasterAutoregMultiVariate(
        regressor=RandomForestRegressor(random_state    =123,
                                        n_estimators    =parameters['RandomForestSetting']['n_estimators'],
                                        min_samples_leaf=parameters['RandomForestSetting']['min_samples_leaf'],
                                        max_features    =parameters['RandomForestSetting']['max_features'],
                                        max_depth       =parameters['RandomForestSetting']['max_depth'],
                                        n_jobs          =-1),
        level    =level,
        lags     =parameters['RandomForestSetting']['Lags'],  # 26
        steps    =steps
    )
    forecaster.fit(series=X)
    prediction = forecaster.predict(steps=steps)
    result = ((1 + prediction).product() - 1).values * 100
    results.extend(result)

    return results

#Load saved PCA variable from directory
var_path = glob('variables/*')

#set path for result
result_path = './....csv'

#Load Kospi data
d_rtn = pd.read_excel('./data/....xlsx', sheet_name='Kdrtn', index_col=0, parse_dates=True)
d_idx = pd.read_excel('./data/....xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)
w_rtn = d_idx.pct_change(periods=7).dropna().rename(columns={'IDX':'y'})
w_rtn.index = w_rtn.index - pd.Timedelta(1,'w')

#to loop through all date, get array of Unique date
date_arr = list(sorted(set([x.split('\\')[1].replace('.pkl','')[-10:] for x in var_path])))[4:-1]

total_results = pd.DataFrame()

if __name__ == "__main__":
    p = multiprocessing.Pool(os.cpu_count()-1)

    for date_each in tqdm(date_arr):
        var_path_by_date = [x for x in var_path if (date_each in x) and ('All' not in x)]

        #make lagging a week to avoid Look ahead bias
        date_each_forward = dt.datetime.strftime(pd.to_datetime(date_each) + dt.timedelta(7), '%Y-%m-%d')
        results = [date_each_forward]

        #get result from models
        result = p.map(ensemble_learning, var_path_by_date)

        #merge result into base array
        results.extend(np.array(result).flatten())
        results_df = pd.DataFrame(results).T.set_index(keys=0)
        total_results = pd.concat([total_results, results_df])

    total_results_df = total_results.sort_index()
    total_results_df.to_csv(result_path)