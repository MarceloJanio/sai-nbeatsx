import os
import pickle
import numpy as np
import pandas as pd
import glob
import math
import torch as t

import matplotlib.pyplot as plt

from utils.data.datasets.epf import EPF, EPFInfo
from utils.numpy.metrics import rmae, mae, mape, smape, rmse, gwtest, plot_GW_test_pvals
from utils.numpy.metrics import GW_CPA_test

def compute_loss(name, y, y_hat, y_hat_baseline):
    if name == 'rmae':
        return rmae(y=y, y_hat1=y_hat, y_hat2=y_hat_baseline)
    if name == 'mae':
        return mae(y=y, y_hat=y_hat)
    if name == 'mape':
        return mape(y=y, y_hat=y_hat)
    if name == 'smape':
        return smape(y=y, y_hat=y_hat)
    if name == 'rmse':
        return rmse(y=y, y_hat=y_hat)

def epf_naive_forecast(Y_df):
    """Function to build the naive forecast for electricity price forecasting
    
    The function is used to compute the accuracy metrics MASE and RMAE, the function
    assumes that the number of prices per day is 24. And computes naive forecast for
    days of the week and seasonal Naive forecast for weekends.
        
    Parameters
    ----------
    Y_df : pandas.DataFrame
        Dataframe containing the real prices in long format
        that contains variables ['ds', 'unique_id', 'y']
    
    Returns
    -------
    Y_hat_df : pandas.DataFrame
        Dataframe containing the predictions of the epf naive forecast.
    """
    assert type(Y_df) == pd.core.frame.DataFrame
    assert all([(col in Y_df) for col in ['unique_id', 'ds', 'y']])    

    # Init the naive forecast
    Y_hat_df = Y_df[24 * 7:].copy()
    Y_hat_df['dayofweek'] = Y_df['ds'].dt.dayofweek
    
    # Monday, Saturday and Sunday 
    # we have a naive forecast using weekly seasonality
    weekend_indicator = Y_hat_df['dayofweek'].isin([0,5,6])
    
    # Tuesday, Wednesday, Thursday, Friday 
    # we have a naive forecast using daily seasonality
    week_indicator = Y_hat_df['dayofweek'].isin([1,2,3,4])

    naive = Y_df['y'].shift(24).values[24 * 7:]
    seasonal_naive = Y_df['y'].shift(24*7).values[24 * 7:]
    
    Y_hat_df['y_hat'] = naive * week_indicator + seasonal_naive * weekend_indicator
    return Y_hat_df

#
DATASET = 'FR' # NP, PJM, BE, FR, DE
TIPO = '3' # 0, 1, 2, 3
METRIC = 'MAE'
ENSEMBLE = True

days = 728

TEST_DATE = {'NP': '2016-12-27',
             'PJM':'2016-12-27',
             'BE':'2015-01-04',
             'FR': '2015-01-04',
             'DE':'2016-01-04',
             'BR':'2020-01-04'}

test_date = TEST_DATE[DATASET]
Y_df, X_df, _ = EPF.load_groups(directory='./data', groups=[DATASET])

y_insample_df = Y_df[Y_df['ds']<test_date].reset_index(drop=True)
X_t_insample_df = X_df[X_df['ds']<test_date].reset_index(drop=True)
y_outsample_df = Y_df[Y_df['ds']>=test_date].reset_index(drop=True)
X_t_outsample_df = X_df[X_df['ds']>=test_date].reset_index(drop=True)

y_insample = y_insample_df['y']
y_outsample = y_outsample_df['y']

Y_df = y_insample_df.append(y_outsample_df).reset_index(drop=True)
y_hat_naive = epf_naive_forecast(Y_df=Y_df)
y_hat_naive = y_hat_naive['y_hat'][-len(y_outsample):].values

# y_hat_dicts = pickle.load(open(f'C:/Users/Marcelo Janio/Desktop/nbeatsx-attention/results/FR/nbeats_x/result_test_FR11iter300.p', 'rb'))

y_hat_dicts1 = pickle.load(open(f'C:/Users/Marcelo Janio/Desktop/nbeatsx-attention/results/FR/nbeats_x/result_test_FR00iter300.p', 'rb'))
# y_hat_dicts2 = pickle.load(open(f'C:/Users/Marcelo Janio/Desktop/nbeatsx-attention/results/FR/nbeats_x/result_test_FR01iter300.p', 'rb'))
y_hat_dicts3 = pickle.load(open(f'C:/Users/Marcelo Janio/Desktop/nbeatsx-attention/results/FR/nbeats_x/result_test_FR10iter300.p', 'rb'))
y_hat_dicts4 = pickle.load(open(f'C:/Users/Marcelo Janio/Desktop/nbeatsx-attention/results/FR/nbeats_x/result_test_FR11iter300.p', 'rb'))


print(y_hat_dicts1['run_time'])
# print(y_hat_dicts2['run_time'])
print(y_hat_dicts3['run_time'])
print(y_hat_dicts4['run_time'])

ensemble = []
for i in range(len(y_hat_dicts1['y_hat'])):
    ensemble.append((y_hat_dicts1['y_hat'][i] + y_hat_dicts2['y_hat'][i] + y_hat_dicts3['y_hat'][i] + y_hat_dicts4['y_hat'][i])/4)

y_hat_dicts = {'y_hat': ensemble}

run_loss_mae = np.round(compute_loss(name='mae', y=y_outsample, y_hat=y_hat_dicts['y_hat'], y_hat_baseline=y_hat_naive), 2)
print('MAE:', run_loss_mae)

run_loss_rmae = np.round(compute_loss(name='rmae', y=y_outsample, y_hat=y_hat_dicts['y_hat'], y_hat_baseline=y_hat_naive), 2)
print('rMAE:', run_loss_rmae)

run_loss_mape = np.round(compute_loss(name='mape', y=y_outsample, y_hat=y_hat_dicts['y_hat'], y_hat_baseline=y_hat_naive), 2)
print('MAPE:', run_loss_mape)

# run_loss_smape = np.round(compute_loss(name='smape', y=y_outsample, y_hat=y_hat_dicts['y_hat'], y_hat_baseline=y_hat_naive), 2)
# print('sMAPE:', run_loss_smape)

run_loss_rmse = np.round(compute_loss(name='rmse', y=y_outsample, y_hat=y_hat_dicts['y_hat'], y_hat_baseline=y_hat_naive), 2)
print('RMSE:',run_loss_rmse)


print(y_outsample.shape)
#plot first 300
plt.figure(figsize=(20, 5))
plt.plot(y_outsample, label='Real')
#plt.plot(y_hat_naive[:300], label='Naive')
plt.plot(y_hat_dicts['y_hat'], label='NBEATS')
plt.legend()
plt.show()