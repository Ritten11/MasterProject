#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
from glob import glob
import datetime as dtm
import xarray as xr
import os


MAIN_DIR = '/projects/0/ctdas/awoude/Ritten/transport_results/'
SAVE_DIR = '/gpfs/home5/awoude/Ritten/EKF/Results/'

def get_cte(sites, run):
    """Get CTE mole fractions"""
    dfs = []
    path = MAIN_DIR + run + '/'
    files = sorted(glob(path + '*.nc'))
    for f in files:
        with nc.Dataset(f) as ds:
            name = ds.site_code
            if not name in sites: continue
            df = pd.DataFrame()
            simulated = (ds['modelsamplesensemble'][:, 1:].sum(axis=1) + ds['modelsamplesmean'][:]) * 1e6
            inds = np.where(simulated.mask==False)
            simulated = simulated[inds]#.reshape(-1, 5)
            times = np.array(nc.num2date(ds['time'][inds], ds['time'].units))
            times = [dtm.datetime(t.year, t.month, t.day, t.hour) for t in times]
            observed = ds['value'][inds] * 1e6
            height = ds['intake_height'][inds]
            df['time'] = times
            df['simulated'] = simulated
            df['observed'] = observed
            df['name'] = name
#             df['height'] = height # Kan je toevoegen, hoeft voor nu niet denk ik
            dfs.append(df)
    df = pd.concat(dfs)
    df.index = [df['time'], df['name']]
    df = df.drop(['time', 'name'], axis=1)
    return df

def get_error(targets, predictions):
    """Calculate the difference between targets and predictions"""
    return targets - predictions
def get_mse(targets, predictions):
    """Mean Square Error"""
    return (get_error(targets, predictions) ** 2).mean()
def get_rmse(targets, predictions):
    """Root mean square error"""
    return np.sqrt(get_mse(targets, predictions))
def get_index_of_agreement(targets, predictions):
    """Calculate the index of agreement
    Willmott (1981)
    the ratio of the mean square error and the potential error."""
    mse = get_mse(targets, predictions)
    t_mean = targets.mean()
    err_pot = np.mean((np.abs(predictions - t_mean) + np.abs(targets - t_mean))**2)
    return 1 - (mse / err_pot)
def get_correlation(targets, predictions):
    """Calculate the correlation coefficient"""
    return np.corrcoef(targets, predictions)[0,1]
def get_ame(targets, predictions):
    """Absolute maximum error"""
    return np.abs(get_error(targets, predictions)).max()
def get_mae(targets, predictions):
    """Mean absolute error"""
    return np.abs(get_error(targets, predictions)).mean()
def get_me(targets, predictions):
    """Mean error"""
    return (get_error(targets, predictions)).mean()
def get_bias(targets, predictions):
    """Calculate the bias"""
    error = np.array(predictions) - np.array(targets)
    bias = np.mean(error)
    return bias
def get_abs_bias(targets, predictions):
    """Calculate the bias"""
    error = np.abs(np.array(predictions) - np.array(targets))
    bias = np.mean(error)
    return bias
def get_taylor(targets, predictions):
    """Calculate the taylor skill score (2001)"""
    r = get_correlation(targets, predictions)
    r0 = 0.97
    sf = np.std(targets) / np.std(predictions)
    
    s = (4*(1 + r)**4) / ((sf + 1/sf)**2 * (1 + r0)**4)
    return s


def evaluate_site(df, site, measures):
    data = df.xs(site, level=1)
    sim = data['simulated']
    obs = data['observed']

    measure_dict = {}
    for measure in measures:
        if measure == 'N':
            value = len(data)
        if measure == 'error':
            value = get_error(sim, obs)
        if measure == 'mse':
            value = get_mse(sim,obs)
        if measure == 'rmse':
            value = get_rmse(sim,obs)
        if measure == 'index_of_agreement':
            value = get_index_of_agreement(sim,obs)
        if measure == 'correlation':
            value = get_correlation(sim, obs)
        if measure == 'ame':
            value = get_ame(sim, obs)
        if measure == 'mae':
            value = get_mae(sim, obs)
        if measure == 'me':
            value = get_me(sim, obs)
        if measure == 'bias':
            value = get_bias(sim, obs)
        if measure == 'abs_bias':
            value = get_abs_bias(sim, obs)
        if measure == 'taylor':
            value = get_taylor(sim, obs)
        measure_dict[measure] = value
    
    return measure_dict

files_CTE2018 = [os.path.basename(x) for x in glob(MAIN_DIR + 'CTE2018/' + '*.nc')]
sites_CTE2018 = set([x.split('_')[1] for x in files_CTE2018])

files_GCPsib4 = [os.path.basename(x) for x in glob(MAIN_DIR + 'GCPsib4/' + '*.nc')]
sites_GCPsib4 = set([x.split('_')[1] for x in files_GCPsib4])

files_monthly_avg = [os.path.basename(x) for x in glob(MAIN_DIR + 'monthly_avg/' + '*.nc')]
sites_monthly_avg = set([x.split('_')[1] for x in files_monthly_avg])

common_sites = sites_CTE2018.intersection(sites_monthly_avg.intersection(sites_GCPsib4))

set_of_sites = {x.upper() for x in common_sites}

runs = {'CTE2018', 'monthly_avg', 'GCPsib4'}

measures = ['N','rmse', 'me', 'abs_bias']

print(f'Analysing the sites {set_of_sites}\n')
dfs = []
for run in runs:
    raw_dat = get_cte(set_of_sites, run)
    print('----Data had been loaded----')
    data_dict = {}
    for site in set_of_sites:
        performance_dict = evaluate_site(raw_dat, site, measures)
        print(measures)
        print(performance_dict.keys())
        data_dict[site] = list(performance_dict.values())
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=measures)
    df['run'] = run
    dfs.append(df)
    print(df)
data = pd.concat(dfs)
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols].sort_index()
print(data)
save_file = SAVE_DIR + 'run_evaluation.pkl'
data.to_pickle(save_file)