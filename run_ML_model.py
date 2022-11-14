#!/usr/bin/env python3

import xarray as xr # needed for reading data
import pandas as pd # Used for stroing data
import numpy as np
import pickle as pkl  # Needed for saving model objects
import os
from itertools import repeat # Needed for repeating a variable multiple times

import matplotlib.pyplot as plt

import statsmodels.api as sm    # Used for bot he SARIMA and SARIMAX models
from sklearn import metrics     # Used for importing various performance measures
from multiprocessing import Pool

# Depending on the machine on which the code is run, data might be stored in different directories.
# Indicate which machine is used to make sure the path to the data can be found.
# Can either be "local" or "Snellius"

MACHINE = 'Snellius'

if MACHINE == 'Snellius':
    pred_var_path = '/gpfs/work1/0/ctdas/awoude/Ritten/predictor_vars/' # For retrieving the set of aggregated scaling vectors

    SAVE_DIR = pers_file_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/trained_models/' # used for storing the trained model
    
    results_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/results/' # used for storing the trained model
    
elif MACHINE == 'local':
    pred_var_path = './' # For retrieving the set of aggregated scaling vectors

    SAVE_DIR = pers_file_dir = './models/' # used for storing the trained model

    results_dir = './results/'

def eval_model(model, dat, model_name, test_or_train, show_fit=False):
    '''
    Evaluate the model using the provided testing data
    :param model: The model which is to be tested
    :param dat: The data that is to be used for testing the model. Also includes any predictor variables.
                Can both be used for testing the fit on both trianing data and testing data
    :param model_name: The used ML-algorithm
    :param test_or_train: Flag to indicate whether the passed data is training data ot testing data
    :param show_fit: Flag to indicate whether a plot of the fit should be provided
    :return:
    '''
    flux_dat = dat.prior_flux_per_s.values
    if model_name == "SARIMA":
        true_dat = dat.sf_per_eco.values
        if test_or_train == 'test':
            start_index = len(model.fittedvalues)
            final_model = model.append(true_dat)
        elif test_or_train == 'train':
            start_index = 0
            final_model = model
        else:
            raise Exception(f'test_or_train not specified:{test_or_train}')
        prediction = final_model.get_prediction(start=start_index)
        predict_ci = prediction.conf_int()
        pred_dat = prediction.predicted_mean
        if show_fit:

            # Graph
            fig, ax = plt.subplots(figsize=(9,4))
            title = test_or_train + ' data: predicted sf of eco_region ' + str(dat.eco_regions.values)
            ax.set(title=title, xlabel='Date', ylabel='Scaling factor')

            # Plot data points
            dat.plot.scatter(x='time',y='sf_per_eco', ax=ax, label='Observed', c='C00')
            # Plot predictions
            plt.plot(dat.time.values, pred_dat, label='One-step-ahead forecast', c='C01')
            ci = predict_ci
            ax.fill_between(dat.time.values, ci[:,0], ci[:,1], color='C01', alpha=0.1)

            legend = ax.legend(loc='lower right')

            plt.show()
    else:
        raise NotImplementedError(f'Model evaluation of {model_name} not implemented')

    # Make sure all provided datasets heve the same length
    print(f'len true target_dat = {len(true_dat)}, pred target_dat = {len(pred_dat)}, flux_dat = {len(flux_dat)}')
    assert (len(true_dat)==len(pred_dat)) and (len(true_dat)==len(flux_dat)), 'Passed datasets are do not have the same length: '

    # Determine the performance in scaling factor space
    sf_ME = (np.sum(true_dat)-np.sum(pred_dat))/len(true_dat)
    sf_MAE = metrics.mean_absolute_error(true_dat, pred_dat)
    sf_MAPE = metrics.mean_absolute_percentage_error(true_dat, pred_dat)
    sf_RMSE = np.sqrt(metrics.mean_squared_error(true_dat, pred_dat))
    sf_r2 = metrics.r2_score(true_dat, pred_dat)

    # Move evaluation to flux space
    true_flux = true_dat * flux_dat
    pred_flux = pred_dat * flux_dat

    # Determine the performance in flux space
    flux_ME = (np.sum(true_dat)-np.sum(pred_flux))/len(true_dat)
    flux_MAE = metrics.mean_absolute_error(true_flux, pred_flux)
    flux_MAPE = metrics.mean_absolute_percentage_error(true_flux, pred_flux)
    flux_RMSE = np.sqrt(metrics.mean_squared_error(true_flux, pred_flux))
    flux_r2 = metrics.r2_score(true_flux, pred_flux)
    return {'sf_ME_'+test_or_train:sf_ME,
           'sf_MAE_'+test_or_train:sf_MAE,
           'sf_MAPE_'+test_or_train:sf_MAPE,
           'sf_RMSE_'+test_or_train:sf_RMSE,
           'sf_r2_'+test_or_train:sf_r2,
           'flux_ME_'+test_or_train:flux_ME,
           'flux_MAE_'+test_or_train:flux_MAE,
           'flux_MAPE_'+test_or_train:flux_MAPE,
           'flux_RMSE_'+test_or_train:flux_RMSE,
           'flux_r2_'+test_or_train:flux_r2}

def get_model_dir(algorithm, start_year):
    file_dir = SAVE_DIR + algorithm + '/' + start_year + '/'
    return file_dir

def get_file_name(algorithm, eco_region):
    file_name = algorithm + '_' + str(eco_region) + '.pkl'
    return file_name

def get_full_path(algorithm, start_year, eco_region):
    file_dir = get_model_dir(algorithm, start_year)
    file_name = get_file_name(algorithm, eco_region)
    return file_dir + file_name

def write_model(model, model_name, start_year, eco_region):
    '''
    Function used to save a model in the correct directory with an identifiable name. Uses Pickle for saving the model object
    :param model: The model which is to be saved
    :param model_name: The used ML-algorithm
    :param start_year: The date at which the training data starts
    :param eco_region: The name of the eco_region to which the model applies
    :return: None
    '''
    file_dir = get_model_dir(model_name, start_year)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    file_name = get_file_name(model_name, eco_region)
    file = file_dir + file_name 
    with open(file, "wb") as f:
        pkl.dump(model, f, protocol=5)

def read_model(file_path):
    with open(file_path, 'rb') as f:
        model = pkl.load(f)
    print(f"finished loading model {file_path}")
    return model

def train_model(train_dat, model_name, eco_region):
    '''
    Function for training a model on the provided training data
    :param train_dat: The data used for training. Includes both target data and predictor data
    :param model_name: Name of the ML-algorithm to be used for training
    :return: A trianed model
    '''
    
    start_year = str(train_dat.time.dt.year.min().values)
    
    print(f"starting process for {model_name}, at eco_region {eco_region} using data starting at {start_year}")
    
    if model_name == 'SARIMA':
        target_data = train_dat.sf_per_eco
        model = sm.tsa.statespace.SARIMAX(target_data.values,
                                         order=(2,0,2),             # Defining the regular AR, I and MA dependencies
                                         seasonal_order=(1,0,1,52),     # Defining the seasonal dependencies
                                         trend = 'c'                # Adding an intercept term)
                                          )
        fitted_model=model.fit(maxiter=100) # method='cg'
    else:
        raise NotImplementedError(f'Training of model {model_name} has not been implemented')
#     eco_region = str(train_dat.eco_regions.values)

    # Save model for future usage
    write_model(fitted_model, model_name, start_year, eco_region)
    return fitted_model

def test_eco_region(eco_dat, model_name):
    region, data = eco_dat
    results_df = pd.DataFrame()
    
    #Set aside the testing data. Using the classical 80%-20% split
    test_ds = data.loc[dict(time=slice("2017", "2020"))]
    region_dat = data.loc[dict(time=slice("2000", "2016"))]
    for year in range(2000, 2017):
        # Load the model if it has been trained already
        file_name = get_full_path(model_name, str(year), region)
        
        # Determine training data and train model
        train_ds = region_dat.loc[dict(time=slice(str(year), "2016"))]

        try:
            trained_model = read_model(file_name)
            print(f'file "{file_name}" has already contains a trained model. Skipping training process')
        except pkl.UnpicklingError:
            print(f'Could no unpickle model from eco-region {region} and starting year {year}. Training new model')
            # Pickled model somehow got corrupted. Train a new model
            trained_model = train_model(train_ds, model_name, region)
        except (EOFError, FileNotFoundError):
            print(f'No file exists, or the the existing file is empty. Training new model')
            # If no model exists, train a new one
            trained_model = train_model(train_ds, model_name, region)

        # Evaluate the model, both on training and testing data
        train_results = eval_model(trained_model, train_ds, model_name, 'train', show_fit=False)
        test_results = eval_model(trained_model, test_ds, model_name, 'test', show_fit=False)
        model_params = {
                'eco_region':region,
                'start_year':year,
                'N_train_years':(2017-year),
                'N_train_obs':len(train_ds.time),
                'N_test_years':4,
                'N_test_obs':len(test_ds.time)
        }

        # unpack all dicts to form single results dict
        model_results = {**model_params, **train_results, **test_results}
        if len(results_df) == 0:
            results_df = pd.DataFrame(model_results, index=[0])
        else:
            results_df = results_df.append(model_results, ignore_index=True)
    return results_df

def run_model(model_name, complete_ds):
    eco_region_dat = list(complete_ds.groupby("eco_regions"))
    eco_region_dat = [(region, data.load(scheduler='sync')) for region, data in eco_region_dat]
    with Pool(32) as pool:
        list_of_results = pool.starmap(test_eco_region, zip(eco_region_dat, repeat(model_name)))
    results = pd.concat(list_of_results)
    return results

# Loading all necessary data
with xr.open_dataset(pred_var_path + 'vars_per_eco_update.nc') as ds:
    complete_ds = ds

results = run_model('SARIMA', complete_ds)

results_file = results_dir + 'SARIMA_results.pkl'
print(results)

results.to_pickle(results_file)

