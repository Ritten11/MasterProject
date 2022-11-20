#!/usr/bin/env python3

import xarray as xr # needed for reading data
import pandas as pd # Used for stroing data
import numpy as np
import pickle as pkl  # Needed for saving model objects
import os
from itertools import repeat # Needed for repeating a variable multiple times

import matplotlib.pyplot as plt

import statsmodels.api as sm    # Used for bot he SARIMA and SARIMAX models
import statsmodels.tsa as sm_tsa # Used for type checking SARIMA models
from sklearn import metrics     # Used for importing various performance measures
from multiprocessing import Pool

from json import dumps # Needed for casting a dict to a string

# Depending on the machine on which the code is run, data might be stored in different directories.
# Indicate which machine is used to make sure the path to the data can be found.
# Can either be "local" or "Snellius"

MACHINE = 'Snellius'

if MACHINE == 'Snellius':
    pred_var_path = '/gpfs/work1/0/ctdas/awoude/Ritten/predictor_vars/' # For retrieving the set of aggregated scaling vectors

    SAVE_DIR = pers_file_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/trained_models/' # used for storing the trained model

    SF_DIR = '/gpfs/work1/0/ctdas/awoude/Ritten/fitted_sf/' # used for storing the scaling factor produced bij ML models

    RESULTS_DIR = '/gpfs/work1/0/ctdas/awoude/Ritten/results/' # used for storing the trained model

    CPU_COUNT = 32 # Snellius allows for usage of ut to 32 threads on the same node without additional costs

elif MACHINE == 'local':
    pred_var_path = './' # For retrieving the set of aggregated scaling vectors

    SAVE_DIR = pers_file_dir = './models/' # used for storing the trained model

    SF_DIR = './fitted_sf/' # used for storing the scaling factor produced bij ML models

    RESULTS_DIR = './results/'

    CPU_COUNT = multiprocessing.cpu_count() - 2 # 2 CPUs are subtracted to maintain a relatively fast PC when running the model
else:
    raise NotImplementedError(f'machine "{MACHINE}" has not been implemented')



SARIMA_params = {'order': (2,0,2),             # Defining the regular AR, I and MA dependencies
                 'seasonal_order': (1,0,1,52), # Defining the seasonal dependencies
                 'trend': 'c'                  # Adding an intercept term
                 }

transCom_dict = {1:'North American Boreal',
                   2:'North American Temperate',
                   7:'Eurasia Boreal',
                   8:'Eurasia Temperate',
                   11:'Europe',
                  }

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
            title = test_or_train + ' data: predicted sf of ecoregion ' + str(dat.eco_regions.values)
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
    '''
    The path to the directory in which the model should be saved
    :param algorithm: The algorithm used for creating the model, sometimes also refered to as 'model_name'
    :param start_year: The year at which the training data started
    :return: Directory of the (to be) saved model
    '''
    file_dir = SAVE_DIR + algorithm + '/' + start_year + '/'
    return file_dir

def get_file_name(algorithm, eco_region, extention):
    '''
    Function used for generating the file name of a model
    :param algorithm: The algorithm used for creating the model, sometimes also refered to as 'model_name'
    :param eco_region: The eco_region to which the model applies
    :return: file name of the (to be) saved model
    '''
    file_name = algorithm + '_' + str(eco_region) + '.' + extention
    return file_name

def get_model_path(algorithm, start_year, eco_region):
    '''
    Function for automatically generating the location of a saved model
    :param algorithm: The algorithm used for creating the model, sometimes also refered to as 'model_name'
    :param start_year: The year at which the training data started
    :param eco_region: The eco_region to which the model applies
    :return: The correct file path of the model.
    '''
    file_dir = get_model_dir(algorithm, start_year)
    file_name = get_file_name(algorithm, eco_region, 'pkl')
    return file_dir + file_name

def get_sf_dir(algorithm):
    '''
    Function for automatically generating the location of the fitted scaling factors
    :param algorithm: The algorithm used for creating the model, sometimes also refered to as 'model_name'
    :return: The directory at which the results file should be stored.
    '''
    file_dir = SF_DIR + algorithm + '/'
    return file_dir

def get_sf_path(algorithm, eco_region):
    '''
    Function for automatically generating the full path to the fitted scaling factors
    :param algorithm: The algorithm used for generating the scaling factors, sometimes also refered to as 'model_name'
    :param eco_region: The eco_region to which the scaling factors apply
    :return: The correct file path of the scaling factors.
    '''
    file_dir = get_sf_dir(algorithm)
    file_name = get_file_name(algorithm, eco_region, 'nc')
    return file_dir + file_name

def get_results_dir(algorithm):
    '''
    Function for automatically generating the location of the analysed results
    :param algorithm: The algorithm used for creating the model, sometimes also refered to as 'model_name'
    :return: The directory at which the results file should be stored.
    '''
    file_dir = RESULTS_DIR + algorithm + '/'
    return file_dir

def get_results_path(algorithm, eco_region):
    '''
    Function for automatically generating the full path to the analysed results
    :param algorithm: The algorithm used for generating the results, sometimes also refered to as 'model_name'
    :param eco_region: The eco_region to which the results apply
    :return: The correct file path of the results.
    '''
    file_dir = get_results_dir(algorithm)
    file_name = get_file_name(algorithm, eco_region, 'pkl')
    return file_dir + file_name

def write_model(model, model_name, start_year, eco_region):
    '''
    Function used to save a model in the correct directory with an identifiable name. Uses Pickle for saving the model object
    :param model: The model which is to be saved
    :param model_name: The used ML-algorithm
    :param start_year: The date at which the training data starts
    :param eco_region: The name of the ecoregion to which the model applies
    :return: None
    '''

    file_dir = get_model_dir(model_name, start_year)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    file_name = get_file_name(model_name, eco_region)
    file = file_dir + file_name
    with open(file, "wb") as f:
        pkl.dump(model, f, protocol=5)

def write_results(data, algorithm, ecoregion):
    '''
    Used for writing results of the sub model into a pickled file
    :param data: The data that is to be pickled
    :param file_path: The location at which the data should be stored
    :return: None
    '''
    results_dir = get_results_dir(algorithm)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    file_path = get_results_path(algorithm, ecoregion)
    print(f'writing intermediate results for region {ecoregion}')
    data.to_pickle(file_path, protocol=5)

def write_sf(data, algorithm, ecoregion):
    '''
    Used for writing results of the sub model into a pickled file
    :param data: The data that is to be pickled
    :param file_path: The location at which the data should be stored
    :return: None
    '''
    sf_dir = get_sf_dir(algorithm)
    if not os.path.isdir(sf_dir):
        os.makedirs(sf_dir)
    file_path = get_sf_path(algorithm, ecoregion)
    print(f'writing intermediate results for region {ecoregion}')
    data.to_netcdf(file_path)

def read_SARIMA_model(file_path, train_dat):
    '''
    Function for unpickling trained models
    :param file_path: location of the pickle file
    :return: The unpickled model
    '''
    with open(file_path, 'rb') as f:
        model = pkl.load(f)

    # In earlier version of the model, the saved objects contained too much redundant data. Some remnants may remain of these large pickled objects.
    if isinstance(model, sm_tsa.statespace.sarimax.SARIMAXResultsWrapper):
        trained_model = model
        with open(file_path, "wb") as f:
            pkl.dump(model.params, f, protocol=5)
    elif isinstance(model, np.ndarray): # The newly pickled objects should only contain a single numpy array with the parameters for each term in the SARIMA model
        target_dat = train_dat.sf_per_eco.values
        trained_model = sm.tsa.SARIMAX(target_dat, **SARIMA_params)
        trained_model = trained_model.filter(model)
    else:
        raise NotImplementedError(f'Unkown file type: {type(model)} encountered when loading model')
    print(f"finished loading model {file_path}")
    return trained_model

def train_model(train_dat, model_name, eco_region, display = 0):
    '''
    Function for training a model on the provided training data
    :param train_dat: The data used for training. Includes both target data and predictor data
    :param model_name: Name of the ML-algorithm to be used for training
    :return: A trianed model
    '''

    start_year = str(train_dat.time.dt.year.min().values)

    print(f"starting process for {model_name}, at ecoregion {eco_region} using data starting at {start_year}")

    if model_name == 'SARIMA':
        target_data = train_dat.sf_per_eco
        model = sm.tsa.statespace.SARIMAX(target_data.values,
                                         **SARIMA_params
                                          )
        fitted_model=model.fit(maxiter=100, disp=display) # method='cg'
    else:
        raise NotImplementedError(f'Training of model {model_name} has not been implemented')
#     eco_region = str(train_dat.eco_regions.values)

    # Save model for future usage
    write_model(fitted_model, model_name, start_year, eco_region)
    return fitted_model

def test_eco_region(eco_dat, model_name):
    region, data = eco_dat
    results_df = pd.DataFrame()
    sf_data = [None] * 17

    #Set aside the testing data. Using the classical 80%-20% split
    test_ds = data.loc[dict(time=slice("2017", "2020"))]
    region_dat = data.loc[dict(time=slice("2000", "2016"))]
    for year in range(2000, 2017):
        # Load the model if it has been trained already
        file_name = get_model_path(model_name, str(year), region)

        # Determine training data and train model
        train_ds = region_dat.loc[dict(time=slice(str(year), "2016"))]


        try:
            trained_model = read_SARIMA_model(file_name, train_ds)
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
        print(f'Generating perforamce on training set - region: {region}, year: {year}')
        train_results = eval_model(trained_model, train_ds, model_name, 'train', show_fit=False)
        print(f'Generating perforamce on test set - region: {region}, year: {year}')
        test_results = eval_model(trained_model, test_ds, model_name, 'test', show_fit=False)
        model_params = {
                'eco_region':region,
                'start_year':year,
                'N_train_years':(2017-year),
                'N_train_obs':len(train_ds.time),
                'N_test_years':4,
                'N_test_obs':len(test_ds.time)
        }

        sf_data[model_params['N_train_years']-1] = create_sf_dataset(trained_model, xr.concat([train_ds, test_ds], 'time'), region)

        # unpack all dicts to form single results dict
        model_results = pd.DataFrame({**model_params, **train_results, **test_results}, index=[region])
        if len(results_df) != 0:
            results_df = pd.concat([results_df, model_results])
        else:
            results_df = model_results
    write_results(results_df, model_name, region)
    sf_ds = xr.concat(sf_data, 'n_train_years', data_vars='minimal', compat='no_conflicts')
    write_sf(sf_ds, model_name, region)
    return results_df, sf_ds

def create_sf_dataset(model, data, eco_region):

    start_year = pd.DatetimeIndex(data.time).year.min()
    n_train_years = 2017-start_year
    test_data = data.loc[dict(time=slice("2017", "2020"))]

    final_model = model.append(test_data.sf_per_eco.values)

    # Determine the predicted scaling factor
    prediction = final_model.get_prediction(start=0)
    pred_sf = prediction.predicted_mean
    pred_sf = xr.DataArray(
        data=[pred_sf],
        dims=["n_train_years", "time"],
        coords=dict(
            time=data.time,
            n_train_years = [n_train_years],
        ),
        attrs=dict(
            Description="Predicted scaling factor",
            Units="-",
            # model=model_name,
        ),
        name='predicted_sf'
    )

    # Store the prior flux
    prior_flux = data.prior_flux_per_s
    prior_flux.attrs['Description'] = 'Total prior flux for each eco-region'
    prior_flux.attrs['Units'] = 'mol s^-1'

    # Determine the predicted flux
    pred_flux = prior_flux * pred_sf
    pred_flux = pred_flux.rename('predicted_flux').transpose()
    pred_flux.attrs['Description'] = 'Total predicted flux eco_region'
    pred_flux.attrs['Units'] = 'mol s^-1'

    # Store the optimized scaling factor
    opt_sf = data.sf_per_eco
    opt_sf.attrs['Description'] = 'The Effective optimized scaling factor of each eco-region'
    opt_sf.attrs['Units'] = '-'

    # Determine the optimised flux
    opt_flux = data.opt_flux
    opt_flux = opt_flux.rename('optimized_flux')
    opt_flux.attrs['Description'] = 'Total optimized flux of each eco-region'
    opt_flux.attrs['Units'] = 'mol s^-1'

    # Store the time points used for training
    training_time = data.time.where((data.time < test_data.time.values[0]))
    training_time = training_time.rename('training_time')
    training_time = training_time.expand_dims({'n_train_years':[n_train_years]})
    training_time.attrs['Description'] = 'List of time points used for training the model for each number of training years'

    # Store the time points used for training
    testing_time = data.time.where((data.time >= test_data.time.values[0]))
    testing_time = testing_time.rename('testing_time')
    testing_time.attrs['Description'] = 'List of time points used for testing the model'

    # Store the surface area of the complete eco_region
    surface_area = data.eco_area.min()

    # Determine TransCom region
    tc_region = int((eco_region-1)/19+1)
    transCom = xr.DataArray(
        data=tc_region,
        coords=dict(
            eco_regions=data.coords['eco_regions'].values,
        ),
        attrs=dict(
            description="TransCom region of the eco region",
            full_names=dumps(transCom_dict),
        ),
        name='tc_region'
    )
    return xr.merge([prior_flux, pred_sf, pred_flux, opt_sf, opt_flux, training_time, testing_time, surface_area, transCom])

def run_model(model_name, complete_ds):
    # the models will be evaluated per ecoregion. Hence, the original dataset is split into a separate one for each ecoregion
    eco_region_dat = list(complete_ds.groupby("eco_regions"))

    # Preload all data to prevent loading error during multithreading process
    eco_region_dat = [(region, data.load(scheduler='sync')) for region, data in eco_region_dat]

    if MACHINE == 'local': # reduce number of ecoregions in order to maintain speed within debugging process
        eco_region_dat = eco_region_dat[:2]

    with Pool(CPU_COUNT) as pool:
        list_of_results = pool.starmap(test_eco_region, zip(eco_region_dat, repeat(model_name)))

    res_list, sf_list = map(list, zip(*list_of_results))

    sf_ds = xr.concat(sf_list, 'eco_regions', data_vars='minimal')
    corrected_train_time = sf_ds.training_time.isel(dict(eco_regions=1)).squeeze()
    corrected_test_time = sf_ds.testing_time.isel(dict(eco_regions=1)).squeeze()
    sf_ds.update({'training_time':corrected_train_time, 'testing_time':corrected_test_time})

    results_df = pd.concat(res_list)

    return results_df, sf_ds



import timeit

# Loading all necessary data
with xr.open_dataset(pred_var_path + 'vars_per_eco_update.nc') as ds:
    complete_ds = ds
start_time = timeit.default_timer()
res_df, sf_ds = run_model('SARIMA', complete_ds)
stop_time = timeit.default_timer()

print(f'Elapsed time: {stop_time-start_time}')

results_file = RESULTS_DIR + 'SARIMA_results.pkl'
print(res_df)
res_df.to_pickle(results_file)

sf_file = SF_DIR + 'SARIMA_sf.nc'
print(sf_ds)
sf_ds.to_netcdf(sf_file)
# Used in order to gain more insights on how long it takes for Snellius to finish the script
import timeit

# Loading all necessary data
with xr.open_dataset(pred_var_path + 'vars_per_eco_update.nc') as ds:
    complete_ds = ds
start_time = timeit.default_timer()
res_df, sf_ds = run_model('SARIMA', complete_ds)
stop_time = timeit.default_timer()

print(f'Elapsed time: {stop_time-start_time}')

results_file = RESULTS_DIR + 'SARIMA_results.pkl'
print(res_df)
res_df.to_pickle(results_file)

sf_file = SF_DIR + 'SARIMA_sf.nc'
print(sf_ds)
sf_ds.to_netcdf(sf_file)