import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import pickle as pkl
import xarray as xr

import os

import statsmodels.api as sm  # Used for bot he SARIMA and SARIMAX models
import statsmodels.tsa as sm_tsa  # Used for type checking SARIMA models
from sklearn import metrics     # Used for importing various performance measures

from json import dumps # needed for making a string out of a dictionary
from itertools import repeat # Needed for repeating a variable multiple times


from abc import ABC, abstractmethod


def eval_model(sf_data, test_or_train):
    """
        Evaluate the model using the provided testing data
        :param sf_data: XArray dataset containing predicted scaling factors, analysed scaling factors and
        flux data
        :param test_or_train: Flag to indicate whether the passed data is training data ot testing data
        :return: A dictionary with the results according to various performance measures.
        """

    print(sf_data)
    pred_dat = sf_data.predicted_sf.values
    flux_dat = sf_data.prior_flux_per_s.values
    true_dat = sf_data.sf_per_eco.values
    #
    # print(len(pred_dat))
    # print(len(flux_dat))
    # print(len(true_dat))
    # Make sure all provided datasets heve the same length
    assert (len(true_dat) == len(pred_dat)) and (len(true_dat) == len(flux_dat)), \
        'Passed datasets are do not have the same length: '

    # Determine the performance in scaling factor space
    sf_ME = (np.sum(true_dat) - np.sum(pred_dat)) / len(true_dat)
    sf_MAE = metrics.mean_absolute_error(true_dat, pred_dat)
    sf_MAPE = metrics.mean_absolute_percentage_error(true_dat, pred_dat)
    sf_RMSE = np.sqrt(metrics.mean_squared_error(true_dat, pred_dat))
    sf_r2 = metrics.r2_score(true_dat, pred_dat)

    # Move evaluation to flux space
    true_flux = true_dat * flux_dat
    pred_flux = pred_dat * flux_dat

    # Determine the performance in flux space
    flux_ME = (np.sum(true_dat) - np.sum(pred_flux)) / len(true_dat)
    flux_MAE = metrics.mean_absolute_error(true_flux, pred_flux)
    flux_MAPE = metrics.mean_absolute_percentage_error(true_flux, pred_flux)
    flux_RMSE = np.sqrt(metrics.mean_squared_error(true_flux, pred_flux))
    flux_r2 = metrics.r2_score(true_flux, pred_flux)
    return {'sf_ME_' + test_or_train: sf_ME,
            'sf_MAE_' + test_or_train: sf_MAE,
            'sf_MAPE_' + test_or_train: sf_MAPE,
            'sf_RMSE_' + test_or_train: sf_RMSE,
            'sf_r2_' + test_or_train: sf_r2,
            'flux_ME_' + test_or_train: flux_ME,
            'flux_MAE_' + test_or_train: flux_MAE,
            'flux_MAPE_' + test_or_train: flux_MAPE,
            'flux_RMSE_' + test_or_train: flux_RMSE,
            'flux_r2_' + test_or_train: flux_r2}


def plot_fit(target_dat, pred_dat, pred_ci, test_or_train):
    # Graph
    fig, ax = plt.subplots(figsize=(9, 4))
    title = test_or_train + ' data: predicted sf of eco_region ' + str(target_dat.eco_regions.values)
    ax.set(title=title, xlabel='Date', ylabel='Scaling factor')

    # Plot data points
    target_dat.plot.scatter(x='time', y='sf_per_eco', ax=ax, label='Observed', c='C00')
    # Plot predictions
    plt.plot(target_dat.time.values, pred_dat, label='One-step-ahead forecast', c='C01')
    ci = pred_ci
    ax.fill_between(target_dat.time.values, ci[:, 0], ci[:, 1], color='C01', alpha=0.1)

    legend = ax.legend(loc='lower right')

    plt.show()


class ML_model(ABC):

    transcom_dict = {1: 'North American Boreal',
                     2: 'North American Temperate',
                     7: 'Eurasia Boreal',
                     8: 'Eurasia Temperate',
                     11: 'Europe',
                     }

    def __init__(self, machine, model_params, show_fit = False):
        self.MACHINE = machine
        if machine == 'Snellius':
            self.PRED_VAR_PATH = '/gpfs/work1/0/ctdas/awoude/Ritten/predictor_vars/'  # For retrieving the set of aggregated scaling vectors

            self.SAVE_DIR = self.pers_file_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/trained_models/'  # used for storing the trained model

            self.SF_DIR = '/gpfs/work1/0/ctdas/awoude/Ritten/fitted_sf/'  # used for storing the scaling factor produced bij ML models

            self.RESULTS_DIR = '/gpfs/work1/0/ctdas/awoude/Ritten/results/'  # used for storing the trained model

            self.CPU_COUNT = 32  # Snellius allows for usage of ut to 32 threads on the same node without additional costs

        elif machine == 'local':
            self.PRED_VAR_PATH = './'  # For retrieving the set of aggregated scaling vectors

            self.SAVE_DIR = self.pers_file_dir = './models/'  # used for storing the trained model

            self.SF_DIR = './fitted_sf/'  # used for storing the scaling factor produced bij ML models

            self.RESULTS_DIR = './results/'

            self.CPU_COUNT = multiprocessing.cpu_count() - 2  # The -2 is placed in order to maintain a relatively fast PC when running the model

        else:
            raise NotImplementedError(f'machine "{machine}" has not been implemented')

        self.MODEL_PARAMS = model_params

        self.show_fit = show_fit

        self.sf_data = None # Will be initialized after running the model
        self.results = None # Will be initialized after running the model

    @property
    @abstractmethod
    def MODEL_NAME(self):
        """
                    Every implementation of this class needs receive a name
                    :return:
                    """
        raise NotImplementedError

    def get_model_name(self):
        """
                    Helper function for retrieving class attribute MODEL_NAME
                    :return:
                    """
        return str(type(self).MODEL_NAME)

    @abstractmethod
    def read_model(self, file_path, train_dat):
        """
                    Function for unpickling trained models
                    :param file_path: location of the pickle file
                    :param train_dat: Some models need to be initialized with data upon which it was trained
                    :return: The unpickled model
                    """
        raise NotImplementedError

    @abstractmethod
    def train_model(self, data):
        """
            Methods which should handle the training of a model.
            :param data: Data needed for training the model
            :return: A trained model
            """
        raise NotImplementedError

    @abstractmethod
    def get_prediction(self, model, data, test_or_train):
        """
            Method used for extracting the prediction of a model
            :param model: The trained model which is to make the prediction
            :param data: Data needed for making the prediction. Should be an XArray.DataSet containing both target and predictor variables
            :param test_or_train: Boolean value indicating whether the prediction should be on the training data or the testing data
            :return: timeseries of predicted SFs
            """
        raise NotImplementedError

    def create_sf_dataset(self, prediction, data, eco_region):

        start_year = pd.DatetimeIndex(data.time).year.min()
        n_train_years = 2017 - start_year
        test_data = data.loc[dict(time=slice("2017", "2020"))]

        # Store the predicted scaling factor
        pred_sf = xr.DataArray(
            data=[prediction],
            dims=["n_train_years", "time"],
            coords=dict(
                time=data.time,
                n_train_years=[n_train_years],
            ),
            attrs=dict(
                Description="Predicted scaling factor",
                Units="-",
                model=self.get_model_name(),
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
        training_time = training_time.expand_dims({'n_train_years': [n_train_years]})
        training_time.attrs[
            'Description'] = 'List of time points used for training the model for each number of training years'

        # Store the time points used for training
        testing_time = data.time.where((data.time >= test_data.time.values[0]))
        testing_time = testing_time.rename('testing_time')
        testing_time.attrs['Description'] = 'List of time points used for testing the model'

        # Store the surface area of the complete eco_region
        surface_area = data.eco_area.min()

        # Determine TransCom region
        tc_region = int((eco_region - 1) / 19 + 1)
        transcom = xr.DataArray(
            data=tc_region,
            coords=dict(
                eco_regions=data.coords['eco_regions'].values,
            ),
            attrs=dict(
                description="TransCom region of the eco region",
                full_names=dumps(ML_model.transcom_dict),
            ),
            name='tc_region'
        )
        return xr.merge(
            [prior_flux, pred_sf, pred_flux, opt_sf, opt_flux, training_time, testing_time, surface_area, transcom])

    def pred_eco_region(self, eco_data):
        region = float(eco_data.eco_regions.values)

        sf_data = [None] * 17

        # Set aside the testing data. Using the classical 80%-20% split
        test_ds = eco_data.loc[dict(time=slice("2017", "2020"))]
        region_dat = eco_data.loc[dict(time=slice("2000", "2016"))]
        for year in range(2000, 2017):
            # Load the model if it has been trained already
            file_name = self.get_model_path(str(year), region)

            # Determine training data and train model
            train_ds = region_dat.loc[dict(time=slice(str(year), "2016"))]

            try:
                trained_model = self.read_model(file_name, train_ds)
                print(f'file "{file_name}" has already contains a trained model. Skipping training process')
            except pkl.UnpicklingError:
                print(f'Could no unpickle model from eco-region {region} and starting year {year}. Training new model')
                # Pickled model somehow got corrupted. Train a new model
                trained_model = self.train_model(train_ds)
            except (EOFError, FileNotFoundError):
                print(f'No file exists, or the the existing file is empty. Training new model')
                # If no model exists, train a new one
                trained_model = self.train_model(train_ds)

            # Generate predictions on both training set and testing set
            train_prediction, train_ci = self.get_prediction(trained_model, train_ds, 'train')

            if self.show_fit:
                plot_fit(train_ds, train_prediction, train_ci, 'train')

            test_prediction, test_ci = self.get_prediction(trained_model, test_ds, 'test')

            if self.show_fit:
                plot_fit(test_ds, test_prediction, test_ci, 'test')



            sf_data[(2017-year) - 1] = self.create_sf_dataset(np.concatenate([train_prediction, test_prediction]),
                                                              xr.concat([train_ds, test_ds], 'time'),
                                                              region)
        sf_ds = xr.concat(sf_data, 'n_train_years', data_vars='minimal', compat='no_conflicts')
        self.write_sf(sf_ds, region)
        return sf_ds

    def analyse_sf_data(self, sf_data):
        region = float(sf_data.eco_regions.values)

        grouped_sf_data = sf_data.groupby('n_train_years')
        results_df = pd.DataFrame()
        for n_train_years, sf_data_per_year in grouped_sf_data:

            # Extract the first year used for testing
            test_data = sf_data_per_year.where(sf_data_per_year.testing_time.notnull() , drop=True)

            first_testing_year = pd.DatetimeIndex(test_data.time).year.min()

            # Determine training data and train model
            train_data = sf_data_per_year.where(sf_data_per_year.training_time.notnull(), drop=True)

            model_params = {
                'eco_region': region,
                'start_year': first_testing_year-n_train_years,
                'N_train_years': n_train_years,
                'N_train_obs': len(train_data.time),
                'N_test_years': 4,
                'N_test_obs': len(test_data.time)
            }

            # Evaluate the model, both on training and testing data
            print(f'Generating perforamce on training set - region: {region}, n_train_years: {n_train_years}')
            train_results = eval_model(train_data, 'train')

            print(f'Generating performance on test set - region: {region}, n_train_years: {n_train_years}')
            test_results = eval_model(test_data, 'test')

            # unpack all dicts to form single results dict
            model_results = pd.DataFrame({**model_params, **train_results, **test_results}, index=[region])
            if len(results_df) != 0:
                results_df = pd.concat([results_df, model_results])
            else:
                results_df = model_results

        self.write_results(results_df, region)
        return results_df

    def run_model(self, data_file):

        # Loading all necessary data
        with xr.open_dataset(self.PRED_VAR_PATH + data_file) as ds:
            complete_ds = ds

        # the models will be evaluated per ecoregion. Hence, the original dataset is split into a separate one for each ecoregion
        eco_region_dat = list(complete_ds.groupby("eco_regions"))

        # Preload all data to prevent loading error during multithreading process
        eco_region_dat = [data.load(scheduler='sync') for _, data in eco_region_dat]

        if self.MACHINE == 'local':  # reduce number of ecoregions in order to maintain speed within debugging process
            eco_region_dat = eco_region_dat[:5]

        with Pool(self.CPU_COUNT) as pool:
            sf_list = pool.map(self.pred_eco_region, eco_region_dat)

        sf_ds = xr.concat(sf_list, 'eco_regions', data_vars='minimal')
        corrected_train_time = sf_ds.training_time.isel(dict(eco_regions=1)).squeeze()
        corrected_test_time = sf_ds.testing_time.isel(dict(eco_regions=1)).squeeze()
        sf_ds.update({'training_time': corrected_train_time, 'testing_time': corrected_test_time})
        self.sf_data = sf_ds

        # Saving the scaling factor results
        sf_file = self.SF_DIR + 'SARIMA_sf.nc'
        sf_ds.to_netcdf(sf_file)

        return sf_ds

    def test_model(self, sf_data):
        '''
        Very similar to the run_model method, but instead of predicint the scaling factors, this method analyses
        the result
        :param sf_data: The data which is to be analysed. The complete NetCDF4 file, with coordinates
        [eco_regions, n_train_years, time]
        :return: The performance of the predicted scaling factors using various measures
        '''
        if sf_data is None:
            sf_data = self.sf_data

        eco_region_sf_dat = list(sf_data.groupby("eco_regions"))
        eco_region_sf_dat = [data.load(scheduler='sync') for _, data in eco_region_sf_dat]

        with Pool(self.CPU_COUNT) as pool:
            res_list = pool.map(self.analyse_sf_data, eco_region_sf_dat)

        # Reformatting the results object
        results_df = pd.concat(res_list)
        self.results = results_df

        # Saving the results file
        results_file = self.RESULTS_DIR + 'SARIMA_results.pkl'
        results_df.to_pickle(results_file)

        return results_df

    # ==================================================== v HELPER FUNCTIONS v ========================================
    def get_model_dir(self, start_year):
        """
        The path to the directory in which the model should be saved
        :param start_year: The year at which the training data started
        :return: Directory of the (to be) saved model
        """
        file_dir = self.SAVE_DIR + self.get_model_name() + '/' + start_year + '/'
        return file_dir

    def get_file_name(self, eco_region, extention):
        """
        Function used for generating the file name of a model
        :param eco_region: The eco_region to which the model applies
        :param extention: The file extention given to the file name
        :return: file name of the (to be) saved model or results file
        """
        file_name = self.get_model_name() + '_' + str(eco_region) + '.' + extention
        return file_name

    def get_model_path(self, start_year, eco_region):
        """
        Function for automatically generating the location of a saved model
        :param start_year: The year at which the training data started
        :param eco_region: The eco_region to which the model applies
        :return: The correct file path of the model.
        """
        file_dir = self.get_model_dir(start_year)
        file_name = self.get_file_name(eco_region, 'pkl')
        return file_dir + file_name

    def get_sf_dir(self):
        """
        Function for automatically generating the location of the fitted scaling factors
        :return: The directory at which the results file should be stored.
        """
        file_dir = self.SF_DIR + self.get_model_name() + '/'
        return file_dir

    def get_sf_path(self, eco_region):
        """
        Function for automatically generating the full path to the fitted scaling factors
        :param eco_region: The eco_region to which the scaling factors apply
        :return: The correct file path of the scaling factors.
        """
        file_dir = self.get_sf_dir()
        file_name = self.get_file_name(eco_region, 'nc')
        return file_dir + file_name

    def get_results_dir(self):
        """
        Function for automatically generating the location of the analysed results
        :return: The directory at which the results file should be stored.
        """
        file_dir = self.RESULTS_DIR + self.get_model_name() + '/'
        return file_dir

    def get_results_path(self, eco_region):
        '''
        Function for automatically generating the full path to the analysed results
        :param algorithm: The algorithm used for generating the results, sometimes also refered to as 'model_name'
        :param eco_region: The eco_region to which the results apply
        :return: The correct file path of the results.
        '''
        file_dir = self.get_results_dir()
        file_name = self.get_file_name(eco_region, 'pkl')
        return file_dir + file_name

    def write_model(self, model, start_year, eco_region):
        """
        Function used to save a model in the correct directory with an identifiable name. Uses Pickle for saving the model object
        :param model: The model which is to be saved
        :param start_year: The date at which the training data starts
        :param eco_region: The name of the eco_region to which the model applies
        :return: None
        """

        file_dir = self.get_model_dir(start_year)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        file_name = self.get_file_name(eco_region, 'pkl')
        file = file_dir + file_name
        with open(file, "wb") as f:
            pkl.dump(model, f, protocol=5)

    def write_results(self, data, eco_region):
        '''
        Used for writing results of the sub model into a pickled file
        :param data: The data that is to be pickled
        :param file_path: The location at which the data should be stored
        :return: None
        '''
        results_dir = self.get_results_dir()
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        file_path = self.get_results_path(eco_region)
        print(f'writing intermediate results for region {eco_region}')
        data.to_pickle(file_path, protocol=5)

    def write_sf(self, data, eco_region):
        '''
        Used for writing results of the sub model into a pickled file
        :param data: The data that is to be pickled
        :param file_path: The location at which the data should be stored
        :return: None
        '''
        sf_dir = self.get_sf_dir()
        if not os.path.isdir(sf_dir):
            os.makedirs(sf_dir)
        file_path = self.get_sf_path(eco_region)
        print(f'writing intermediate results for region {eco_region}')
        data.to_netcdf(file_path)

class SARIMA_model(ML_model):
    MODEL_NAME = 'SARIMA'

    def __init__(self, order, seasonal_order, trend, machine = 'local'):
        '''

        :param order: Defining the regular AR, I and MA dependencies
        :param seasonal_order: Defining the seasonal dependencies
        :param trend: Pass 'c' to add an intercept term
        :param machine: Machine upon which the code is run. Can either be 'local' or 'Snellius'
        '''
        super(SARIMA_model, self).__init__(machine,
                                           {'order':order,
                                            'seaonal_order':seasonal_order,
                                            'trend':trend},)

    def read_model(self, file_path, train_dat):
        '''
        Method for reading previously trained models from a save-file
        :param file_path: Path to where the file can be found
        :param train_dat: Data used to train the model. Is needed for some algorithms to initialize the model
        :return: The trained model
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
            trained_model = sm.tsa.SARIMAX(target_dat, **self.MODEL_PARAMS)
            trained_model = trained_model.filter(model)
        else:
            raise NotImplementedError(f'Unkown file type: {type(model)} encountered when loading model')
        print(f"finished loading model {file_path}")
        return trained_model

    def train_model(self, train_dat):
        '''
        If no previously trained model is available, or if the available save file is corrupted, a new model needs to be
        trained.
        :param train_dat: Data needed for training the model
        :return: A trained model
        '''
        start_year = str(train_dat.time.dt.year.min().values)
        eco_region = float(train_dat.eco_regions.values)

        print(f"starting training process of SARIMA model, at eco-region {eco_region} using data starting at {start_year}")

        target_data = train_dat.sf_per_eco
        model = sm.tsa.statespace.SARIMAX(target_data.values,
                                          **self.MODEL_PARAMS
                                          )
        fitted_model = model.fit(maxiter=100, disp=0)  # method='cg'
        #     eco_region = str(train_dat.eco_regions.values)

        # Save model for future usage
        self.write_model(fitted_model, start_year, eco_region)
        return fitted_model


    def get_prediction(self, model, data, test_or_train):
        '''
        Method used for extracting the prediction of a model
        :param model: The trained model which is to make the prediction
        :param data: Data needed for making the prediction. Should be an XArray.DataSet containing both target and predictor variables
        :param test_or_train: The functionality of the method is slightly changed depending on whether the prediction is
         on testing or training data
        :return: timeseries of predicted SFs
        '''
        if test_or_train == 'test':
            start_index = len(model.fittedvalues)
            final_model = model.append(data.sf_per_eco.values)
        elif test_or_train == 'train':
            start_index = 0
            final_model = model
        else:
            raise Exception(f'test_or_train not specified:{test_or_train}')
        prediction = final_model.get_prediction(start=start_index)
        pred_ci = prediction.conf_int()
        pred_dat = prediction.predicted_mean

        return pred_dat, pred_ci
import abc


































