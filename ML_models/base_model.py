import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import pickle as pkl
import xarray as xr

import os

import statsmodels.api as sm  # Used for both SARIMA and SARIMAX models
import statsmodels.tsa as sm_tsa  # Used for type checking SARIMA models
from sklearn import metrics     # Used for importing various performance measures

from json import dumps # needed for making a string out of a dictionary
from itertools import repeat # Needed for repeating a variable multiple times

import abc
from abc import ABC, abstractmethod

# Due to the way the scaling factors are determined, factors associated with a flux near 0 can get massive scaling
# factors. See Thesis section (??) for more in depth explanation
def cap_outliers(dat, scalar = 4):
    mean = np.mean(dat.values.flatten())
    sd = np.std(dat.values.flatten())

    dat = dat.where(dat < (mean+sd*scalar), (mean+sd*scalar))
    dat = dat.where(dat > (mean-sd*scalar), (mean-sd*scalar))
    return dat

def to_tc_region(sf_data):
    """
    Aggregate the dataset from eco_region level to transCom level
    :param sf_data: Xarray dataset with ecoregions
    :return: Xarray dataset with transCom regions
    """
    tc_flux_data = sf_data[['optimized_flux', 'predicted_flux', 'prior_flux_per_s',
                            'eco_area', 'testing_time', 'training_time', 'tc_region']].groupby('tc_region').sum()

    # Determine the optimized scaling factor of the transCom region
    opt_sf = tc_flux_data.optimized_flux / tc_flux_data.prior_flux_per_s
    tc_flux_data['sf_per_tc'] = cap_outliers(opt_sf)  # Deal with exploding scaling factors
    pred_sf_list = [None] * len(sf_data.n_train_years.values)  # Initialize a list with correct number of entries

    # Aggregate the ecoregions separately for each training data split
    for n_years in tc_flux_data.n_train_years.values:
        year_data = tc_flux_data.loc[dict(n_train_years = n_years)]  # select the right training data split
        year_data_train = year_data.where(year_data.training_time.notnull(), drop=True)
        year_data_test = year_data.where(year_data.testing_time.notnull(), drop=True)
        year_data_combined = xr.merge([year_data_train, year_data_test])  # combine the training and testing data

        # calculate the effective scaling factor
        eff_sf = year_data_combined.predicted_flux / year_data_combined.prior_flux_per_s
        pred_sf_list[n_years-1] = cap_outliers(eff_sf)

    tc_flux_data['predicted_sf'] = xr.concat(pred_sf_list, 'n_train_years').transpose('tc_region', 'n_train_years', 'time')
    return tc_flux_data

def eval_model(sf_data, test_or_train, target_var):
    """
        Evaluate the model using the provided testing data
        :param sf_data: XArray dataset containing predicted scaling factors, analysed scaling factors and
        flux data
        :param test_or_train: Flag to indicate whether the passed data is training data ot testing data
        :return: A dictionary with the results according to various performance measures.
        """

    # print(sf_data)
    pred_dat = sf_data.predicted_sf.values
    flux_dat = sf_data.prior_flux_per_s.values
    true_dat = sf_data[target_var].values

    # Make sure all provided datasets heve the same length
    assert (len(true_dat) == len(pred_dat)) and (len(true_dat) == len(flux_dat)), \
        'Passed datasets are do not have the same length: '

    # Determine the performance in scaling factor space
    sf_ME = (np.sum(true_dat) - np.sum(pred_dat)) / len(true_dat)
    sf_MAE = metrics.mean_absolute_error(true_dat, pred_dat)
    sf_MAPE = metrics.mean_absolute_percentage_error(true_dat, pred_dat) * 100
    sf_RMSE = np.sqrt(metrics.mean_squared_error(true_dat, pred_dat))
    sf_r2 = metrics.r2_score(true_dat, pred_dat)

    # Move evaluation to flux space
    true_flux = true_dat * flux_dat
    pred_flux = pred_dat * flux_dat

    # Determine the performance in flux space
    flux_ME = (np.sum(true_flux) - np.sum(pred_flux)) / len(true_flux)
    flux_MAE = metrics.mean_absolute_error(true_flux, pred_flux)
    flux_MAPE = metrics.mean_absolute_percentage_error(true_flux, pred_flux) * 100
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
    region = str(target_dat.eco_regions.values)
    # Plot data points
    target_dat.plot.scatter(x='time', y='sf_per_eco', ax=ax, label='Observed', c='C00')
    h, l = ax.get_legend_handles_labels()
    # Plot predictions
    line = ax.plot(target_dat.time.values, pred_dat, c='C01')
    ci = pred_ci
    ax.fill_between(target_dat.time.values, ci[:, 0], ci[:, 1], color='C01', alpha=0.1)
    fill = ax.fill(np.NaN, np.NaN, color='C01', alpha=0.1)

    plt.title(f'Performance on {test_or_train} data of eco-region {region}')

    legend = ax.legend([(line[0], fill[0]), h[0]], ['Forecast with uncertainty', l[0]], loc='lower right')

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
            # For retrieving the set of aggregated scaling vectors
            self.PRED_VAR_PATH = '/gpfs/work1/0/ctdas/awoude/Ritten/predictor_vars/'

            # used for storing the trained model
            self.SAVE_DIR = self.pers_file_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/trained_models/'

            # used for storing the scaling factor produced bij ML models
            self.SF_DIR = '/gpfs/work1/0/ctdas/awoude/Ritten/fitted_sf/'

            # used for storing the trained model
            self.RESULTS_DIR = '/gpfs/work1/0/ctdas/awoude/Ritten/results/'

            # Snellius allows for usage of ut to 32 threads on the same node without additional costs
            self.CPU_COUNT = 32

        elif machine == 'local':
            self.PRED_VAR_PATH = './'  # For retrieving the set of aggregated scaling vectors

            self.SAVE_DIR = self.pers_file_dir = './trained_models_download/' #'./models/'  # used for storing the trained model

            self.SF_DIR = './fitted_sf/'  # used for storing the scaling factor produced bij ML models

            self.RESULTS_DIR = './results/'

            # The -2 is placed in order to maintain a relatively fast PC when running the model
            self.CPU_COUNT = int(multiprocessing.cpu_count() / 2)

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
            :param data: Data needed for making the prediction. Should be an XArray.DataSet containing both
            target and predictor variables
            :param test_or_train: Boolean value indicating whether the prediction should be on the training data
            or the testing data
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

        print(f'Determining sf prediction of region {region}')
        try:
            sf_ds = self.load_sf_data(region)
        except (EOFError, FileNotFoundError):
            print(f'No sf data exists on region {region}. Making new sf prediction')
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
                    # print(f'file "{file_name}" already contains a trained model. Skipping training process')
                except pkl.UnpicklingError:
                    print(f'Could no unpickle model from eco-region {region} and starting year {year}. Training new model')
                    # Pickled model somehow got corrupted. Train a new model
                    trained_model = self.train_model(train_ds)
                except (EOFError, FileNotFoundError):
                    print(f'No file exists, or the existing file is empty. Training new model')
                    # If no model exists, train a new one
                    trained_model = self.train_model(train_ds)
                except ValueError as ve:
                    print(ve)
                    print(f'The loaded file contained an incompatible model. Training a new model')
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

    def analyse_sf_data(self, sf_data, per_tc_region = False):
        if per_tc_region:
            region = float(sf_data.tc_region.values)
        else:
            region = float(sf_data.eco_regions.values)

        grouped_sf_data = sf_data.groupby('n_train_years')
        results_df = pd.DataFrame()
        for n_train_years, sf_data_per_year in grouped_sf_data:

            # Extract the first year used for testing
            test_data = sf_data_per_year.where(sf_data_per_year.testing_time.notnull(), drop=True)
            first_testing_year = pd.DatetimeIndex(test_data.time).year.min()

            # Determine training data and train model
            train_data = sf_data_per_year.where(sf_data_per_year.training_time.notnull(), drop=True)

            model_params = {
                'start_year': first_testing_year-n_train_years,
                'N_train_years': n_train_years,
                'N_train_obs': len(train_data.time),
                'N_test_years': 4,
                'N_test_obs': len(test_data.time)
            }

            if per_tc_region:
                model_params['tc_region'] = region
                target_var = 'sf_per_tc'
            else:
                model_params['eco_region'] = region
                target_var = 'sf_per_eco'

            # Evaluate the model, both on training and testing data
            print(f'Generating performance on training set - region: {region}, n_train_years: {n_train_years}')
            train_results = eval_model(train_data, 'train', target_var=target_var)

            print(f'Generating performance on test set - region: {region}, n_train_years: {n_train_years}')
            test_results = eval_model(test_data, 'test', target_var=target_var)

            # unpack all dicts to form single results dict
            model_results = pd.DataFrame({**model_params, **train_results, **test_results}, index=[region])
            if len(results_df) != 0:
                results_df = pd.concat([results_df, model_results])
            else:
                results_df = model_results

        self.write_results(results_df, region, per_tc_region)
        return results_df

    def run_model(self, data_file, debug=False):

        # Loading all necessary data
        with xr.open_dataset(self.PRED_VAR_PATH + data_file) as ds:
            complete_ds = ds

        # the models will be evaluated per eco-region. Hence, the original dataset is split into a separate dataset
        # for each eco-region
        eco_region_dat = list(complete_ds.groupby("eco_regions"))

        # Preload all data to prevent loading error during multithreading process
        eco_region_dat = [data.load(scheduler='sync') for _, data in eco_region_dat]

        if debug:
            eco_region_dat = eco_region_dat[:6]
        # if self.MACHINE == 'local':  # reduce number of eco-regions in order to maintain speed within debugging process
        #     eco_region_dat = eco_region_dat

        with Pool(self.CPU_COUNT) as pool:
            results = pool.map_async(self.pred_eco_region, eco_region_dat)
            try:
                sf_list = results.get(timeout=2400)
            except TimeoutError as e:
                print(f'Session stopped due to timeout: {e}')


        sf_ds = xr.concat(sf_list, 'eco_regions', data_vars='minimal')
        corrected_train_time = sf_ds.training_time.isel(dict(eco_regions=1)).squeeze()
        corrected_test_time = sf_ds.testing_time.isel(dict(eco_regions=1)).squeeze()
        sf_ds.update({'training_time': corrected_train_time, 'testing_time': corrected_test_time})
        self.sf_data = sf_ds

        # Saving the scaling factor results
        sf_file = self.SF_DIR + self.get_model_name() + '_sf.nc'
        sf_ds.to_netcdf(sf_file)

        return sf_ds

    def test_model(self, sf_data, per_tc_region = False):
        """
        Very similar to the run_model method, but instead of predicting the scaling factors, this method analyses
        the result
        :param sf_data: The data which is to be analysed. The complete NetCDF4 file, with coordinates
        [eco_regions, n_train_years, time]
        :return: The performance of the predicted scaling factors using various measures
        """
        if sf_data is None:
            sf_data = self.sf_data

        if per_tc_region:
            sf_data = to_tc_region(sf_data)
            region_sf_dat = list(sf_data.groupby("tc_region"))
        else:
            region_sf_dat = list(sf_data.groupby("eco_regions"))

        region_sf_dat = [data.load(scheduler='sync') for _, data in region_sf_dat]

        with Pool(self.CPU_COUNT) as pool:
            res_list = pool.starmap(self.analyse_sf_data, zip(region_sf_dat, repeat(per_tc_region)))

        # Reformatting the results object
        results_df = pd.concat(res_list)
        self.results = results_df

        # Saving the results file
        if per_tc_region:
            results_file = self.RESULTS_DIR + self.get_model_name() + '_results_per_tc.pkl'
        else:
            results_file = self.RESULTS_DIR + self.get_model_name() + '_results.pkl'

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

    def get_file_name(self, eco_region, extension, per_tc_region=False):
        """
        Function used for generating the file name of a model
        :param eco_region: The eco_region to which the model applies
        :param extension: The file extension given to the file name
        :param per_tc_region: Flag on whether the file is based on a tc_region
        :return: file name of the (to be) saved model or results file
        """
        if per_tc_region:
            file_name = 'tc_' + self.get_model_name() + '_' + str(eco_region) + '.' + extension
        else:
            file_name = self.get_model_name() + '_' + str(eco_region) + '.' + extension
        return file_name

    def get_model_path(self, start_year, eco_region, save_type='pkl'):
        """
        Function for automatically generating the location of a saved model
        :param start_year: The year at which the training data started
        :param eco_region: The eco_region to which the model applies
        :return: The correct file path of the model.
        """
        file_dir = self.get_model_dir(start_year)
        file_name = self.get_file_name(eco_region, save_type)
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

    def get_results_path(self, eco_region, per_tc_region = False):
        """
        Function for automatically generating the full path to the analysed results
        :param eco_region: The eco_region to which the results apply
        :param per_tc_region: Indication on whether the results are based on TransCom regions
        :return: The correct file path of the results.
        """
        file_dir = self.get_results_dir()
        file_name = self.get_file_name(eco_region, 'pkl', per_tc_region)
        return file_dir + file_name

    def write_model(self, model, start_year, eco_region):
        """
        Function used to save a model in the correct directory with an identifiable name. Uses Pickle for saving
        the model object
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

    def write_results(self, data, region, per_tc_region):
        """
        Used for writing results of the sub model into a pickled file
        :param data: The data that is to be pickled
        :param region: Region of the results which are to be stored. Can either be an eco-region or a TransCom region
        :return: None
        """
        results_dir = self.get_results_dir()
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        file_path = self.get_results_path(region, per_tc_region)
        print(f'writing intermediate results for region {region}')
        data.to_pickle(file_path, protocol=5)
        # print(f'finished writing results region')

    def write_sf(self, data, eco_region):
        """
        Used for writing results of the sub model into a pickled file
        :param data: The data that is to be pickled
        :param eco_region: Eco-region of the results which are to be stored
        :return: None
        """
        sf_dir = self.get_sf_dir()
        if not os.path.isdir(sf_dir):
            os.makedirs(sf_dir)
        file_path = self.get_sf_path(eco_region)
        print(f'writing intermediate results for region {eco_region}')
        data.to_netcdf(file_path)
        print(f'Finished writing results for region {eco_region}')

    def load_sf_data(self, region):
        sf_file = self.get_sf_path(region)

        with xr.open_dataset(sf_file) as loaded_ds:
            sf_ds = loaded_ds
        return sf_ds


