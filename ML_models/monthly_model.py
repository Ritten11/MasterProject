import pickle as pkl
import numpy as np
import pandas as pd
import xarray as xr
import os


import statsmodels.api as sm  # Used for both SARIMA and SARIMAX models
import statsmodels.tsa as sm_tsa  # Used for type checking SARIMA models
#
# import abc
# from abc import ABC, abstractmethod

from ML_models.base_model import ML_model


class monthly_model(ML_model):
    MODEL_NAME = 'monthly'

    def __init__(self, machine = 'local', show_fit = False):
        """
        Initiation of simple monthly model
        :param machine: Machine upon which the code is run. Can either be 'local' or 'Snellius'
        """
        super(monthly_model, self).__init__(machine, {}, show_fit = show_fit)

    def read_model(self, file_path, train_dat):
        """
        Method for reading previously trained models from a save-file
        :param file_path: Path to where the file can be found
        :param train_dat: Data used to train the model. Is needed for some algorithms to initialize the model
        :return: The trained model
        """
        with xr.open_dataset(file_path) as ds:
            trained_model = ds
        print(f"finished loading model {file_path}")
        return trained_model

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
        file_name = self.get_file_name(eco_region, 'nc')
        file = file_dir + file_name
        model.to_netcdf(file)


    def train_model(self, train_dat):
        """
        If no previously trained model is available, or if the available save file is corrupted, a new model needs to be
        trained.
        :param train_dat: Data needed for training the model
        :return: A trained model
        """
        start_year = str(train_dat.time.dt.year.min().values)
        eco_region = float(train_dat.eco_regions.values)

        grouped_data = train_dat.sf_per_eco.groupby('time.month')

        print(
            f"starting training process of monthly model, at eco-region {eco_region} using data starting at {start_year}")
        mean = grouped_data.mean()
        mean = mean.rename('mean')
        std = grouped_data.std()
        upper_lim = mean + std
        upper_lim = upper_lim.rename('upper limit')

        lower_lim = mean - std
        lower_lim = lower_lim.rename('lower limit')

        ci = np.array([lower_lim.values, upper_lim.values]).T
        # ci = xr.concat([mean-std, mean+std], pd.Index(['lower', 'upper'], name='new_dim'))
        # ci = ci.rename('ci')
        n = grouped_data.count().astype('int32')
        n = n.rename('N')
        fitted_model = xr.merge([mean, lower_lim, upper_lim, n])
        fitted_model.attrs['confidence interval'] = 'The upper and lower limit have been set to be 1 std from the mean'
        # fitted_model.attrs['N'] = str(N)

        self.write_model(fitted_model, start_year, eco_region)
        return fitted_model

    def get_prediction(self, model, data, test_or_train):
        """
        Method used for extracting the prediction of a model
        :param model: The trained model which is to make the prediction
        :param data: Data needed for making the prediction. Should be an XArray.DataSet containing both target
        and predictor variables
        :param test_or_train: The functionality of the method is slightly changed depending on whether the prediction is
         on testing or training data
        :return: timeseries of predicted SFs
        """
        predictions = {date: model.sel(month=(date.astype('datetime64[M]').astype(int) % 12 + 1)) for date in
                         data.time.values}
        predictions = xr.concat(list(predictions.values()), pd.Index(list(predictions.keys()), name="time"))
        predictions.attrs['Opt_method'] = 'Monthly average scaling factors of period'
        pred_dat = predictions['mean'].values

        pred_ci = np.stack([predictions['lower limit'].values, predictions['upper limit'].values]).T

        return pred_dat, pred_ci
