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

# No model needs to be trained. The sole purpose of this class is to allow for easy evaluation of the prior flux model


class prior_model(ML_model):
    MODEL_NAME = 'prior'

    def __init__(self, machine = 'local', show_fit = False):
        """
        Initiation of simple monthly model
        :param machine: Machine upon which the code is run. Can either be 'local' or 'Snellius'
        """
        super(prior_model, self).__init__(machine, {}, show_fit = show_fit)

    def read_model(self, file_path, train_dat):
        """
        Method for reading previously trained models from a save-file
        :param file_path: Path to where the file can be found
        :param train_dat: Data used to train the model. Is needed for some algorithms to initialize the model
        :return: The trained model
        """
        return None

    def write_model(self, model, start_year, eco_region):
        """
        Function used to save a model in the correct directory with an identifiable name. Uses Pickle for saving
        the model object
        :param model: The model which is to be saved
        :param start_year: The date at which the training data starts
        :param eco_region: The name of the eco_region to which the model applies
        :return: None
        """

    def train_model(self, train_dat):
        """
        If no previously trained model is available, or if the available save file is corrupted, a new model needs to be
        trained.
        :param train_dat: Data needed for training the model
        :return: A trained model
        """
        return None

    def get_prediction(self, model, data, test_or_train):
        """
        All the 'get_prediction' function needs to do, is return an array of ones of the correct length. This will
        result in no changes being made to the prior model and hence an evaluation of the prior model
        :param model: The trained model which is to make the prediction
        :param data: Data needed for making the prediction. Should be an XArray.DataSet containing both target
        and predictor variables
        :param test_or_train: The functionality of the method is slightly changed depending on whether the prediction is
         on testing or training data
        :return: timeseries of predicted SFs
        """
        pred_dat = np.ones(len(data.time.values))

        pred_ci = np.stack([pred_dat, pred_dat]).T

        return pred_dat, pred_ci
