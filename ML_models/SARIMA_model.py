import pickle as pkl
import numpy as np
import os


import statsmodels.api as sm  # Used for both SARIMA and SARIMAX models
import statsmodels.tsa as sm_tsa  # Used for type checking SARIMA models
# 
# import abc
# from abc import ABC, abstractmethod

from ML_models.base_model import ML_model

class SARIMA_model(ML_model):
    MODEL_NAME = 'SARIMA'

    def __init__(self, order, seasonal_order, trend, machine = 'local'):
        """

        :param order: Defining the regular AR, I and MA dependencies
        :param seasonal_order: Defining the seasonal dependencies
        :param trend: Pass 'c' to add an intercept term
        :param machine: Machine upon which the code is run. Can either be 'local' or 'Snellius'
        """
        super(SARIMA_model, self).__init__(machine,
                                           {'order': order,
                                            'seasonal_order': seasonal_order,
                                            'trend': trend},)

    def read_model(self, file_path, train_dat):
        """
        Method for reading previously trained models from a save-file
        :param file_path: Path to where the file can be found
        :param train_dat: Data used to train the model. Is needed for some algorithms to initialize the model
        :return: The trained model
        """
        # print(file_path)
        with open(file_path, 'rb') as f:
            model = pkl.load(f)

        # In earlier version of the model, the saved objects contained too much redundant data.
        # Some remnants may remain of these large pickled objects.
        if isinstance(model, sm_tsa.statespace.sarimax.SARIMAXResultsWrapper):
            trained_model = model
            with open(file_path, "wb") as f:
                pkl.dump(model.params, f, protocol=5)

        # The newly pickled objects should only contain a single numpy array with the
        # parameters for each term in the SARIMA model
        elif isinstance(model, np.ndarray):

            # the number of prameters is largely determined by the seasonal and regular  AR, I and MA terms
            number_of_params = sum(self.MODEL_PARAMS['seasonal_order'][:3]) + sum(self.MODEL_PARAMS['order'])

            # Step needed to verify whether the loaded model has the correct number of parameters
            if (self.MODEL_PARAMS['trend'] == 'c') or (self.MODEL_PARAMS['trend'] == 't'):
                number_of_params += 1
            elif self.MODEL_PARAMS['trend'] == 'ct':
                number_of_params += 2
            elif isinstance(self.MODEL_PARAMS['trend'], list):
                number_of_params += sum(self.MODEL_PARAMS['trend'])

            number_of_params += 1 # additional term for the \sigma^2_E

            if len(model) == number_of_params:
                target_dat = train_dat.sf_per_eco.values
                trained_model = sm.tsa.SARIMAX(target_dat, **self.MODEL_PARAMS)
                trained_model = trained_model.filter(model)
            else:
                raise ValueError(f'Model at "{file_path}" has {len(model)} parameters, while {number_of_params} are needed.')

        else:
            raise NotImplementedError(f'Unknown file type: {type(model)} encountered when loading model')
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
        file_name = self.get_file_name(eco_region, 'pkl')
        file = file_dir + file_name
        with open(file, "wb") as f:
            pkl.dump(model.params, f, protocol=5)


    def train_model(self, train_dat):
        """
        If no previously trained model is available, or if the available save file is corrupted, a new model needs to be
        trained.
        :param train_dat: Data needed for training the model
        :return: A trained model
        """
        start_year = str(train_dat.time.dt.year.min().values)
        eco_region = float(train_dat.eco_regions.values)

        print(f"starting training process of SARIMA model, at eco-region {eco_region} using data starting at {start_year}")

        target_data = train_dat.sf_per_eco
        model = sm.tsa.statespace.SARIMAX(target_data.values,
                                          **self.MODEL_PARAMS
                                          )
        fitted_model = model.fit(maxiter=100, disp=0)  # method='cg'

        # Save model for future usage
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
