import numpy as np
import netCDF4 as nc # used for reading netCDF4 data
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import xarray as xr
import logging


from datetime import datetime , timedelta

import enum

import glob
import os
from typing import Type, TypeVar # needed for type checking in constructor function



# Global variables
WRITE_PATH = '/gpfs/work1/0/ctdas/awoude/Ritten/predictor_vars/'
VAR_DATA_PATH = '/gpfs/work1/0/ctdas/awoude/Ritten/raw_dat/'




############################# Helper classes ###############################
class M_freq(enum.Enum):
    '''
    Used for labeling the measurement frequecy of the data_object
    '''
    hour = 1
    day = 2
    week = 3
    month = 4

    
class Aggr_style(enum.Enum):
    '''
    Used for specifying how to aggregate multiple datapoints
    '''
    SUM = 1 # sum
    AVG = 2 # average
    MAX = 3 # max
    MIN = 4 # min
    
class Week_day(enum.Enum):
    '''
    Used for translating a weekday to an integer
    '''
    
    mon = 0
    tue = 1
    wed = 2
    thu = 3
    fri = 4
    sat = 5
    sun = 6

class Param:
    '''
    Provides a quick overview of all the relevant information of a parameter
    '''
    def __init__(self, name, param_id, aggr_style: Aggr_style):
        '''
        A small helper class to store important information of the used variables
        :param name: The name of the variable. This should be the same name as the one used in the file names
        :param param_id: The ECMWF parameter ID. Not important for the functioning of the code, but it is good practice
        to keep the reference to the right parameters
        :param aggr_style: The used aggregation style. This should be an element from the enumerate class Aggr_style
        '''
        self.name = name
        self.id = param_id
        self.aggr_style = aggr_style
        
    def get_name(self):
        return self.name+'_'+self.aggr_style.name
    
    def __str__(self):
        return f'{type(self)}\n\
        name: {self.name}\n\
        id: {self.id}\n\
        aggr_style: {self.aggr_style}'
    
############################# Helper functions ###############################



next_day = lambda date, day: date + timedelta(days=(day-date.weekday())%7)


# lambda function for greating a datetime object out of the file name
extract_date = lambda file: pd.to_datetime(file.split('_')[-2])


############################# Core classes ###################################


class YearObj:
    '''
    Class used for reading data on a parameter for entire year and aggregate this according to the desired spatial/temporal
    resolution.
    '''
    def __init__(self, year, param: Param):
        '''
        Constructor function of the YearObj. This object handles the aggregation of an entire year worth of data of a single variable
        :param year: The year which is to be aggregated
        :param param: The exact parameter definition. The most important fields are param.name and param.aggr_style
        '''
        self.year = year
        self.param = param

        # This script is only able to aggregate hourly data. A start was made to make it possible the aggregate other
        # temporal resolutions, but this project was eventually abandoned.
        self.freq = M_freq.hour

        # A list of all the Saturdays of the year. Only the Saturdays are chosen as the series of scaling factors starts
        # at 2000-01-01, which is a Saturday
        self.week_dates = self.get_week_dates()

        # Flag which is used to check wether the relevant data has been loaded already
        self.is_loaded = False

        # Is used to check whether the loaded file is the correct shape
        self.shape = None
        
    def get_week_dates(self):
        
        # Lambda function for extracting the first saterday after a specific date.
        next_Saterday = lambda date: date + timedelta(days=(Week_day.sat.value-date.weekday())%7)
        
        jan_first = pd.to_datetime(self.year, format='%Y')
        start_date = next_Saterday(jan_first) # In CTDAS, every week starts at Saturday
        week_dates = [start_date + timedelta(i*7) for i in range(50)]
        while (max(week_dates)+timedelta(7)).year<(self.year+1): # Make sure to store all weeks untill the following week would be in the next year
            week_dates.append(start_date+timedelta((len(week_dates))*7))
        return week_dates
    
    def load_data(self):
        if self.freq != M_freq.hour:
            raise ValueError(f'Measurement frequency has the wrong value: {self.freq} instead of {M_freq.hour}')
        aggr_data = {}
        
        for week in self.week_dates:
            print(f'loading data from week {week}')
            dates = [week+timedelta(day) for day in range(7)]
            aggr_days = {}
            for day in dates:
                file = self.date_to_filename(day)
                if os.path.exists(file) and os.access(file, os.R_OK):
                    with nc.Dataset(file) as ds:
                        aggr_days[day] = self.aggr_func()(ds[self.param.name][:], axis=0).astype(np.float32)
                        dim_coords = {'lon':np.arange(ds.lonmin,ds.lonmax+1, 1), 
                                    'lat':np.arange(ds.latmin,ds.latmax+1, 1)}
                else:
                    logging.warning(f'Missing file {file}')
            if len(aggr_days)>1:
                if len(aggr_days) < 7:
                    if self.param.aggr_style == Aggr_style.s:
                        logging.warning(f'Skipped week {week} --> param.name: {self.param.name}, due to missing files')
                        print('--------- TODO: Fix the aggregate sum function for weeks with missing files ---------')
                        continue
                    logging.warning(f'Data for week {week} is incomplete')
                week_data = np.dstack(aggr_days.values())
                week_data = np.moveaxis(week_data, -1, 0)
                d = xr.DataArray(name=self.param.get_name(), 
                                 data=self.aggr_func()(week_data[:], axis=0),
                                 #aggr_f = self.param.aggr_style.name,
                                 dims=['lat','lon'], 
                                 coords=dim_coords)
                aggr_data[week] = d
                
            elif len(aggr_days) == 1:
                week_data = pd.Series(aggr_days)
                aggr_data[week] = week_data
                logging.warning(f'Data for week {week} is incomplete')
            else:
                logging.warning(f'Data for week {week} missing completely')
        self.data = xr.concat(list(aggr_data.values()), pd.Index(list(aggr_data.keys()), name="time"))
        self.is_loaded = True
        self.shape = self.data.shape
        self.freq = M_freq.week
        
    def check_completeness(self, write_dat = True):
        self.missing_files = []
        self.incomplete_weeks = []
        self.missing_weeks = []
        for week in self.week_dates:  
            dates = [week+timedelta(day) for day in range(7)]
            nr_accesible_files = 0
            for day in dates:
                file = self.date_to_filename(day)
                if os.path.exists(file) and os.access(file, os.R_OK):
                    nr_accesible_files += 1
                else:
                    self.missing_files.append(file)
            if nr_accesible_files > 0:
                if nr_accesible_files < 7:
                    if self.param.aggr_style == Aggr_style.s:
                        self.missing_weeks.append(week.strftime('%Y%m%d'))
                        continue
                    self.incomplete_weeks.append(week.strftime('%Y%m%d'))
            else:
                self.missing_weeks.append(week.strftime('%Y%m%d'))
        if write_dat:
            self.write_missing_data()
        
    
#     def load_data(self):
#         aggr_data = {}
#         self.freq = M_freq.day
#         for file in self.file_list[0:21]:
#             date = extract_date(file)
#             with nc.Dataset(file) as ds:
#                 print(f'loading data from {date}')
#                 aggr_data[date] = self.aggr_func()(ds[self.param.name][:], axis=0).astype(np.float32)
#         self.data = pd.Series(aggr_data)
#         self.is_loaded = True
#         self.shape = np.shape(self.data)
    
    
#     def aggr_to_week(self):
#         new_data_dict = {}
#         if self.freq != M_freq.day:
#             raise ValueError(f'Measurement frequency has the wrong value: {self.freq} instead of {M_freq.day}')
#         for i in range(int(len(self.data.index)/7)):
#             s = self.data[i:(i+7)]
#             d = self.aggr_func()(s, axis=0)
#             new_data_dict[self.data.index[i*7]]=d
#         self.data = pd.Series(new_data_dict)
#         self.freq = M_freq.week
#         self.shape = np.shape(self.data)
    
    
#     def aggr_to_month(self):
#         print('Still needs to be implemented!!!!')
#         new_data_dict = {}
#         if self.freq != M_freq.day:
#             raise ValueError(f'Measurement frequency has the wrong value: {self.freq} instead of {M_freq.day}')
#         for s in self.data.groupby(self.data.index.month):
#             d = self.aggr_func()(s[1], axis=0)
#             index = pd.to_datetime(str(self.year)+str(s[0]).zfill(2), format='%Y%m')
#             new_data_dict[index] = d

#         self.data = pd.Series(new_data_dict)
#         self.freq = M_freq.month
#         self.shape = np.shape(self.data)
        
        
        
    def aggr_func(self):
        if self.param.aggr_style == Aggr_style.SUM:
            return np.sum
        elif self.param.aggr_style == Aggr_style.AVG:
            return np.mean
        elif self.param.aggr_style == Aggr_style.MAX:
            return np.max
        elif self.param.aggr_style == Aggr_style.MIN:
            return np.min
        else:
            raise NotImplementedError(f"Aggr_style {self.param.aggr_style} is not implemented!")
            
    def dump(self):
        '''
        Function for writing the entire dataset object.
        '''
        file_name = WRITE_PATH + 'yearly/' + self.param.name + '_' + str(self.year) + '_' + str(self.freq.name) + '.pkl'
        
        with open(file_name, 'wb') as out:
            pkl.dump(self, out)
    
    def write_missing_data(self):
        cwd = os.getcwd()
        if len(self.missing_files) > 0:
            dir_name = cwd + '/missing_data/' + self.param.name + '/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            miss_file_name = dir_name + 'files.txt'
            with open(miss_file_name, 'a') as file:
                file.write(f'{os.linesep}'.join(self.missing_files))
                file.write(f'{os.linesep}')
            miss_week_name = dir_name + 'weeks.txt'
            with open(miss_week_name, 'a') as file:
                file.write(f'{os.linesep}'.join(self.missing_weeks))
                file.write(f'{os.linesep}')
            incomp_week_name = dir_name + 'incomplete.txt'
            with open(incomp_week_name, 'a') as file:
                file.write(f'{os.linesep}'.join(self.incomplete_weeks))
                file.write(f'{os.linesep}')
            
    
    def date_to_filename(self, date):
        return VAR_DATA_PATH + str(date.year) + '/' + self.param.name + '_' + date.strftime('%Y%m%d') + '_00p01.nc'


            
    def __str__(self):
        return f'obj_type: {type(self)}\n \
        year: {self.year}\n \
        param_name: {self.param.name}\n \
        param_id: {self.param.id}\n \
        shape: {self.shape}\n \
        aggr_style: {self.param.aggr_style}\n \
        is_loaded: {self.is_loaded}\n \
        m_freq: {self.freq}'
    
    
    
class DataObj:
    '''
    Class used for combining data from multiple parameters and years.
    '''
    def __init__(self, params, m_freq):
        self.params = params
        self.years_loaded = []
        self.ds = xr.Dataset(None)
        self.m_freq = m_freq
    
    
    def load_years(self, years):
        for y in years:
            if (len(self.ds) > 0) and (y in self.ds.time.dt.year):
                print(f'Year {y} has already been loaded')
            else:
                for p in self.params:
                    year_data = YearObj(y, p)
                    year_data.load_data()
                    self.aggr_data(year_data)
                    self.ds = xr.merge([self.ds, year_data.data])
                self.years_loaded.append(y)
    
    def check_data(self, years, write_dat = True):
        self.complete_years = []
        self.incomplete_years = []
        for p in self.params:
            for year in years:
                print(f'checking param {p.name} for year {year}')
                year_obj = YearObj(year, p)
                year_obj.check_completeness(write_dat)
                if len(year_obj.missing_files) == 0:
                    self.complete_years.append(year_obj)
                else:
                    self.incomplete_years.append(year_obj)
        for year_obj in self.incomplete_years: print(f'For param {year_obj.param.name}, year {year_obj.year} is incomplete') 

        
    
    def aggr_data(self, year_obj: YearObj):
        print('-----------------WARNING - AGGREGATING TO DIFF TIME FEQUENCIES NOT IMPLEMENTED YET----------------------------')
#         if self.m_freq == M_freq.day:
#             pass
#         elif self.m_freq == M_freq.week:
#             year_obj.aggr_to_week()
#         else:
#             raise NotImplementedError(f"m_freq {self.m_freq} is not implemented!")
    
    def dump(self):
        '''
        Function for writing the entire dataset object.
        '''
        if len(self.years_loaded)>0:
            param_names = self.params[0].name
            if len(self.params)>4:
                param_names = str(len(self.params))+'-params'
            else:
                for param in self.params[1:]:
                    param_names += '-' + param.name
            year_range = str(min(self.years_loaded)) + '-' + str(max(self.years_loaded))
            file_name = WRITE_PATH + param_names + '_' + year_range + '.pkl'

            with open(file_name, 'wb') as out:
                pkl.dump(self, out)
        else:
            logging.error('Tried saving an empty object')
            
    def dump_ds(self):
        '''
        Function for writing the entire dataset object.
        '''
        if len(self.years_loaded)>0:
            param_names = self.params[0].name
            if len(self.params)>4:
                param_names = str(len(self.params))+'-params'
            else:
                for param in self.params[1:]:
                    param_names += '-' + param.name
            year_range = str(min(self.years_loaded)) + '-' + str(max(self.years_loaded))
            file_name = WRITE_PATH + param_names + '_' + year_range + '.nc'

            with open(file_name, 'wb') as out:
                self.ds.to_netcdf(out)
        else:
            logging.error('Tried saving an empty object')
    
    
    def __str__(self):
        param_names = [p.name for p in self.params]
        return f'obj_type: {type(self)}\n \
        params: {param_names}\n \
        years_loaded: {self.years_loaded}\n \
        m_freq: {self.m_freq}'
     