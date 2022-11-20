#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc # used for reading netCDF4 data
import pandas as pd
import xarray as xr

import glob
import os, sys, stat
import datetime   # Used for comparing dates with each other

def get_state_vec(f, state_elem_ds):
    with xr.open_dataset(f) as ds:
        sf_total  = ds['flux_multiplier_m']
        # combined dataset in which the elements of the state vector and the scaling vectors are aligned based on longitude and latitude
        comb = state_elem_ds.merge(sf_total)
        gr = comb.groupby('regions').max()['flux_multiplier_m'] # All values should be the same, so taking the max or min should not make a difference
        return gr

def extract_date(f):
    date_hours = f.split('_')[-2] # extracting the first day of the week
    date = date_hours[:-2] # removing the final two number prepresenting the hours (?)
    return pd.to_datetime(date) # parse the date into the correct data type


    
pers_dat_dir = '/projects/0/ctdas/awoude/Ritten/' # For data specifically stored for my perfonal usage
pers_file_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/' # For storing and retrieving my own data files
sf_files = sorted(glob.glob(pers_dat_dir + 'flux_data/flux1x1_20??????00_20??????00.nc'))

state_vec_file = '/projects/0/ctdas/input/ctdas_2012/covariances/gridded_NH/griddedNHparameters.nc'

file_name = 'weekly_sv.nc'
file_path = pers_file_dir + file_name
with xr.open_dataset(state_vec_file) as ds:
    state_vec_params = ds
    print(len(np.unique(ds['regions'][:])))
    # fixing minor error in the naming of dimension in the source file:
    ds = ds.swap_dims({"longitude": "lon2", "latitude":"lat2", "lon":"lon2", "lat":"lat2"})
    ds = ds.swap_dims({"lon2": "lon", "lat2":"lat"})
    ds = ds.reset_coords(names=['longitude' ,'latitude'], drop=True)
    
    state_elem_ds = ds
ds = xr.Dataset(None)
sv = {}
for i, f in enumerate(sf_files):
    with xr.open_dataset(f) as ds:
        print(f"Currently processing file #{i}")
        date = extract_date(f)
        sf_per_er = get_state_vec(f, state_elem_ds)
        sv[date] = sf_per_er

sv_dat_arr = xr.concat(list(sv.values()), pd.Index(list(sv.keys()), name="time"))
sv_dat_arr.attrs['Description'] = 'Weekly analysed state vector'
sv_dat_arr.attrs['Units'] = 'None'
sv_dat_arr.to_netcdf(file_path)
os.chmod(file_path, stat.S_IRWXU) # Needed for setting permissions after writing