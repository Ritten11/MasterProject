#!/usr/bin/env python3

import os
import sys
import xarray as xr
import numpy as np
import netCDF4 as nc # used for reading netCDF4 data
import pandas as pd
import matplotlib.pyplot as plt
import glob

pers_dat_dir = '/projects/0/ctdas/awoude/Ritten/' # For data specifically stored for my perfonal usage
pers_file_dir = '/gpfs/work1/0/ctdas/awoude/Ritten/' # For storing and retrieving my own data files


regionsfile = '/gpfs/home5/awoude/Ritten/EKF/regions.nc' # Contains the data on the distribution of ecoregions

state_vec_file = '/projects/0/ctdas/input/ctdas_2012/covariances/gridded_NH/griddedNHparameters.nc'

resp_file = '/home/awoude/notebooks/ForRitten/SiB4_TER_avg.nc'

pred_var_path = '/gpfs/work1/0/ctdas/awoude/Ritten/predictor_vars/' # For retrieving the set of aggregated scaling vectors

flux_files = sorted(glob.glob(pers_dat_dir + 'flux_data/flux1x1_20??????00_20??????00.nc'))

monthly_sv_file = pers_dat_dir + 'monthly_sv.nc'
smoothed_sv_file = pers_dat_dir + 'smoothed_sv.nc'

def extract_date(f):
    date_hours = f.split('_')[-2] # extracting the first day of the week
    date = date_hours[:-2] # removing the final two number prepresenting the hours (?)
    return pd.to_datetime(date) # parse the date into the correct data type


with xr.open_dataset(monthly_sv_file) as ds:
    monthly_sv = ds
    
with xr.open_dataset(smoothed_sv_file) as ds:
    smoothed_sv = ds
    
with xr.open_dataset(state_vec_file) as ds:
    state_vec_params = ds
    # fixing minor error in the naming of dimensions in the source file:
    ds = ds.swap_dims({"longitude": "lon2", "latitude":"lat2", "lon":"lon2", "lat":"lat2"})
    ds = ds.swap_dims({"lon2": "lon", "lat2":"lat"})
    ds = ds.reset_coords(names=['longitude' ,'latitude'], drop=True)
    
    state_vec_ds = ds

# Loading of the prior flux landscape
with xr.open_dataset(flux_files[0]) as ds:
    date = extract_date(flux_files[0])
    combined_flux = ds.flux_bio_prior_mean + ds.flux_ocean_prior_mean
    combined_flux = combined_flux.assign_coords(time=date)
    prior_flux_ds = combined_flux

for i, f in enumerate(flux_files[1:],1):
    if i%100==0:
        print(f"Currently loading flux file #{i}")
    date = extract_date(f)
    with xr.open_dataset(f) as ds:
        combined_flux = ds.flux_bio_prior_mean + ds.flux_ocean_prior_mean
    combined_flux = combined_flux.assign_coords(time=date)
    prior_flux_ds = xr.concat([prior_flux_ds, combined_flux], dim='time')

# Loading the distribution of the ecoregions and area assoiated with each cell
with xr.open_dataset(regionsfile) as ds:
    ecoregion_da = ds[['regions','grid_cell_area']]
    

# Loading of the orignal scaling factor.
with xr.open_dataset(pers_file_dir+'sf.nc') as ds:
    sf_gridded = ds.rename({'sf':'scaling_factor'})

# Loadin of hte file containing the total ecosystem respiration (TER). Used as weights for the aggreation proces of going from grid space to statevector space
with xr.open_dataset(resp_file) as ds:
    ds = ds.swap_dims({"Longitude": "lon", "Latitude":"lat"})
    ds = ds.reset_coords(names=['Longitude' ,'Latitude'], drop=True)
    
    resp_ds = ds * ecoregion_da.grid_cell_area  # moving from [umol m^2 s^1] to [umol s^1]

# TER is only defined above lang. Where no TER is available, use the gridcell area as weight.
weight_matrix = np.where(np.isnan(resp_ds.TER), ecoregion_da.grid_cell_area, resp_ds.TER)
weight_da = xr.DataArray(name='grid_cell_weight', 
                         data=weight_matrix,
                         dims=['lat','lon'], 
                         coords=(ecoregion_da.lat, ecoregion_da.lon))

# Loading of the perdictor variables
with xr.open_dataset(pred_var_path+'36-params_2000-2020.nc') as ds:
    ds['s10m_AVG'] = np.sqrt(ds.u10m_AVG*ds.u10m_AVG+ds.v10m_AVG*ds.v10m_AVG)
    var_ds = xr.merge([ds, ecoregion_da, resp_ds, weight_da])

# Define which transcomregions should be included in the optimisation process.
included_trans_regions = ['North American Boreal', 'North American Temperate', 'Eurasia Boreal', 'Eurasia Temperate', 'Europe']

# Use the region files to find the ecoregions within the selected transcom regions
with xr.open_dataset(regionsfile) as ds:
    region_ds = ds

tc_names = [b''.join(n).decode().strip() for n in region_ds.transcom_names.values]
trans_filter = [name in included_trans_regions for name in tc_names]
needed_eco_regions = region_ds.xform.sel(dim_transcom23=trans_filter)
needed_eco_regions = needed_eco_regions.sum(axis=1)
needed_eco_regions = needed_eco_regions.rename({'carbontracker_240_regions':'eco_regions'})


var_only_ds = var_ds.drop(['regions', 'grid_cell_area', 'grid_cell_weight', 'TER', 'time', 'lon', 'lat'])


# Transform prior flux from [mol m-2 s-1] to [mol s-1] 
flux_per_s = prior_flux_ds * ecoregion_da.grid_cell_area
flux_per_s = flux_per_s.rename('prior_flux_per_s')





scaled_ds = var_only_ds * var_ds.grid_cell_weight

# Create a map allowing all elements within the statevector to be grouped according to element within the statevector
map_sv_to_eco = xr.merge([region_ds.regions.rename('eco_regions'), state_vec_ds.regions, sf_gridded]).groupby('regions').first()


# Move TER and prior flux to statevector space
sv_space = xr.merge([state_vec_ds.regions, resp_ds, flux_per_s]).groupby('regions').sum()

# Multiply the prior fluxes with the 'monthly' model
monthly_flux = (sv_space.prior_flux_per_s*monthly_sv).rename({'monthly_sv':'monthly_flux'})

# Multiply the prior fluxes with the 'smoothed' model
smoothed_flux = (sv_space.prior_flux_per_s*smoothed_sv).rename({'smoothed_sv':'smoothed_flux'})

# Mutliply the prior fluxes with the optimised state vector
opt_flux = (sv_space.prior_flux_per_s*map_sv_to_eco.scaling_factor).rename('opt_flux')
# xr.merge([monthly_flux, smoothed_flux, map_sv_to_eco.drop(['scaling_factor'])]).groupby('eco_regions').sum()

# Move model fluxes from sv space to ecoregion space
eco_space = xr.merge([monthly_flux, 
                      smoothed_flux, 
                      opt_flux, 
                      sv_space.prior_flux_per_s, 
                      map_sv_to_eco]).groupby('eco_regions').sum()

# Due to the way the scaling factors are determined, factors associated with a flux near 0 can get massive scaling factors. See Thesis section (??) for more in depth explanation
def cap_outliers(dat, scalar = 1):
    mean = np.mean(dat.values.flatten())
    sd = np.std(dat.values.flatten())
    
    dat = dat.where(dat < (mean+sd*scalar), (mean+sd*scalar))
    dat = dat.where(dat > (mean-sd*scalar), (mean-sd*scalar))
    return dat

sf_per_eco = eco_space.opt_flux/eco_space.prior_flux_per_s
monthly_sf = eco_space.monthly_flux/eco_space.prior_flux_per_s
smoothed_sf = eco_space.smoothed_flux/eco_space.prior_flux_per_s

eco_space['sf_per_eco'] = cap_outliers(sf_per_eco)
eco_space['monthly_sf'] = cap_outliers(monthly_sf)
eco_space['smoothed_sf'] = cap_outliers(smoothed_sf)

scaled_ds_per_eco = xr.merge([scaled_ds, 
                              var_ds.regions.rename('eco_regions'), 
                              var_ds.grid_cell_weight]).groupby('eco_regions').sum()

scaled_ds_per_eco = scaled_ds_per_eco / scaled_ds_per_eco.grid_cell_weight

area_per_eco = xr.merge([ecoregion_da.grid_cell_area, ecoregion_da.regions.rename('eco_regions')]).groupby('eco_regions').sum()
area_per_eco.attrs.update({'units':'m2', 'long name': 'surface area of each ecoregion'})
area_per_eco = area_per_eco.rename({'grid_cell_area':'eco_area'})

complete_ds = xr.merge([eco_space, scaled_ds_per_eco, area_per_eco])


filtered_ds = complete_ds.where(needed_eco_regions, drop=True)
filtered_ds = filtered_ds.drop('grid_cell_weight')
filtered_ds = filtered_ds.merge(var_ds[['regions','grid_cell_weight', 'grid_cell_area', 'TER', 'time', 'lon', 'lat']], compat='override')
print(filtered_ds)

file_name= pred_var_path + 'vars_per_eco_update.nc'
with open(file_name, 'wb') as out:
    filtered_ds.to_netcdf(out)

filtered_ds.to_netcdf(out)