import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def detrend_data(ds):
    new_ds = xr.Dataset()
    data = ds.to_array()
    new_ds = data.rolling(time=11*52, min_periods=1).mean()
    return new_ds

with xr.open_dataset('./vars_per_eco_update.nc') as ds:
    var_dat = ds
used_vars = ['t2m_MAX', 'sd_MIN', 'sd_MAX']

deterended_data = detrend_data(var_dat[used_vars])

print(deterended_data)

deterended_data = deterended_data.to_dataset('variable')

print(deterended_data)

var_dat['t2m_MAX'].loc[dict(eco_regions=1.0)].plot()
plt.show()

deterended_data['t2m_MAX'].loc[dict(eco_regions=1.0)].plot()
plt.show()
