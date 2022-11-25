# <center> Improving Data Assimilation Approach for Estimating CO2 Surface Fluxes Using ML </center>
## <center> Code repository of Master Thesis </center>
## <center> by H.M. Roothaert </center>

<div style="text-align: right"> Internal Supervisor: Dr. Celestine P. Lawrence (Artificial Intelligence, University of Groningen) <br>
External Supervisors: Prof. Dr. Wouter Peters (University of Wageningen, The Netherlands) <br>
Auke M. van der Woude (, MSc.) (University of Wageningen, The Netherlands) </div>

This repository contains all code used for conducting the research done in my [Master's thesis](https://fse.studenttheses.ub.rug.nl/id/eprint/28972). This code is roughly divided into 5 parts:
- Data retrieval
- Data pre-processing
- Hyperparameter fitting
- Model framework
- Figure generation

Note that some parts of the project have been done on [<i>Snellius</i>](https://www.surf.nl/en/dutch-national-supercomputer-snellius), the Dutch supercomputer, while others were done on a local machine. 
Both of these machines are Linux based, so both directory structures use the UNIX naming conventions. 
Nonetheless, this slightly complicates the running instructions, as 2 separate directory structures and used. 
Which machine has been used for this step of the project is indicated by a _(Snellius)_,  _(local)_, or _(Snellius/local)_ flag behind the header of each section.  
Furthermore, some used files are too large to upload to GitHub, or their terms of use do not allow for sharing. 
These issues will be stated clearly whenever they occur.

## Data retrieval _(Snellius)_

The data retrieval of the environmental conditions is handled by Dataloader.py. Note that the raw data varies in both spatial and temporal resolution. 
Some datasets provide hourly values on global 1X1 degree grid cells, while others are on weekly 1X1 degree grid cells. 
Keeping the original resolution of all datasets would therefore result in a substantial amount of redundant information. 
As such, Dataloader.py aggregates all data to the highest resolution used within this thesis: weekly values on 1X1 degree grid cells.

Note that all the used data is stored on Snellius. 
One could download all data to a local machine, but the combined size of all used data is several terabytes. 
This is therefore not recommended. Nonetheless, the running instructions are as follows:


First, make sure that the original files are provided in daily NetCDF4 files per variable, and use the following naming convention:
`[base_dir]/[year]/[var_name]_[year][month][day]_[hour]p[minute].nc`

- `[base_dir]`: the directory containing all the raw environmental data. 
- `[year]`: the year in which the data starts. 
- `[var_name]`: the name of the variable. For a list of the used variables, see Table D.5 of [the accompanying thesis](https://fse.studenttheses.ub.rug.nl/id/eprint/28972).
- `[month]`: The month in which the data starts.
- `[day]`: The day in which the data starts.
- `[hour]`: The hour in which the data starts. As we are using daily data files, this value is always `00`.
- `[minute]`: The minute in which the data starts. As we are using daily data files, this value is always `01`.

So an example of a properly named data file of the `t2m` variable which of the first on January 2000 would be `[base_dir]/2000/t2m_20000101_00p00.nc`

### IMPORTANT: check the spatial resolution _(Snellius)_
Make sure all files are on the proper spatial resolution. If a file has a higher spation resolution, you can use the `Scripts/regrid.sh` script to downsample the file to the right resolution. Note that this script is applied to all files in the directory in which it is run. 
To run the script, simply copy `regrid.sh` and `grid_layout.txt` to the directory in which the files are stored and use the following commands:

Make the file executable by the user:
`chmod +x regrid.sh`

Run the file:
`./regrid.sh`

### Dubbel checking that all files are present and working _(Snellius)_

As there are a substantial amount of files (365.25x21 > 9000 files per variable), it is important to verify that all the required files are stored in the right location and are readable. 
This is easily checked in using the `Dataloader.check_data()` function. 
This function logs all missing files and keeps track of all incomplete variables. 
The Dataloader object can be found in the Scripts folder and to get it working, two directories need to be hard coded. 
These are the `VAR_DATA_PATH` and `WRITE_PATH` directories, which should link to the previously used `[base_dir]` and the desired directory to which the down-sampled files should be written.

The Dataloader.check_data() function writes all the missing files to a new directory missing_files. 
See Scripts/datacheck_script.py for an example of how to use this function. 
Make sure this function is not producing any output before continuing any further.

### Lowering the temporal resolution _(Snellius)_

Once it has been verified that the spatial resolution is correct and no files are missing, it is time to change the temporal resolution. 
The provided scaling factors are on a weekly resolution, so it makes sense to also change the environmental variables to weekly values. 
This is also done using the `Dataloader` object. This time, the `Dataloader.load_years()` function loads all the given years for all parameters and aggregates this to weekly values according to the passed aggregation function. 
See `Scripts/dataloader_script.py` for an example.

Once the data has been loaded into the `Dataloader` object, it can be written to a pickle file using the `Dataloader.dump()` function, or to a NetCDF4 file by using the `Dataloader.dump_ds()` function.
The resulting file is named `[x]-params_[first_year]-[last-year].pkl`, or `[x]-params_[first_year]-[last-year].nc` respectively, where `x` is the number of given parameters and `[first-year]-[last-year]` is the range of the used years.

This concludes the data loading part. The next-up are the steps used to preprocess the data.

## Pre-processing the data
This is the point where the extensive data analysis has led to several adjustments in the project. As such, the code for this part has become a bit messy. The main problem is that we decided to switch from modeling the entire state vector to modeling the scaling factor per ecoregion. As a result, the data files used at the beginning of the thesis are shaped using the full state vector, while the later part of the thesis only uses a selected number of ecoregions.

To try and create some order in the chaos that has been created, this preprocessing step has been split into two sections: The state vector aggregation and the ecoregion aggregation

### State vector aggregation _(Snellius)_
The data files aggregated to the state vector level were only used in Part I of the thesis. This part focussed on the current implementation of the smoothed model and the proposed improvement of the monthly mean model. Both of these models are rather simple, and thus they have been included in the data preprocessing step.

The first step is transforming the scaling factors, which have been provided on a 360x180 gridspace, to their state vector representation of 9835 elements. 
This process is handled by the `Scripts/sv_extraction_script.py`, which produces the `weekly_sv.nc` file. 

Once the analyzed statevector has been extracted, the `smoothed` and `monthly` model are generated in the `Scripts/smoothed_monthly_sv.ipynb` notebooks and are saved as `smoothed_sv` and `monthly_sv` respectively.

These are the models used for Part I of the thesis.
The final part of the thesis focussed on the effective scaling factor per ecoregion.

### Ecoregion aggregation _(Snellius)_
At this point, I had gotten more familiar with XArray and managed to combine all the relevant information into a single dataset. 
The result is a dataset containing the effective scaling factors per ecoregion, the combined fluxes per ecoregion, and the environmental conditions per ecoregion. 
This dataset is generated in `Scripts/aggregate_by_eco.py`. 


Now that all data is centralized into a single, relatively small, dataset, it can be downloaded and processed further on a local machine.
Therefore, the final step of the data preprocessing has been done on a local machine. 
This is done in the `Scripts/to_anomaly.ipynb` notebook, and produces a final dataset `vars_per_eco_monthly_anomaly.nc`

## Hyperparameter fitting
Now that the data is ready, it is time to investigate how to implement the ML models.
As mentioned in the thesis, the chosen ML models are the seasonal auto-regressive integrated moving average (SARIMA) and seasonal auto-regressive integrated moving average with exogenous variables(SARIMAX) models. 
How these models have been fitted to the data is explained in the `SARIMA.ipynb` and `SARIMAX.ipynb` notebooks.

## Model framework _(Snellius/local)_

After having determined the set of hyperparameters for the SARIMA and SARIMAX models, it is time to include them in the broader modeling framework. 
To simplify adding new models, the abstract object `ML_model` implemented in `ML_models/base_model.py` contains all the functions which function exactly the same across all models.
All that is required to make a new model is:
- Giving the model a name by defining `MODEL_NAME`.
- Overwrite the constructor method such that model parameters can be stored in `ML_model.MODEL_PARAMS`
- Implement the training procedure
- Implement how the model generates predictions
- Define how the model should be saved
- Define how previously saved models can be loaded

Once these functions have been implemented, the models can be executed and evaluated using the function `ML_model.run_model()` and `ML_model.test_model()`. 
See `./run_ML_models.ipynb` for examples of how to run each of the 4 implemented models.

## Figure generation
After having defined the data and trained the models, it is time to visualize the performance of the models.
All figures used for the first part of the thesis are gathered in the `./figures_used_in_thesis_p1.ipynb`. 
Unfortunately, these figures have all been generated on Snellius, and translating the directories to those of the local machine has proven to be cumbersome. This is therefore still a work in progress.

The figures of Part II are gathered in the `./figures_used_in_thesis_p2.ipynb` notebook, and links to the figures in the appendix are provided in the `./figures_used_in_thesis_app.ipynb` notebook


