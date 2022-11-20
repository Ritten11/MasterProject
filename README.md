# <center> Improving Data Assimilation Approach for Estimating CO2 Surface Fluxes Using ML </center>
## <center> Code repository of Master Thesis </center>
## <center> by H.M. Roothaert </center>

<div style="text-align: right"> Internal Supervisor: Dr. Celestine P. Lawrence (Artificial Intelligence, University of Groningen) <br>
External Supervisors: Prof. Dr. Wouter Peters (University of Wageningen, The Netherlands) <br>
Auke M. van der Woude (, MSc.) (University of Wageningen, The Netherlands) </div>

This repository contains all code used for conducting the research done in my [Master thesis](https://fse.studenttheses.ub.rug.nl/id/eprint/28972). This code is roughly devided into 5 parts:
- Data retrieval
- Data pre-processing
- Model framework
- Hyperparameter fitting
- Figure generation

Note that some parts of the project have been done on [<i>Snellius</i>](https://www.surf.nl/en/dutch-national-supercomputer-snellius), the Dutch supercomputer, while others were done on a local machine. Both of these machines are Linux based, so both directory structures use the UNIX naming conventions. Nonetheless, this slightly complicates the running instructions, as 2 seperate directory structures and used. Furthermore, some of the used files are too large to upload to GitHub, or their terms of use do not allow for sharing. These issues will be stated clearly whenever they occur.

## Data retrieval

The data retrieval of the environmental conditions is handled by the `Dataloader.py`. Note that the raw data varies in both spatial and temporal resolution. Some datasets provide hourly values on a global 1X1 degree grid cells, while others are on weekly 1X1 degree gridcells. Keeping the orginal resolution of all dataset would therefore result in a substantial amount of redundant information. As such, `Dataloader.py` aggregates all data to the highest resolution used within this thesis: weekly values on 1X1 degree grid cells. 

Note that all the used data is stored on Snellius. One could download al data to a local machine, but the combined size of all used data is several terabytes. This is therefore not recommended. Nonetheless, the running instructions are as follows:

First, make sure that the original files are proveded in daily NetCDF4 files per variable, and use the following naming convention:
`[base_dir]/[year]/[var_name]_[year][month][day]_[hour]p[minute].nc`

- `[base_dir]`: the directory containing all the raw environmental data. 
- `[year]`: the year in which the data starts. 
- `[var_name]`: the name of the variable. For a list of the used variables, see Table D.5 of [the accompanying thesis](https://fse.studenttheses.ub.rug.nl/id/eprint/28972).
- `[month]`: The month in which the data starts.
- `[day]`: The day in which the data starts.
- `[hour]`: The hour in which the data starts. As we are using daily data files, this value is always `00`.
- `[minute]`: The minute in which the data starts. As we are using daily data files, this value is always `01`.

So an example of a properly named data file of the `t2m` variable which of the first on january 2000 would be `[base_dir]/2000/t2m_20000101_00p00.nc`

### IMPORTANT: check the spatial resolution
Make sure all files are on the proper spatial resolution. If a file has a higher spation resolution, you can use the `Scripts/regrid.sh` script to downsample the file to the right resolution. Note that this script is applied to all files in the directory in which it is run. 
To run the script, simply copy `regrid.sh` and `grid_layout.txt` to the directory in which the files are stored and use the following commands:

Make the file executable by the user:
`chmod +x regrid.sh`

Run the file:
`./regrid.sh`

### Dubbel checking that all files are present and working

As there are a substantial amount of files (365.25x21 > 9000 files per variable), it is important to verify that all the required files are stored in the right location and are readable. 
This is easily checked in using the `Dataloader.check_data()` function. This function logs all missing files and keeps track of all incomplete variables. 
The `Dataloader` object can be found in the `Scripts` folder and to get it working, two directories need to be hardcoded. 
These are the `VAR_DATA_PATH` and `WRITE_PATH` directories, which should link to the previously used `[base_dir]` and the desired directory in which the down sampled files should be written to.

The `Dataloader.check_data()` function writes all the missing files to a new directory `missing_files`. 
See `Scripts/datacheck_script.py` for an example on how to use this function. 
Make sure this function is not producing any output before continuing any further

### Lowering the temporal resolution

Once it has been verified that the spatial resolution is correct and no files are missing, it is time to change the temporal resolution. 
The provided scaling factors are on a weekly resolution, so it makes sense to also change the environmental variables to weekly values. 
This is also done using the `Dataloader` object. This time, the `Dataloader.load_years()` function loads all the given years for all parameters and aggregates this to weekly values according to the passed aggregation function. 
See `Scripts/dataloader_script.py` for an example.

Once the data has been loaded into the `Dataloader` object, it can be written to a pickle file using the `Dataloader.dump()` function, or to a NetCDF4 file by using the `Dataloader.dump_ds()` funciton.
The resulting file is named `[x]-params_[first_year]-[last-year].pkl`, or `[x]-params_[first_year]-[last-year].nc` respectively, where `x` is the number of given parameters and `[first-year]-[last-year]` is the range of the used years.

This concludes the data loading part. The next-up are the steps used to preprocess the data.

## Pre-processing the data