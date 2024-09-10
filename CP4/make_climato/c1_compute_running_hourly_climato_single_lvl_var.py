"""Compute single level variable hourly climatology"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from config import *
from CP4.make_climato.b1_compute_running_hourly_mean_single_lvl_var import load_single_lvl_var_running_hmean


### CST ###


### FUNC ###

def get_path_single_lvl_var_running_hclimato(ds='CP4A', var='q2', year0=1997, year1=2006, lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get file of single level variable mean over period of a specific year"""
    y_ = str(year0) + '-' + str(year1)
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]

    outpath = CP4OUTPATH + '/climato_hourly/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + y_

    return outpath


def load_single_lvl_var_running_hclimato(ds='CP4A', var='q2', year0=1997, year1=2006, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load single level variable climato"""
    m_ = "-".join([str(m) for m in months])
    outpath = get_path_single_lvl_var_running_hclimato(ds, var, year0, year1, lat_range, lon_range)
    outfile = outpath + '/' + m_ + '_' + str(window) + 'h_hourly_running.nc'
    out = xr.open_dataarray(outfile)

    return out


def load_single_lvl_var_hclimato(ds='CP4A', var='q2', year0=1997, year1=2006, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load single level variable climato"""
    m_ = "-".join([str(m) for m in months])
    outpath = get_path_single_lvl_var_running_hclimato(ds, var, year0, year1, lat_range, lon_range)
    outfile = outpath + '/' + m_ + '_' + str(window) + 'h_hourly_climato.nc'
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--variable", type=str, default='q2')
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[-6., 6.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[37., 50.])

    opts = parser.parse_args()

    ds = opts.dataset
    var = opts.variable
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    window = opts.window
    lat_range = opts.lat_range
    lon_range = opts.lon_range

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"

    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/climato_hourly'):
        os.mkdir(CP4OUTPATH + '/climato_hourly')
    outdir = CP4OUTPATH + '/climato_hourly'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/' + years_):
        os.mkdir(outdir + '/' + years_)
    outdir = outdir + '/' + years_


    print('\n-- Compute running hourly climato --')

    out_years_rm = []

    for y in years:
        print('\n{0}'.format(y))

        data_rm = load_single_lvl_var_running_hmean(ds, var, y, window, months, lat_range, lon_range)
        #imonths = data_rm.groupby('time.month').groups
        #data_rm = data_rm.isel(time=slice(imonths[months[0]][0], imonths[months[-1]][-1]+1)) -> consider only months
        out_years_rm.append(data_rm)

    out_years_rm = xr.concat(out_years_rm, dim='time', coords='minimal', compat='override')

    sys.exit()

    print('\n-- Compute {0}-{1} climato --'.format(years[0], years[-1]))

    months_days_idx = pd.MultiIndex.from_arrays([out_years_rm['time.month'].values, out_years_rm['time.day'].values])
    out_years_rm.coords['days'] = ('time', months_days_idx)
    out_hclimato_rm = out_years_rm.groupby('days').mean()
    out_hclimato_rm = out_hclimato_rm.reset_index('days')
    #out_hclimato_rm = out_years_rm.mean(dim="year")   # -> get climatological mean running diurnal cycle


    print('\n-- Compute {0}-{1} climatological mean --'.format(years[0], years[-1]))

    if (var == 'lsRain') or (var == 'evapSurface'):
        varclim = out_hclimato_rm * 3600
        varclim = varclim.mean(dim='hour') * 24 # daily cum
    else:
        varclim = out_hclimato_rm.mean(dim='hour')  # mean across hours of day
    varclim = varclim.mean(dim='days')  # mean across days


    print('\n-- Save --')

    outfile_hourly_rm = outdir + '/' + months_ + '_' + str(window) + 'h_hourly_running.nc'
    #if not os.path.isfile(outfile_hourly_rm):
    out_hclimato_rm.to_netcdf(outfile_hourly_rm)
    #else:
    #    print('Running hourly climato already exists!')

    outfile_clim = outdir + '/' + months_ + '_' + str(window) + 'h_hourly_climato.nc'
    varclim.to_netcdf(outfile_clim)


print('Done')


