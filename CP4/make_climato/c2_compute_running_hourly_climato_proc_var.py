"""Compute processed variable (Twb/RH/HI) hourly climatology (mean and percentile)"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from config import *
from CP4.make_climato.b2_compute_running_hourly_mean_proc_var import load_proc_var_running_hmean


### CST ###


### FUNC ###

def get_file_proc_var_running_hclimato(ds='CP4A', res=4, var='twb', window=6, year0=1997, year1=2006, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get file of processed variable hourly mean over period of a specific year"""
    y_ = str(year0) + '-' + str(year1)
    m_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"
    res_ = str(res) + 'km'

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/climato_hourly/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + y_
    outfile = outdir + '/' + m_ + '_' + str(window) + 'h_hourly_running.nc'

    return outfile


def load_proc_var_running_hclimato(ds='CP4A', res=4, var='twb', window=6, year0=1997, year1=2006, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load processed variable hourly climato"""
    outfile = get_file_proc_var_running_hclimato(ds, res, var, window, year0, year1, months, lat_range, lon_range)
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='twb')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[-6., 6.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[37., 50.])

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.variable
    window = opts.window
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    lat_range = opts.lat_range
    lon_range = opts.lon_range

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/climato_hourly'):
        os.mkdir(outdir + '/climato_hourly')
    outdir = outdir + '/climato_hourly'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/' + years_):
        os.mkdir(outdir + '/' + years_)
    outdir = outdir + '/' + years_


    print('\n-- Get {0}-hourly {1} data --'.format(window, var))

    out_years_rm = []

    for y in years:
        print('\n{0}'.format(y))

        data_rm = load_proc_var_running_hmean(ds, res, var, window, y, months, lat_range, lon_range)
        imonths = data_rm.groupby('time.month').groups
        #data_rm = data_rm.isel(time=slice(imonths[months[0]][0], imonths[months[-1]][-1]+1))
        out_years_rm.append(data_rm)

    out_years_rm = xr.concat(out_years_rm, dim='time', coords='minimal', compat='override')


    print('\n-- Compute {0}-{1} climato --'.format(years[0], years[-1]))

    months_days_idx = pd.MultiIndex.from_arrays([out_years_rm['time.month'].values, out_years_rm['time.day'].values])
    out_years_rm.coords['days'] = ('time', months_days_idx)
    out_hclimato_rm = out_years_rm.groupby('days').mean()
    out_hclimato_rm = out_hclimato_rm.reset_index('days')


    print('\n-- Compute {0}-{1} climatological mean --'.format(years[0], years[-1]))

    varclim = out_hclimato_rm.mean(dim='hour')
    varclim = varclim.mean(dim='days')  # mean across days


    print('\n-- Save --')

    outfile_hourly_rm = outdir + '/' + months_ + '_' + str(window) + 'h_hourly_running.nc'
    #if not os.path.isfile(outfile_hourly_rm):
    out_hclimato_rm.to_netcdf(outfile_hourly_rm)
    #else:
    #    print('Hourly climato already exists!')

    outfile_clim = outdir + '/' + months_ + '_' + str(window) + 'h_hourly_climato.nc'
    varclim.to_netcdf(outfile_clim)


print('Done')

