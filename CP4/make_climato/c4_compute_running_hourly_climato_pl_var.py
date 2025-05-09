"""Compute pressure level variable hourly climatology"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from config import *
from CP4.make_climato.b4_compute_running_hourly_mean_pl_var import load_pl_var_running_hmean


### CST ###


### FUNC ###

def get_path_pl_var_running_hclimato(ds='CP4A', res=4, var='t_pl', year0=1997, year1=2006, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get path of pressure level variable mean over period of a specific year"""
    y_ = str(year0) + '-' + str(year1)
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    res_ = str(res) + 'km'

    outpath = CP4OUTPATH + '/' + ds + '/' + res_ + '/climato_hourly/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + y_

    return outpath


def load_pl_var_running_hclimato(ds='CP4A', res=4, var='t_pl', year0=1997, year1=2006, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.), levels=None):
    """Load pressure level variable climato"""
    m_ = "-".join([str(m) for m in months])
    if levels == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in levels])

    outpath = get_path_pl_var_running_hclimato(ds, res, var, year0, year1, window, months, lat_range, lon_range)
    outfile = outpath + '/' + m_ + '_' + str(window) + 'h_hourly_levels=' + str(lvls_) + '_running.nc'
    out = xr.open_dataarray(outfile)

    return out


def load_pl_var_hclimato(ds='CP4A', res=4, var='SM', year0=1997, year1=2006, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.), levels=None):
    """Load single level variable climato"""
    m_ = "-".join([str(m) for m in months])
    if levels == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in levels])

    outpath = get_path_pl_lvl_var_running_hclimato(ds, res, var, year0, year1, window, months, lat_range, lon_range)
    outfile = outpath + '/' + m_ + '_' + str(window) + 'h_hourly_levels=' + str(lvls_) + '_climato.nc'
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='t_pl')
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])
    parser.add_argument("--levels", nargs="+", type=float, default=None)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.variable
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    window = opts.window
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    lvls = opts.levels

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)

    res_ = str(res) + 'km'

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"

    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    if lvls == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in lvls])


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


    print('\n-- Compute hourly climato --')

    out_years_rm = []

    for y in years:
        print('\n{0}'.format(y))

        data_rm = load_pl_var_running_hmean(ds, res, var, y, window, months, lat_range, lon_range, lvls)
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

    outfile_hourly = outdir + '/' + months_ + '_' + str(window) + 'h_hourly_levels=' + str(lvls_) + '_running.nc'

    if not os.path.isfile(outfile_hourly):
        out_hclimato_rm.to_netcdf(outfile_hourly)
    else:
        print('Running hourly climato already exists!')

    outfile_clim = outdir + '/' + months_ + '_' + str(window) + 'h_hourly_levels=' + str(lvls_) + '_climato.nc'
    varclim.to_netcdf(outfile_clim)


print('Done')

