"""Make daily max of 6-hours processed variable (Twb/RH) values"""
"""For SM contrast-based analyses"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from config import *
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean


### CST ###


### FUNC ###



### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='twb')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[10., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-20., -10.])

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
    months = [months[0]-1] + months + [months[-1]+1]  # Take 1 month before and after

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

    if not os.path.isdir(outdir + '/dmax'):
        os.mkdir(outdir + '/dmax')
    outdir = outdir + '/dmax'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/' + years_):
        os.mkdir(outdir + '/' + years_)
    outdir = outdir + '/' + years_


    print('\n-- Compute {0}-{1} {2} daily max time series --'.format(years[0], years[-1], var))

    out_ydmax = []

    for y in years:
        print(y, end=' : ', flush=True)

        data = load_proc_var_roll_mean(ds, res, var, y, window, months, lat_range, lon_range)
        ydmax = data.resample(time='D').max()
        out_ydmax.append(ydmax[:-1])  # remove last element (1st day of following year)

    out_ydmax = xr.concat(out_ydmax , dim='time')


    print('\n-- Save --')

    outfile = outdir + '/' + months_ + '_' + str(window) + 'h_dmax.nc'


    if not os.path.isfile(outfile):
        out_ydmax.to_netcdf(outfile)
    else:
        print('Daily max time series already exists!')


print('Done')

