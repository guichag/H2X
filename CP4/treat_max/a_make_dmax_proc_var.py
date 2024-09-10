"""Make daily max of 6-hours processed variable (Twb/RH) values"""

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


# Add AMMACATCH OBS
dsobs = 'AMMACATCH'
station = 'djougou'
y0obs = 2002
y1obs = 2011


### FUNC ###

def get_file_proc_var_dmax(ds='CP4A', var='twb', window=6, year0=1997, year1=2006, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get file of processed variable daily max time series"""
    y_ = str(year0) + '-' + str(year1)
    m_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"

    outdir = CP4OUTPATH + '/dmax/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + y_
    outfile = outdir + '/' + m_ + '_' + str(window) + 'h_dmax.nc'

    return outfile


def load_proc_var_dmax(ds='CP4A', var='twb', window=6, year0=1997, year1=2006, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load processed variable daily max time series"""
    outfile = get_file_proc_var_dmax(ds, var, window, year0, year1, months, lat_range, lon_range)
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--variable", type=str, default='twb')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[-6., 6.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[37., 50.])

    opts = parser.parse_args()

    ds = opts.dataset
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
    
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/dmax'):
        os.mkdir(CP4OUTPATH + '/dmax')
    outdir = CP4OUTPATH + '/dmax'

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

        data = load_proc_var_roll_mean(ds, var, y, window, months, lat_range, lon_range)
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


