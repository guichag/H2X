"""Compute percentile of daily max Twb values
   On rolling N days windows"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from config import DATAPATH, CP4OUTPATH
from CP4.treat_max.a_make_dmax_proc_var import load_proc_var_dmax


### CST ###


### FUNC ###

def get_file_proc_var_dmax_running_quantile(ds='CP4A', res=4, var='twb', quantile=0.95, window_quantile=11, window=6, year0=1997, year1=2006, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get file of processed variable rolling quantile"""
    y_ = str(year0) + '-' + str(year1)
    m_ = "-".join([str(m) for m in months])
    q_ = str(int(quantile*100))
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"
    res_ = str(res) + 'km'

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/dmax/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + y_
    outfile = outdir + '/' + m_ + '_' + str(window) + 'h_dmax_' + str(window_quantile) + 'd_p' + q_ + '.nc'

    return outfile


def load_proc_var_dmax_roll_quantile(ds='CP4A', res=4, var='twb', quantile=0.9, window_quantile=11, window=6, year0=1997, year1=2006, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load processed variable rolling quantile"""
    outfile = get_file_proc_var_dmax_running_quantile(ds, res, var, quantile, window_quantile, window, year0, year1, months, lat_range, lon_range)
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
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--window_quantile", type=int, default=31)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.variable
    window = opts.window
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    q = opts.quantile
    qwindow = opts.window_quantile
    lat_range = opts.lat_range
    lon_range = opts.lon_range

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    q_ = str(int(q*100))

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


    print('\n-- Get data --')

    dmax = load_proc_var_dmax(ds, res, var, window, y0, y1, months, lat_range, lon_range)


    #~ Compute seasonal cycle of daily max

    months_days_idx = pd.MultiIndex.from_arrays([dmax['time.month'].values, dmax['time.day'].values])
    dmax.coords['days'] = ('time', months_days_idx)
    mean = dmax.groupby('days').mean()
    mean_ = mean.reset_index('days')


    iyears = dmax.groupby('time.year').groups

    out_yrs = []

    for y in years:
        print(y, end=' : ', flush=True)

        out_time = []
        out_data = []

        ydmax = dmax.isel(time=iyears[y])

        #~ get residual (remove seasonal cycle)
        ydmax_res = ydmax.values - mean.values
        ydmax_res = xr.DataArray(data=ydmax_res, dims=['time', 'latitude', 'longitude'], coords=dict(time=ydmax.time.values, latitude=(['latitude'], ydmax.latitude.values), longitude=(['longitude'], ydmax.longitude.values)))

        idays = ydmax.groupby('time.day').groups
        imonths = ydmax.groupby('time.month').groups

        for t0, t1 in zip(ydmax.time[:-qwindow+1], ydmax.time[qwindow-1:]):
            ydmax_res_ = ydmax_res.sel(time=slice(t0, t1))
            t = ydmax_res_.time[int(len(ydmax_res_.time)/2)]
            m0 = int(t0['time.month'].values)
            d0 = int(t0['time.day'].values)

            imonths_ = imonths[m0]
            idays_ = idays[d0]

            iall = [np.arange(i,i+qwindow,1) for i in idays_ if i in imonths_]
            iall = np.concatenate(iall)

            out_time.append(t)
            out_data.append(ydmax_res.isel(time=iall))

        times = xr.concat(out_time, dim='time')
        months_days_idx = pd.MultiIndex.from_arrays([times['time.month'].values, times['time.day'].values])

        out_yr = xr.DataArray(data=out_data, dims=['days', 'time', 'latitude', 'longitude'], coords=dict(days=(["days"], months_days_idx), latitude=(["latitude"], ydmax.latitude.values), longitude=(["longitude"], ydmax.longitude.values)))
        #out_yr = xr.DataArray(data=out_data, dims=['days', 'time', 'latitude', 'longitude'], coords=dict(latitude=(["latitude"], ydmax.latitude.values), longitude=(["longitude"], ydmax.longitude.values)))
        #out_yr.coords['days'] = ('time', months_days_idx)

        out_yrs.append(out_yr)


    out_yrs = xr.concat(out_yrs, dim='days')


    print('\n-- Compute quantile --')

    out_qs = []

    for idx in months_days_idx:
        print(idx, end=' : ', flush=True)

        data = out_yrs.sel(days=idx).values.reshape((qwindow)*len(years), len(out_yrs.latitude), len(out_yrs.longitude))

        q_res = np.quantile(data, 0.95, axis=0)
        q = q_res + mean.sel(days=idx)

        out_qs.append(q)   # np.quantile(data, 0.95, axis=0))


    out_qs = np.asarray(out_qs)

    out_qs = xr.DataArray(data=out_qs, dims=['days', 'latitude', 'longitude'], coords=dict(days=(["days"], months_days_idx), latitude=(["latitude"], ydmax.latitude.values), longitude=(["longitude"], ydmax.longitude.values)))
    out_qs = out_qs.reset_index('days')


    print('\n-- Save --')

    outfile = outdir + '/' + months_ + '_' + str(window) + 'h_dmax_' + str(qwindow) + 'd_p' + q_ + '.nc'

    if not os.path.isfile(outfile):
        out_qs.to_netcdf(outfile)
    else:
        print('{0}th percentile of {1} daily max already exists!'.format(q_, var))


print('Done')

