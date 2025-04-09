"""Compute multi level variable hourly mean values
   From 6-hourly averaged data"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from CP4.make_climato.a3_compute_rolling_mean_multi_level_var import load_multi_level_var_roll_mean


### CST ###


### FUNC ###

def get_file_multi_lvl_var_hmean(ds='CP4A', res=4, var='SM', year=2000, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.), level=0.05):
    """Get file of multi level variable hourly mean values"""
    months_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"
    res_ = str(res) + 'km'

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/running_hourly_means/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + str(year)
    outfile = outdir + '/' + months_ + '_' + str(window) + 'h_level=' + str(level) + '.nc'

    return outfile


def load_multi_lvl_var_running_hmean(ds='CP4A', res=4, var='SM', year=2000, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.), level=0.05):
    """Load multi level variable hourly mean values"""
    outfile = get_file_multi_lvl_var_hmean(ds, res, var, year, window, months, lat_range, lon_range, level)
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='SM')
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--daily_window", type=int, default=11)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])
    parser.add_argument("--level", type=float, default=0.05)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.variable
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    window = opts.window
    dwindow = opts.daily_window
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    lvl = opts.level

    years = np.arange(y0, y1+1, 1)
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

    if not os.path.isdir(outdir + '/running_hourly_means'):
        os.mkdir(outdir + '/running_hourly_means')
    outdir = outdir + '/running_hourly_means'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    print('\n-- Compute hourly means --')

    for y in years:
        print('\n{0}'.format(y))

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h_level=' + str(lvl) + '.nc'

        data = load_multi_level_var_roll_mean(ds, res, var, y, window, months, lat_range, lon_range, lvl)

        out_time = []
        out_data = []

        for t0, t1 in zip(data.time[:-(dwindow*24)][::24], data.time[(dwindow*24):][::24]):
            data_ = data.sel(time=slice(t0, t1))
            t = data_.time[int(len(data_.time)/2)]
            out_time.append(t)
            out_data.append(data_.groupby('time.hour').mean())

            print(t.values, end=' : ', flush=True)

        times = xr.concat(out_time, dim='time')
        out = xr.concat(out_data, dim=times)

        out.to_netcdf(outfile)


print('Done')

