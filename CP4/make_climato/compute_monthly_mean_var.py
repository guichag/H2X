"""Compute CP4 output variable monthly mean values"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from DATA.MetUM_variables import create_CP4_filename
from DATA.read_variables import get_var_filenames


### CST ###


### FUNC ###

def get_file_var_monthly_mean(ds='CP4A', res=4, var='q2', year=2000, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get file of processed variable monthly mean values"""
    months_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    res_ = str(res) + 'km'

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/monthly_mean/' + var + '/' + str(year)  #'/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + str(year)
    outfile = outdir + '/' + months_ + '.nc'

    return outfile


def load_var_monthly_mean(ds='CP4A', res=4, var='q2', year=2000,  months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load processed variable monthly mean values"""
    outfile = get_file_var_monthly_mean(ds, res, var, year,  months, lat_range, lon_range)
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='q2')
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[None, None])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[None, None])

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

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    var_id = create_CP4_filename(var)

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

    if not os.path.isdir(outdir + '/monthly_mean'):
        os.mkdir(outdir + '/monthly_mean')
    outdir = outdir + '/monthly_mean'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var


    #~ Get data

    print('\n-- Get data --')

    files = get_var_filenames(ds, var)
    files.sort()


    print('\n-- Compute monthly means --')

    for y in years:
        print('\n{0}'.format(y))

        yfiles = [f for f in files if '_{0}'.format(y) in f and '.{0}'.format(var_id[0]) not in f]

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '.nc'

        out_times = []
        out_months = []

        for m in months:
            mfiles = [f for f in yfiles if '_%d%02d' % (y, m) in f]

            out_days = []

            for fname in mfiles:
                #print(fname)

                ds = xr.open_dataset(fname)

                data = ds[var_id].assign_coords(longitude=ds.longitude - 360)
                data_sub = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
                dmean = data_sub.mean(dim='time')

                out_days.append(dmean)

            out_days = xr.concat(out_days, dim='time', coords='minimal', compat='override')

            out_times.append(data.time[0])
            out_months.append(out_days.mean(dim='time'))

        times = xr.concat(out_times, dim='time')
        out_months = xr.concat(out_months, dim=times, coords='minimal', compat='override')

        print('\n>>> {0} <<<'.format(outfile))  # y

        out_months.to_netcdf(outfile)


print('Done')


