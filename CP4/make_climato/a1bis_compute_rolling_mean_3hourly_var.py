"""Compute CP4 output 3-hourly variable rolling mean values"""

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


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='PBLh')
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])

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

    #months = [months[0]-1] + months + [months[-1]+1]  # Take 1 month before and after -> allow rolling mean


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/rolling_means'):
        os.mkdir(outdir + '/rolling_means')
    outdir = outdir + '/rolling_means'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    #~ Get data

    print('\n-- Get data --')

    files = get_var_filenames(ds, res, var)
    files.sort()


    print('\n-- Compute rolling means --')

    for y in years:
        print('\n{0}'.format(y))

        yfiles = [f for f in files if '_{0}'.format(y) in f and '.{0}'.format(var_id[0]) not in f]

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h.nc'

        out_months = []

        for m in months:
            mfiles = [f for f in yfiles if '_%d%02d' % (y, m) in f]

            for fname in mfiles:
                ds = xr.open_dataset(fname)

                data = ds[var_id].assign_coords(longitude=ds.longitude - 360)
                data_sub = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

                out_months.append(data_sub)
                sys.exit()

        print('\n>>> {0} <<<'.format(outfile))

        out_months = xr.concat(out_months, dim='time')


        print('\nMake rolling mean')

        out_months_rm = out_months.rolling(time=window, center=True).mean()
        out_months_rm.to_netcdf(outfile)


print('Done')

