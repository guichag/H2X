"""Compute processed variable (Twb/RH/HI) rolling mean values"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from DATA.read_variables import load_proc_var_data, load_hi_data


### CST ###


### FUNC ###

def get_file_proc_var_roll_mean(ds='CP4A', res=4, var='twb', year=2000, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Get file of processed variable rolling mean values"""
    months_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    #assert lon_min < lon_max, "incorrect longitude range"
    res_ = str(res) + 'km'

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/rolling_means/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/' + str(year)
    outfile = outdir + '/' + months_ + '_' + str(window) + 'h.nc'

    return outfile


def load_proc_var_roll_mean(ds='CP4A', res=4, var='twb', year=2000, window=6, months=[3, 4, 5], lat_range=(-6., 6.), lon_range=(37., 50.)):
    """Load processed variable rolling mean values"""
    datafile = get_file_proc_var_roll_mean(ds, res, var, year, window, months, lat_range, lon_range)
    data = xr.open_dataarray(datafile)

    return data


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", type=str, default='twb')
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.variable
    y0 = opts.year0
    y1 = opts.year1
    window = opts.window
    months = opts.months
    lat_range = opts.lat_range
    lon_range = opts.lon_range

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    if (var == 'twb') or (var == 'rh'):
        load_func = load_proc_var_data
    elif var == 'hi':
        load_func = load_hi_data
    else:
        print('Incorrect variable ')

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

    if not os.path.isdir(outdir + '/rolling_means'):
        os.mkdir(outdir + '/rolling_means')
    outdir = outdir + '/rolling_means'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    print('\n-- Compute rolling means --')

    #months = np.arange(months[0], months[-1]+1, 1)   # +1 month before&after -> for 31-d rolling mean

    for y in years:
        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h.nc'

        print('\n>>> {0} <<<'.format(outfile))  # y

        out_months = []

        for m in months:
            print('\n{0}'.format(m))

            for d in days:
                print(d, end=' : ', flush=True)
                date = '_%d' % y + '%02d' % m + '%02d' % d

                data = load_func(ds, res, var, y, m, d)
                data = data.assign_coords(longitude=data.longitude - 360)
                data = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

                out_months.append(data)


            if (y == 1997) and (m == 1):
                out_months[0] = out_months[0].drop_vars('forecast_period')
                out_months[0] = out_months[0].drop_vars('forecast_reference_time')
                out_months[0] = out_months[0].drop_vars('height')

            #sys.exit()

        out_months = xr.concat(out_months, dim='time')  # , coords='minimal', compat='override')


        out_months_rm = out_months.rolling(time=window, center=True).mean()
        out_months_rm.to_netcdf(outfile)


print('Done')

