"""Compute CP4 output potential wet-bulb temperature and make rolling mean values"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from utils.meteo_constants import R, cp, p0
from DATA.MetUM_variables import create_CP4_filename
from CP4.make_climato.a1_compute_rolling_mean_var import load_var_roll_mean
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean


### CST ###

cst = R / cp


### FUNC ###



### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[10., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-20., -10.])
    parser.add_argument("--humtype", type=int, default=0)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    window = opts.window
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    humtype = opts.humtype

    years = np.arange(y0, y1+1, 1)
    days = np.arange(1, 30+1, 1)
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

    if not os.path.isdir(outdir + '/rolling_means'):
        os.mkdir(outdir + '/rolling_means')
    outdir = outdir + '/rolling_means'

    if not os.path.isdir(outdir + '/theta_wb'):
        os.mkdir(outdir + '/theta_wb')
    outdir = outdir + '/theta_wb'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    #~ Get data

    for y in years:
        print('\n{0}'.format(y))

        twb = load_proc_var_roll_mean(ds, res, 'twb', y, window, months, lat_range, lon_range) + 273.15
        psrfc = load_var_roll_mean(ds, res, 'p_srfc', y, window, months, lat_range, lon_range)

        thetawb = np.round(twb * (p0/psrfc)**cst, 5)
        thetawb = thetawb.where(thetawb != 0., np.nan)  # replace 0 (from tdata) with nan
        thetawb = thetawb - 273.15


        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h.nc'

        thetawb.to_netcdf(outfile)

        print('\n>>> {0} <<<'.format(outfile))


print('Done')

