"""Compute CP4 output potential wet-bulb temperature on pressure level and make rolling mean values"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from utils.meteo_constants import R, cp, p0
from DATA.MetUM_variables import create_CP4_filename
from DATA.read_variables import load_twb_pl_data


### CST ###

cst = R / cp
lvls = [1000., 950., 925., 900., 850., 800., 700.]


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

    if not os.path.isdir(outdir + '/theta_wb_pl'):
        os.mkdir(outdir + '/theta_wb_pl')
    outdir = outdir + '/theta_wb_pl'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    #~ Get data

    print('\n-- Compute rolling means --')

    for y in years:
        print('\n{0}'.format(y))

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h.nc'

        out_months = []

        for m in months:

            for d in days:
                twbdata = load_twb_pl_data(ds, res, y, m, d, humtype, lat_range, lon_range)

                pressures = np.ones((twbdata.shape[2], twbdata.shape[3]))
                pressures = np.stack([pressures for i in range(len(twbdata.pressure))])
                pressures = np.asarray([pressures[i] * twbdata.pressure.values[i] for i in range(len(twbdata.pressure))])
                pressures = np.stack([pressures for i in range(len(twbdata.time))])

                thetawbdata_sub = np.round(twbdata * (p0/pressures)**cst, 5)
                thetawbdata_sub = thetawbdata_sub.where(thetawbdata_sub != 0., np.nan)  # replace 0 (from tdata) with nan

                out_months.append(thetawbdata_sub)


        print('\n>>> {0} <<<'.format(outfile))

        out_months = xr.concat(out_months, dim='time')


        print('\nMake rolling mean')

        out_months_rm = out_months.rolling(time=window, center=True).mean()
        out_months_rm.to_netcdf(outfile)


print('Done')

