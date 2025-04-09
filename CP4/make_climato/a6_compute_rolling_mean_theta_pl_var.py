"""Compute CP4 output potential temperature on pressure level and make rolling mean values"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from utils.meteo_constants import R, cp, p0
from DATA.MetUM_variables import create_CP4_filename
from DATA.read_variables import get_var_filenames


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
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])

    opts = parser.parse_args()

    ds = opts.dataset
    res= opts.resolution
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

    t_id = create_CP4_filename('t_pl')

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

    if not os.path.isdir(outdir + '/theta_pl'):
        os.mkdir(outdir + '/theta_pl')
    outdir = outdir + '/theta_pl'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    #~ Get data

    print('\n-- Get data --')

    tfiles = get_var_filenames(ds, res, 't_pl')
    tfiles.sort()


    print('\n-- Compute rolling means --')

    for y in years:
        print('\n{0}'.format(y))

        tyfiles = [f for f in tfiles if '_{0}'.format(y) in f and '.{0}'.format(t_id[0]) not in f]

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h.nc'

        out_months = []
        for m in months:
            tmfiles = [f for f in tyfiles if '_%d%02d' % (y, m) in f]

            for tfname in tmfiles:
                tds = xr.open_dataset(tfname)
                tdata = tds[t_id].assign_coords(longitude=tds.longitude - 360)
                tdata_sub = tdata.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), pressure=lvls)

                pressures = np.ones((tdata_sub.shape[2], tdata_sub.shape[3]))
                pressures = np.stack([pressures for i in range(len(tdata_sub.pressure))])
                pressures = np.asarray([pressures[i] * tdata_sub.pressure.values[i] for i in range(len(tdata_sub.pressure))])
                pressures = np.stack([pressures for i in range(len(tdata_sub.time))])

                thetadata_sub = np.round(tdata_sub * (p0/pressures)**cst, 5)
                thetadata_sub = thetadata_sub.where(thetadata_sub != 0., np.nan)   # replace 0 (from tdata) with nan

                out_months.append(thetadata_sub)


        print('\n>>> {0} <<<'.format(outfile))

        out_months = xr.concat(out_months, dim='time')


        print('\nMake rolling mean')

        out_months_rm = out_months.rolling(time=window, center=True).mean()
        out_months_rm.to_netcdf(outfile)


print('Done')

