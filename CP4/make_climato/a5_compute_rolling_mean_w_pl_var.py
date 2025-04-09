"""Convert CP4 output omega on pressure level into vertical wind speed and compute rolling mean values"""
"""https://www.ncl.ucar.edu/Document/Functions/Contributed/omega_to_w.shtml"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from utils.meteo_constants import R, g
from DATA.MetUM_variables import create_CP4_filename
from DATA.read_variables import get_var_filenames


### CST ###

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
    res = opts.resolution
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

    omega_id = create_CP4_filename('omega_pl')
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

    if not os.path.isdir(outdir + '/w_pl'):
        os.mkdir(outdir + '/w_pl')
    outdir = outdir + '/w_pl'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    #~ Get data

    print('\n-- Get data --')

    omegafiles = get_var_filenames(ds, res, 'omega_pl')
    omegafiles.sort()
    tfiles=  get_var_filenames(ds, res, 't_pl')
    tfiles.sort()


    print('\n-- Compute rolling means --')

    for y in years:
        print('\n{0}'.format(y))

        omegayfiles = [f for f in omegafiles if '_{0}'.format(y) in f and '.{0}'.format(omega_id[0]) not in f]
        tyfiles = [f for f in tfiles if '_{0}'.format(y) in f and '.{0}'.format(t_id[0]) not in f]

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + months_ + '_' + str(window) + 'h.nc'

        out_months = []
        for m in months:
            omegamfiles = [f for f in omegayfiles if '_%d%02d' % (y, m) in f]
            tmfiles = [f for f in tyfiles if '_%d%02d' % (y, m) in f]

            for omegafname, tfname in zip(omegamfiles, tmfiles):
                #print(fname)

                omegads = xr.open_dataset(omegafname)
                omegadata = omegads[omega_id].assign_coords(longitude=omegads.longitude - 360)
                omegadata_sub = omegadata.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), pressure=lvls)

                tds = xr.open_dataset(tfname)
                tdata = tds[t_id].assign_coords(longitude=tds.longitude - 360)
                tdata_sub = tdata.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), pressure=lvls)

                pressures = np.ones((omegadata_sub.shape[2], omegadata_sub.shape[3]))
                pressures = np.stack([pressures for i in range(len(omegadata_sub.pressure))])
                pressures = np.asarray([pressures[i] * omegadata_sub.pressure.values[i] for i in range(len(omegadata_sub.pressure))])
                pressures = np.stack([pressures for i in range(len(omegadata_sub.time))])

                wdata_sub = np.round(-(omegadata_sub * R * tdata_sub) / (g * pressures * 100), 5)  # (hPa -> Pa)

                out_months.append(wdata_sub)


        print('\n>>> {0} <<<'.format(outfile))

        out_months = xr.concat(out_months, dim='time')


        print('\nMake rolling mean')

        out_months_rm = out_months.rolling(time=window, center=True).mean()
        out_months_rm.to_netcdf(outfile)


print('Done')

