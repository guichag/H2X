"""Compute variable data"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import DATADIR
from ICON.read_data.read_data import get_variable  #, get_times


### CST ###


### FUNC ###

def load_data_ICON(dataset='ICON', experiment='C5', zoom=8, variable='q2', year=1990, lat_range=(-30., 30.), lon_range=(-180., 180.)):
    """Load variable data"""
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"

    datapath = DATADIR + '/' + dataset + '/' + experiment + '/raw/' + variable + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/z' + str(zoom) + '/' + str(year)
    outfile = datapath + '/' + str(year) + '.nc'

    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ICON')
    parser.add_argument("--experiment", type=str, default='ngc4008')
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--time", type=str, default='PT3H')
    parser.add_argument("--variable", type=str, default='q2')
    parser.add_argument("--year0", type=int, default=2020)
    parser.add_argument("--year1", type=int, default=2049)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[-30., 30.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-180., 180.])

    opts = parser.parse_args()

    ds = opts.dataset
    exp = opts.experiment
    zoom = opts.zoom
    time = opts.time
    var = opts.variable
    y0 = opts.year0
    y1 = opts.year1
    lat_range = opts.lat_range
    lon_range = opts.lon_range

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    
    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)


    #~ Outdir

    if not os.path.isdir(DATADIR + '/' + ds):
        os.mkdir(DATADIR + '/' + ds)
    outdir = DATADIR + '/' + ds

    if not os.path.isdir(outdir + '/' + exp):
        os.mkdir(outdir + '/' + exp)
    outdir = outdir + '/' + exp

    if not os.path.isdir(outdir + '/raw'):
        os.mkdir(outdir + '/raw')
    outdir = outdir + '/raw'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/z' + str(zoom)):
        os.mkdir(outdir + '/z' + str(zoom))
    outdir = outdir + '/z' + str(zoom)


    #~ Get data

    for y in years:
        print(y, end=' : ', flush=True)

        data = get_variable(dataset=ds, experiment=exp, zoom=zoom, time=time, variable=var, year=y, lat_range=lat_range, lon_range=lon_range)

        if (var == 'pr') and (time == 'PT3H'):
            data = data * 3600  # kg m-2 s-1 -> mm h-1

        if (var == 'pr') and (time == 'P1D'):
            data = data * 86400  # kg m-2 s-1 -> mm d-1

        if (lat_range[0] < -30.) and (lat_range[1] > 30.):
            data = data.resample(time='1M').mean()   # FOR CLIMATOLOGICAL ANALYSES
        #sys.exit()

        #~ Save

        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + str(y) + '.nc'

        data.to_netcdf(outfile)


print('Done')

