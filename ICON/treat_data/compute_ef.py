"""Compute evaporative fraction"""
"""from daily LH and SH ICON data"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import DATADIR
from ICON.make_data.a_compute_data import load_data_ICON


### CST ###


### FUNC ###

def compute_ef(lhdata, shdata):
    ef = lhdata / (lhdata + shdata)
    ef = ef.where((lhdata <= 0.) & (shdata <= 0.))  # compute EF only with both fluxes towards the atmosphere (i.e. <= 0 since DOWNWARD heat fluxes)
    out = ef.where((ef > 0.) & (ef <= 1.))

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ICON')
    parser.add_argument("--experiment", type=str, default='ngc4008')
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--year0", type=int, default=2020)
    parser.add_argument("--year1", type=int, default=2049)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[-30., 30.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-180., 180.])

    opts = parser.parse_args()

    ds = opts.dataset
    exp = opts.experiment
    zoom = opts.zoom
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

    if not os.path.isdir(outdir + '/ef'):
        os.mkdir(outdir + '/ef')
    outdir = outdir + '/ef'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/z' + str(zoom)):
        os.mkdir(outdir + '/z' + str(zoom))
    outdir = outdir + '/z' + str(zoom)


    #~ Get data

    print("-- Get data --")

    efs = []

    for y in years:
        print(y, end=' : ', flush=True)

        lh = load_data_ICON(dataset=ds, experiment=exp, zoom=zoom, variable='lh', year=y, lat_range=lat_range, lon_range=lon_range)
        sh = load_data_ICON(dataset=ds, experiment=exp, zoom=zoom, variable='sh', year=y, lat_range=lat_range, lon_range=lon_range)

        ef = compute_ef(lh, sh)


        if not os.path.isdir(outdir + '/' + str(y)):
            os.mkdir(outdir + '/' + str(y))
        outdir_y = outdir + '/' + str(y)

        outfile = outdir_y + '/' + str(y) + '.nc'
        ef.to_netcdf(outfile)


print('Done')

