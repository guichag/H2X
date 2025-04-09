"""Make variable anomaly field over time window"""

import sys
import os
import argparse
import numpy as np
import numpy.ma as ma
import xarray as xr

from config import *
from CP4.make_climato.a6bis_compute_rolling_mean_theta_wb_pl_var import lvls
from CP4.make_composites.a4_make_cross_section_pl_var_vertical_profile import get_path_composite_pl_var, load_composite_cross_section_pl_var, load_composite_cross_section_pl_var_clim


### CST ###

levels = lvls
mean_iaxis = {'latitude': 4, 'longitude': 3}  # long cross-section -> mean across latitude (3)


### FUNC ###

def load_composite_mean_ano_pl(ds='CP4A', res=4, var='t2', year0=1997, year1=2006, months=[3, 4, 5], lat_range=(4., 10.), lon_range=(-14., 10.), t_thresh=24., q_thresh=0.95, n_days=3, window=6, lat_window=(-0.1,0.1), lon_window=(-2.0,2.0), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=26, levels=None):
    """Load variable composite-mean anomaly field"""
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    latwmin = lat_window[0]
    latwmax = lat_window[1]
    lonwmin = lon_window[0]
    lonwmax = lon_window[1]
    latw_ = str(latwmin) + '-' + str(latwmax)
    lonw_ = str(lonwmin) + '-' + str(lonwmax)

    if levels == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in levels])

    datapath = get_path_composite_pl_var(ds, res, var, year0, year1, months, t_thresh, q_thresh, n_days, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_mean_ano_pl.nc'

    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--var", help='variable', type=str, default='t_pl')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])
    parser.add_argument("--q_thresh", type=float, default=0.95)
    parser.add_argument("--t_thresh", type=float, default=26.)
    parser.add_argument("--min_hw_size", type=float, default=100.)   # km2
    parser.add_argument("--max_hw_size", type=float, default=1000000.)   # km2
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--method", help='method', type=str, default='cc3d')
    parser.add_argument("--connectivity", help='connectivity', type=int, default=26)
    parser.add_argument("--lat_window", nargs="+", type=float, default=[-0.1, 0.1])
    parser.add_argument("--lon_window", nargs="+", type=float, default=[-2., 2.])
    parser.add_argument("--cross_section_axis", type=str, default='longitude')  # cross section direction
    parser.add_argument("--levels", nargs="+", type=float, default=None)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.var
    window = opts.window
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    q_thresh = opts.q_thresh
    t_thresh = opts.t_thresh
    min_hw_size = opts.min_hw_size
    max_hw_size = opts.max_hw_size
    n_days = opts.n_days
    meth = opts.method
    cnty = opts.connectivity
    latw = opts.lat_window
    lonw = opts.lon_window
    cross_section_axis = opts.cross_section_axis
    lvls = opts.levels

    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    latwmin = latw[0]
    latwmax = latw[1]
    lonwmin = lonw[0]
    lonwmax = lonw[1]
    latw_ = str(latwmin) + '-' + str(latwmax)
    lonw_ = str(lonwmin) + '-' + str(lonwmax)

    if lvls == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in levels])

    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    space_scale_ = str(int(min_hw_size)) + '-' + str(int(max_hw_size))

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    assert lon_min < lon_max, "incorrect longitude range"

    if meth == 'cc3d':
        meth_ = '_' + meth + '=' + str(cnty)
    else:
        meth_ = meth


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/' + ds):
        os.mkdir(outdir + '/' + ds)
    datadir = outdir + '/' + ds

    if not os.path.isdir(datadir + '/composites'):
        os.mkdir(datadir + '/composites')
    datadir = datadir + '/composites'

    if not os.path.isdir(datadir + '/' + var):
        os.mkdir(datadir + '/' + var)
    datadir = datadir + '/' + var

    if not os.path.isdir(datadir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(datadir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    datadir = datadir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(datadir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_):
        os.mkdir(datadir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_)
    datadir = datadir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_

    if not os.path.isdir(datadir + '/' + years_):
        os.mkdir(datadir + '/' + years_)
    datadir = datadir + '/' + years_

    if not os.path.isdir(datadir + '/' + months_):
        os.mkdir(datadir + '/' + months_)
    datadir = datadir + '/' + months_


    #~ Get data

    print('-- Get data --')

    varhclim = load_composite_cross_section_pl_var_clim(ds, res, var, y0, y1, months, t_thresh, q_thresh, n_days, window, latw, lonw, lat_range, lon_range, min_hw_size, max_hw_size, meth, cnty)
    vardata = load_composite_cross_section_pl_var(ds, res, var, y0, y1, months, t_thresh, q_thresh, n_days, window, latw, lonw, lat_range, lon_range, min_hw_size, max_hw_size, meth, cnty)


    print('-- Treat data --')

    iaxis = mean_iaxis[cross_section_axis]

    varhclim_ = np.nanmean(varhclim, axis=iaxis)
    vardata_ = np.nanmean(vardata, axis=iaxis)

    var_hhee_ano = vardata_ - varhclim_

    ds_ano = xr.DataArray(data=var_hhee_ano, dims=['n', 'time', 'pressure', 'x'], coords=dict(n=range(vardata.shape[0]), time=(['time'], np.arange(0, 21+3, 3)), pressure=(['pressure'], levels), x=(['x'], np.arange(0.5, var_hhee_ano.shape[3], 1))))


    #~ Save

    print('\n-- Save --')

    datafile = datadir + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_mean_ano_pl.nc'

    if not os.path.isfile(datafile):
        ds_ano.to_netcdf(datafile)
    else:
        print('File already exists!')


print('Done')

