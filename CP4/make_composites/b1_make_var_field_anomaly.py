"""Make variable anomaly field over time window"""

import sys
import os
import argparse
import numpy as np
import numpy.ma as ma
import xarray as xr

from config import *
from CP4.make_composites.a1_make_var_field import get_path_composite_var, load_composite_var, load_composite_var_clim


### CST ###


### FUNC ###

def load_composite_mean_ano_field(ds='CP4', res=4, var_ref='twb', var='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], lat_range=(9., 18.), lon_range=(-8., 10.), t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=(-72, 72), sampling_time=[20, 20], min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    """Load variable composite-mean anomaly field"""
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '-to-' + str(time_window[1])
    res_ = str(res) + 'km'

    datapath = get_path_composite_var(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    datafile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_mean_ano_field_' + str(sampling_time[0]) + 'h_' + str(sampling_time[1]) + 'h.nc'

    out = xr.open_dataarray(datafile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--var_ref", help='variable used to make HHE', type=str, default='twb')
    parser.add_argument("--variable", help='variable', type=str, default='twb')
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
    parser.add_argument("--spatial_window", nargs="+", type=float, default=[4., 4.])
    parser.add_argument("--time_window", nargs="+", type=int, default=[-72, 72])
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--method", help='method', type=str, default='cc3d')
    parser.add_argument("--connectivity", help='connectivity', type=int, default=26)
    parser.add_argument("--sampling_time", nargs="+", type=int, default=[19, 19])

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var_ref = opts.var_ref
    var = opts.variable
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
    sw = opts.spatial_window
    tw = opts.time_window
    n_days = opts.n_days
    meth = opts.method
    cnty = opts.connectivity
    samtime = opts.sampling_time

    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    swlat = sw[0]
    swlon = sw[1]
    sw_ = str(swlat) + 'x' + str(swlon)

    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    space_scale_ = str(int(min_hw_size)) + '-' + str(int(max_hw_size))

    tw_before = tw[0]
    tw_after = tw[1]
    assert tw_before <= tw_after, "Incorrect number of time steps"
    tw_ = str(tw_before) + '_to_' + str(tw_after)

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

    if var == 'lsRain':
        mul_fac = 3600  # kg / m2 / s -> mm / h
    else:
        mul_fac = 1

    print('\n !!! SAMPLING TIME: {0}h - {1}h !!!'.format(samtime[0], samtime[1]))


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/composites_' + var_ref):
        os.mkdir(outdir + '/composites_' + var_ref)
    outdir = outdir + '/composites_' + var_ref

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_):
        os.mkdir(outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_)
    outdir = outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_

    if not os.path.isdir(outdir + '/' + years_):
        os.mkdir(outdir + '/' + years_)
    outdir = outdir + '/' + years_

    if not os.path.isdir(outdir + '/' + months_):
        os.mkdir(outdir + '/' + months_)
    outdir = outdir + '/' + months_


    #~ Get data

    print('-- Get data --')

    varhclim = load_composite_var_clim(ds, res, var_ref, var, y0, y1, months, t_thresh, q_thresh, n_days, window, sw, tw, lat_range, lon_range, min_hw_size, max_hw_size, meth, cnty) * mul_fac
    vardata = load_composite_var(ds, res, var_ref, var, y0, y1, months, t_thresh, q_thresh, n_days, window, sw, tw, lat_range, lon_range, min_hw_size, max_hw_size, meth, cnty) * mul_fac

    n_hhee = vardata.shape[0]
    time = np.arange(tw[0], tw[1], 1)
    xs = np.arange(0.5, vardata.shape[3], 1)
    ys = np.arange(0.5, vardata.shape[2], 1)

    varhclim_all = xr.DataArray(data=varhclim, dims=['n', 'time', 'y', 'x'], coords=dict(n=(range(n_hhee)), time=(['time'], time), x=(['x'], xs), y=(['y'], ys)))
    vardata_all = xr.DataArray(data=vardata, dims=['n', 'time', 'y', 'x'], coords=dict(n=(range(n_hhee)), time=(['time'], time), x=(['x'], xs), y=(['y'], ys)))


    print('-- Treat data --')

    varhclim_time = varhclim_all.sel(time=slice(samtime[0], samtime[1]))
    vardata_time = vardata_all.sel(time=slice(samtime[0], samtime[1]))

    if var == 'lsRain':
        varhclim_time = varhclim_time.mean(dim='time') * abs((samtime[1] - samtime[0]))  # get cumulative val
        vardata_time = vardata_time.mean(dim='time') * abs((samtime[1] - samtime[0]))  # get cumulative val
    else:
        varhclim_time = varhclim_time.mean(dim='time')
        vardata_time = vardata_time.mean(dim='time')

    var_hhee_ano = vardata_time - varhclim_time

    ds_ano = xr.DataArray(data=var_hhee_ano, dims=['n', 'y', 'x'], coords=dict(n=range(n_hhee), x=(['x'], xs), y=(['y'], ys)))

    ds_ano_mean = np.nanmean(ds_ano.values, axis=0) # mean across events

    print('\n{0} min: {1} / max: {2}'.format(var, ds_ano_mean.min(), ds_ano_mean.max()))


    #~ Save

    print('\n-- Save --')

    datafile = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(tw[0], tw[1]) + '_mean_ano_field_' + str(samtime[0]) + 'h_' + str(samtime[1]) + 'h.nc'

    if not os.path.isfile(datafile):
        ds_ano.to_netcdf(datafile)
    else:
        print('File already exists!')


print('Done')

