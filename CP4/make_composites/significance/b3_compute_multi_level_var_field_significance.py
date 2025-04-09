"""Compute variable composite fields significance"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from scipy.stats import mannwhitneyu

from config import CP4OUTPATH
from CP4.make_composites.b3_make_multi_level_var_field_anomaly import load_multi_lvl_composite_mean_ano_field
from CP4.make_composites.significance.a_make_var_field_significance import get_path_significance
from CP4.make_composites.significance.a3_make_multi_level_var_field_significance import load_multi_level_composite_significance_fields


### CST ###


### FUNC ###

def load_multi_level_composite_significance_pvalues(ds='CP4', res=4, var_ref='twb', var='SM', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], lat_range=(9., 18.), lon_range=(-8., 10.), t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], sampling_time=[0, 6], min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18, level=0.05):
    "Load variable HHE composite pvalues"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])
    res_ = str(res) + 'km'

    datapath = get_path_significance(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(time_window[0], time_window[1]) + '_pvalues_' + str(sampling_time[0]) + 'h_' + str(sampling_time[1]) + 'h_level=' + str(level) + '.nc'

    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--var_ref", help='variable used to make HHE', type=str, default='twb')
    parser.add_argument("--variable", help='variable', type=str, default='SM')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])
    parser.add_argument("--q_thresh", type=float, default=0.95)
    parser.add_argument("--t_thresh", type=float, default=26.)
    parser.add_argument("--min_hw_size", type=float, default=100.)  # km2
    parser.add_argument("--max_hw_size", type=float, default=1000000.)  # km2
    parser.add_argument("--spatial_window", nargs="+", type=float, default=[4., 4.])
    parser.add_argument("--time_window", nargs="+", type=int, default=[-72, 72])
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--method", help='method', type=str, default='cc3d')
    parser.add_argument("--connectivity", help='connectivity', type=int, default=26)
    parser.add_argument("--sampling_time", nargs="+", type=int, default=[0, 6])
    parser.add_argument("--level", type=float, default=0.05)

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
    lvl = opts.level

    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    swlat = sw[0]
    swlon = sw[1]
    sw_ = str(swlat) + 'x' + str(swlon)

    space_scale = str(min_hw_size) + '-' + str(max_hw_size)

    tw_before = tw[0]
    tw_after = tw[1]
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

    if not os.path.isdir(outdir + '/significance'):
        os.mkdir(outdir + '/significance')
    outdir = outdir + '/significance'


    #~ Get data

    print('-- Get data --')

    anos_hhee = load_multi_lvl_composite_mean_ano_field(ds, res, var_ref, var, y0, y1, months, lat_range, lon_range, t_thresh, q_thresh, n_days, window, sw, tw, samtime, min_hw_size, max_hw_size, meth, cnty, lvl)

    anos_all = load_multi_level_composite_significance_fields(ds, res, var_ref, var, y0, y1, months, lat_range, lon_range, t_thresh, q_thresh, n_days, window, sw, tw, samtime, min_hw_size, max_hw_size, meth, cnty, lvl)

    n_hhee = len(anos_hhee.n)
    print('>>> {0} events <<<'.format(n_hhee))

    n = n_hhee * len(years)

    ano_mean = np.nanmean(anos_hhee.values, axis=0) # mean across events

    print('\n{0} min: {1} / max: {2}'.format(var, ano_mean.min(), ano_mean.max()))


    print('-- Treat data --')

    anos_all_ = anos_all.values.reshape(n, anos_all.shape[2], anos_all.shape[3])

    mwtest = mannwhitneyu(anos_hhee, anos_all_, axis=0, nan_policy='omit')

    mwpval = mwtest[1]

    print('Mean pvalue: %.5f'% np.nanmean(mwpval))

    ds_pvals = xr.DataArray(data=mwpval, dims=['y', 'x'], coords=dict(x=(['x'], anos_hhee.x.values), y=(['y'], anos_hhee.y.values)))


    #~ Save

    print('-- Save --')

    outfile = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(tw[0], tw[1]) + '_pvalues_' + str(samtime[0]) + 'h_' + str(samtime[1]) + 'h_level=' + str(lvl) + '.nc'

    if not os.path.isfile(outfile):
        ds_pvals.to_netcdf(outfile)
    else:
        print('File already exist!')


print('Done')

