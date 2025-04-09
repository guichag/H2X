"""Compute pressure levels variable composite significance"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from scipy.stats import mannwhitneyu

from config import CP4OUTPATH
from CP4.make_climato.a6bis_compute_rolling_mean_theta_wb_pl_var import lvls
from CP4.make_composites.b4_make_pl_var_anomaly import load_composite_mean_ano_pl
from CP4.make_composites.significance.a_make_var_field_significance import get_path_significance
from CP4.make_composites.significance.a4_make_pl_var_significance import load_composite_significance_pl


### CST ###

levels = lvls


### FUNC ###

def load_composite_significance_pvalues_pl(ds='CP4A', res=4, var='twb', year0=1997, year1=2006, months=[3, 4, 5], lat_range=(4., 10.), lon_range=(-14., 10.), t_thresh=24., q_thresh=0.95, n_days=3, window=6, lat_window=(-0.1,0.1), lon_window=(-2.0,2.0), min_hw_size=100., max_hw_size=1000000., levels=None):
    "Load variable HHE composite pvalues"
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

    datapath = get_path_significance(ds, res, var, year0, year1, months, t_thresh, q_thresh, n_days, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_pvalues.nc'

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
    parser.add_argument("--min_hw_size", type=float, default=100.)  # km2
    parser.add_argument("--max_hw_size", type=float, default=1000000.)  # km2
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--lat_window", nargs="+", type=float, default=[-0.1, 0.1])
    parser.add_argument("--lon_window", nargs="+", type=float, default=[-2., 2.])
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
    latw = opts.lat_window
    lonw = opts.lon_window
    lvls = opts.levels

    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    latwmin = latw[0]
    latwmax = latw[1]
    assert latwmin < latwmax, "incorrect latitude window range"
    lonwmin = lonw[0]
    lonwmax = lonw[1]
    assert lonwmin < lonwmax, "incorrect longitude window range"
    latw_ = str(latwmin) + '-' + str(latwmax)
    lonw_ = str(lonwmin) + '-' + str(lonwmax)

    space_scale = str(min_hw_size) + '-' + str(max_hw_size)

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    assert lon_min < lon_max, "incorrect longitude range"

    if lvls == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in lvls])


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/composites'):
        os.mkdir(outdir + '/composites')
    outdir = outdir + '/composites'

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days)):
        os.mkdir(outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days))
    outdir = outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days)

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

    anos_hhee = load_composite_mean_ano_pl(ds, var, y0, y1, months, lat_range, lon_range, t_thresh, q_thresh, n_days, window, latw, lonw, min_hw_size, max_hw_size, lvls)
    anos_all = load_composite_significance_pl(ds, var, y0, y1, months, lat_range, lon_range, t_thresh, q_thresh, n_days, window, latw, lonw, min_hw_size, max_hw_size, lvls)

    n_hhee = len(anos_hhee.n)
    print('>>> {0} events <<<'.format(n_hhee))

    n = n_hhee * len(years)


    print('-- Treat data --')

    anos_all_ = anos_all.values.reshape(n, anos_all.shape[2], anos_all.shape[3], anos_all.shape[4])

    mwtest = mannwhitneyu(anos_hhee, anos_all_, axis=0, nan_policy='omit')

    mwpval = mwtest[1]

    print('Mean pvalue: %.5f'% np.nanmean(mwpval))

    ds_pvals = xr.DataArray(data=mwpval, dims=['time', 'pressure', 'x'], coords=dict(time=(['time'], np.arange(0, 21+3, 3)), pressure=(['pressure'], levels), x=(['x'], anos_hhee.x.values)))


    #~ Save

    print('-- Save --')

    outfile = outdir + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_pvalues.nc'

    if not os.path.isfile(outfile):
        ds_pvals.to_netcdf(outfile)
    else:
        print('File already exist!')


print('Done')


