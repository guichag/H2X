"""Compute variable composite fields significance"""
"""Merge regions"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from scipy.stats import mannwhitneyu

from config import CP4OUTPATH
from CP4.make_composites.b1_make_var_field_anomaly import load_composite_mean_ano_field
from CP4.make_composites.significance.a_make_var_field_significance import get_path_significance_merge, load_composite_significance_fields
from CP4.plots.p_config import study_regions


### CST ###


### FUNC ###

def load_composite_significance_pvalues_merge(ds='CP4A', res=4, var_ref='twb', var='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], regions=['WSahel', 'CSahel', 'ESahel'], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], sampling_time=[-20, 20], min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable HHE composite pvalues"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])
    res_ = str(res) + 'km'

    datapath = get_path_significance_merge(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, regions)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(time_window[0], time_window[1]) + '_pvalues_' + str(sampling_time[0]) + 'h_' + str(sampling_time[1]) + 'h.nc'

    out = xr.open_dataarray(outfile)

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
    parser.add_argument("--q_thresh", type=float, default=0.95)
    parser.add_argument("--t_thresh", type=float, default=26.)
    parser.add_argument("--min_hw_size", type=float, default=100.)  # km2
    parser.add_argument("--max_hw_size", type=float, default=1000000.)  # km2
    parser.add_argument("--spatial_window", nargs="+", type=float, default=[4., 4.])
    parser.add_argument("--time_window", nargs="+", type=int, default=[-72, 72])
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--method", help='method', type=str, default='cc3d')
    parser.add_argument("--connectivity", help='connectivity', type=int, default=26)
    parser.add_argument("--sampling_time", nargs="+", type=int, default=[19, 19])
    parser.add_argument("--regions", nargs="+", type=str, default=['WSahel', 'CSahel', 'ESahel', 'Tchad'])

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var_ref = opts.var_ref
    var = opts.variable
    window = opts.window
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
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
    regs = opts.regions

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

    regs_ = "-".join(regs)

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

    if not os.path.isdir(outdir + '/' + regs_):
        os.mkdir(outdir + '/' + regs_)
    outdir = outdir + '/' + regs_

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

    out_ano_hhee = []
    out_ano_all = []

    for reg in regs:
        coords = study_regions[reg]
        lat_range = coords[0]
        lon_range = coords[1]

        anos_hhee = load_composite_mean_ano_field(ds, res, var_ref, var, y0, y1, months, lat_range, lon_range, t_thresh, q_thresh, n_days, window, sw, tw, samtime, min_hw_size, max_hw_size, meth, cnty)

        anos_all = load_composite_significance_fields(ds, res, var_ref, var, y0, y1, months, lat_range, lon_range, t_thresh, q_thresh, n_days, window, sw, tw, samtime, min_hw_size, max_hw_size, meth, cnty)

        out_ano_hhee.append(anos_hhee)
        out_ano_all.append(anos_all)

        print('\n>>> {0} (N={1}) <<<'.format(reg, anos_hhee.shape[0]))

    out_ano_hhee = xr.concat(out_ano_hhee, dim='n')
    out_ano_all = xr.concat(out_ano_all, dim='n')

    n_hhee = len(out_ano_hhee.n)
    print('\n>>> {0} events <<<'.format(n_hhee))

    n = n_hhee * len(years)


    out_ano_mean = np.nanmean(out_ano_hhee.values, axis=0) # mean across events

    print('\n{0} min: {1} / max: {2}'.format(var, out_ano_mean.min(), out_ano_mean.max()))


    print('-- Treat data --')

    out_ano_all_ = out_ano_all.values.reshape(n, anos_all.shape[2], anos_all.shape[3])

    mwtest = mannwhitneyu(out_ano_hhee, out_ano_all_, axis=0, nan_policy='omit')

    mwpval = mwtest[1]

    print('Mean pvalue: %.5f'% np.nanmean(mwpval))

    ds_pvals = xr.DataArray(data=mwpval, dims=['y', 'x'], coords=dict(x=(['x'], anos_hhee.x.values), y=(['y'], anos_hhee.y.values)))


    #~ Save

    print('-- Save --')

    outfile = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(tw[0], tw[1]) + '_pvalues_' + str(samtime[0]) + 'h_' + str(samtime[1]) + 'h.nc'

    if not os.path.isfile(outfile):
        ds_pvals.to_netcdf(outfile)
    else:
        print('File already exist!')


print('Done')

