"""Make humid heat extreme event identification"""
"""Contiguous area in a given range with Twb > absolute and relative thresholds for at least 3 days"""

import sys
import os
import argparse
import glob
import dill
import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
import pickle as pkl
import cc3d
import numba as nb

from config import CP4OUTPATH, datares
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean
from CP4.treat_max.a_make_dmax_proc_var import load_proc_var_dmax
from CP4.treat_max.b_compute_running_perc_dmax_proc_var import load_proc_var_dmax_roll_quantile


### CST ###


### FUNC

#~~~ /home/users/ljackson/h2x/python ~~~#

# timexlatxlon array which mainly has zero values -> [day index, lat index, lon index, HW value]
@nb.njit(fastmath=True)
def make_sparse_array(arr):
    w = np.where(arr>0)
    nrow = w[0].size
    sparse_arr = np.zeros((4,nrow),dtype='uint32')
    for i in range(nrow):
        sparse_arr[:,i] = [w[0][i], w[1][i], w[2][i], arr[w[0][i],w[1][i],w[2][i]]]
    return sparse_arr


@nb.njit(fastmath=True)
def calc_hw_duration_area(sparse_arr, N):
    hw_duration = np.zeros(N)
    hw_area = np.zeros(N)
    for i in range(1,N+1):
        w = sparse_arr[3,:]==i
        hw_duration[i-1] = np.unique(sparse_arr[0,w]).size
        hw_area[i-1] = np.sum(w)/hw_duration[i-1]
    return [hw_duration, hw_area]


# Remove small heatwaves from the dataset
@nb.njit(fastmath=True)
def remove_small_heatwaves(hhd, hw_sparse, hw_to_remove):
    for i in hw_to_remove:
        w = hw_sparse[3,:]==i
        w = hw_sparse[:3,w]
        for j in range(w.shape[1]):
            ix0, ix1, ix2 = w[:,j]
            hhd[ix0,ix1,ix2] = 0
    return hhd

#~~~ /home/users/ljackson/h2x/python ~~~#


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--variable", help='variable', type=str, default='twb')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--quantile_window", type=int, default=31)
    parser.add_argument("--q_thresh", type=float, default=0.95)
    parser.add_argument("--t_thresh", type=float, default=26.)
    parser.add_argument("--min_hw_size", type=float, default=100.)  # km2
    parser.add_argument("--max_hw_size", type=float, default=1000000.)  # km2
    parser.add_argument("--lat_range", nargs="+", type=float, default=[9., 18.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-8., 10.])
    parser.add_argument("--data_res", type=float, default=4.4)
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--connectivity", type=int, default=26)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var = opts.variable
    window = opts.window
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    qwindow = opts.quantile_window
    q_thresh = opts.q_thresh
    t_thresh = opts.t_thresh
    min_hw_size = opts.min_hw_size
    max_hw_size = opts.max_hw_size
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    data_res = opts.data_res
    n_days = opts.n_days
    cnty = opts.connectivity

    months_ = "-".join([str(m) for m in months])
    years = np.arange(y0, y1+1, 1)
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)

    res_ = str(res) + 'km'
    cnty_ = str(cnty)

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]

    areamin = int(np.ceil(min_hw_size / datares**2))


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/composites_' + var):
        os.mkdir(outdir + '/composites_' + var)
    outdir = outdir + '/composites_' + var

    if not os.path.isdir(outdir + '/hhe_tables'):
        os.mkdir(outdir + '/hhe_tables')
    outdir = outdir + '/hhe_tables'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/' + months_):
        os.mkdir(outdir + '/' + months_)
    outdir = outdir + '/' + months_

    if not os.path.isdir(outdir + '/' + space_scale):
        os.mkdir(outdir + '/' + space_scale)
    outdir = outdir + '/' + space_scale

    if not os.path.isdir(outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + '_cc3d=' + cnty_):
        os.mkdir(outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + '_cc3d=' + cnty_)
    outdir = outdir + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + '_cc3d=' + cnty_

    print(outdir)


    #~ Get data

    dmax = load_proc_var_dmax(ds, res, var, window, y0, y1, months, lat_range, lon_range)
    q_data = load_proc_var_dmax_roll_quantile(ds, res, var, q_thresh, qwindow, window, y0, y1, months, lat_range, lon_range)

    iyears = dmax.groupby('time.year').groups

    for y in years:
        print('\n0 {0}'.format(y))

        twbdata = load_proc_var_roll_mean(ds, res, var, y, window, months, lat_range, lon_range)  # for plot
        iyears_h = twbdata.groupby('time.year').groups
        twbdata_y = twbdata.isel(time=iyears_h[y])
        imonths_h = twbdata_y.groupby('time.month').groups

        dmax_y = dmax.isel(time=iyears[y])
        imonths = dmax_y.groupby('time.month').groups
        dmax_y = dmax_y.isel(time=slice(imonths[months[0]][0], imonths[months[-1]][-1]+1))

        dmax_y_ = dmax_y[int(qwindow/2):-int(qwindow/2)]

        hhd1 = np.zeros(dmax_y_.values.shape)
        hhd1[np.logical_and(dmax_y_.values > t_thresh, dmax_y_.values > q_data)] = 1


        # Get HWs

        hw1 = cc3d.connected_components(hhd1, connectivity=cnty)
        N1 = np.max(hw1)


        # Get duration and area of HWs

        hw_sparse1 = make_sparse_array(hw1)

        hw_duration1, hw_area1 = calc_hw_duration_area(hw_sparse1, N1)


        # Remove HWs below duration threshold

        print('\n>>> Removing HW shorter than {0} days <<<'.format(n_days))

        hw_to_remove_duration = (np.arange(N1)+1)[hw_duration1 < n_days]

        hhd2 = remove_small_heatwaves(hhd1, hw_sparse1, hw_to_remove_duration)


        # Get new HWs

        hw2 = cc3d.connected_components(hhd2, connectivity=cnty)
        N2 = np.max(hw2)

        hw_sparse2 = make_sparse_array(hw2)

        hw_duration2, hw_area2 = calc_hw_duration_area(hw_sparse2, N2)


        # Remove HWs below area threshold

       	print('\n>>> Removing HW smaller than {0} pixels <<<'.format(areamin))

        hw_to_remove_area = (np.arange(N2)+1)[hw_area2 < areamin]

        hhd3 = remove_small_heatwaves(hhd2, hw_sparse2, hw_to_remove_area)


       # Get new HWs

        hw3 = cc3d.connected_components(hhd3, connectivity=cnty)
        N3 = np.max(hw3)

        hw_sparse3 = make_sparse_array(hw3)

        hw_duration3, hw_area3 = calc_hw_duration_area(hw_sparse3, N3)


        # Cut out HW objects

        print('\n>>> Cutting out {0} HW objects <<<'.format(N3))


        for g in range(N3):
            g_ = g+1

            pos3d = np.broadcast_to(hw3==g_, dmax_y_.shape) # for 3d masking
            npos3d = np.broadcast_to(hw3!=g_, dmax_y_.shape)  # for 3d masking

            hw_obj = dmax_y_.copy()
            hw_obj.values[npos3d] = np.nan
            tmaxpos_3d = np.unravel_index(np.nanargmax(hw_obj.values), hw_obj.shape)  # -> time, lat, lon of max
            hw_obj_max = hw_obj[tmaxpos_3d[0]]

            tmax = np.nanmax(hw_obj_max)

            #pos2dmax  = np.unravel_index(np.nanargmax(hw_obj_max), hw_obj_max.shape)
            ilatmax, ilonmax = np.unravel_index(np.nanargmax(hw_obj_max), hw_obj_max.shape)

            latmax = hw_obj.latitude.values[ilatmax]  # pos2dmax[0]]
            lonmax = hw_obj.longitude.values[ilonmax]  # pos2dmax[1]]

            dur = hw_duration3[g]
            area = hw_area3[g] * (data_res**2)

            # Get date of the max Twbmax over the 3 consecutive days
            y = int(dmax_y_['time.year'][tmaxpos_3d[0]].values)
            m = int(dmax_y_['time.month'][tmaxpos_3d[0]].values)
            d = int(dmax_y_['time.day'][tmaxpos_3d[0]].values)

            twbdata_m = twbdata_y.isel(time=imonths_h[m])
            idays_h = twbdata_m.groupby('time.day').groups
       	    twbdata_d = twbdata_m.isel(time=idays_h[d])

            hmax = np.argmax(twbdata_d.isel(latitude=ilatmax, longitude=ilonmax).values)

            datestr = str(y) + '-' + str(m).zfill(2) + '-' + str(d).zfill(2)  #  + '-' + str(hmax).zfill(2)
            hw_id = datestr + '_' + str(g_)

            dic = {}

            dic['date'] = datestr
            dic['year'] = y
            dic['month'] = m
            dic['day'] = d
            dic['hmax'] = hmax
            dic['duration'] = dur
            dic['area'] = area
            dic['tmax'] = tmax
            dic['tmaxlat'] = latmax
            dic['tmaxlon'] = lonmax
            dic['HW_ID'] = hw_id  # some unique ID including date


            print('{0} ({1},{2}): {3} C | {4} km2'.format(datestr, latmax, lonmax, tmax, area))


            outfile_p = outdir + '/' + hw_id + ".p"

            pkl.dump(dic, open(outfile_p, "wb"))

            #sys.exit()



print('Done')

