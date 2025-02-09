"""Make single level variable composite fields"""

import sys
import os
import argparse
import dill
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import pytz

from datetime import datetime
from timezonefinder import TimezoneFinder

from config import CP4OUTPATH
from CP4.make_climato.a1_compute_rolling_mean_var import load_var_roll_mean
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean
from CP4.make_climato.c1_compute_running_hourly_climato_single_lvl_var import load_single_lvl_var_running_hclimato
from CP4.make_composites.make_hhee_data import get_files_hhee_data, load_hhee_data


### CST ###

tf = TimezoneFinder()
utc_zone = pytz.timezone('UTC')


### FUNC ###

def get_path_composite_var(ds='CP4A', res=4, var_ref='twb', var='t2', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, method='cc3d', connectivity=18, lat_range=(9., 18.), lon_range=(-8., 10.)):
    "Get path of variable composite fields files"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    res_ = str(res) + 'km'

    if method == 'cc3d':
        method_ = '_' + method + '=' + str(connectivity)
    else:
        method_ = method

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/composites_' + var_ref + '/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + method_ + '/' + years_ + '/' + months_

    return outdir


def load_composite_var(ds='CP4A', res=4, var_ref='twb', var='t2', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(9., 18.), lon_range=(-8., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable HHE composite"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def load_composite_var_clim(ds='CP4A', res=4, var_ref='twb', var='t2', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(9., 18.), lon_range=(-8., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable climato HHE composite"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_climato'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def load_composite_coords_hhee(ds='CP4A', res=4, var_ref='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(9., 18.), lon_range=(-8., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable composite HHE coordinates"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var_ref, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_coords'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def load_composite_dates_hhee(ds='CP4A', res=4, var_ref='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(4., 10.), lon_range=(-14., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable composite HHE dates"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var_ref, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_dates'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out

def load_composite_features_hhee(ds='CP4A', res=4, var_ref='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(9., 18.), lon_range=(-8., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable composite HHE features (duration, area, tmax)"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var_ref, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_features'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def load_composite_durations_hhee(ds='CP4A', res=4, var_ref='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(9., 18.), lon_range=(-8., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable composite HHE durations"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var_ref, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_durations'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def load_composite_ids_hhee(ds='CP4A', res=4, var_ref='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], lat_range=(9., 18.), lon_range=(-8., 10.), min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable composite HHE times"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_composite_var(ds, res, var_ref, var_ref, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_ids'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def get_path_composite_var_merge(ds='CP4A', res=4, var_ref='twb', var='t2', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, method='cc3d', connectivity=18, regions=['WSahel', 'ESahel']):
    "Get path of variable composite fields files"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])
    regs_ = "-".join(regions)
    res_ = str(res) + 'km'

    if method == 'cc3d':
        method_ = '_' + method + '=' + str(connectivity)
    else:
        method_ = method

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/composites_' + var_ref + '/' + var + '/' + regs_ + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + method_ + '/' + years_ + '/' + months_

    return outdir


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--var_ref", help='variable used to make HHE', type=str, default='twb')
    parser.add_argument("--variable", help='variable', type=str, default='q2')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--daily_window", type=int, default=11)
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
    parser.add_argument("--connectivity", type=int, default=26)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var_ref = opts.var_ref
    var = opts.variable
    window = opts.window
    dwindow = opts.daily_window
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

    time_arr = np.arange(-200, 200+1, 1)  # for time re-indexing

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


    #~ Get data

    print('-- Get data --')

    files = get_files_hhee_data(ds, res, var_ref, months, lat_range, lon_range, min_hw_size, max_hw_size, t_thresh, q_thresh, n_days, meth, cnty)
    files.sort()
    files = [f for f in files if not ('02-26' in f) and not ('02-27' in f) and not ('02-28' in f) and not ('02-29' in f) and not ('02-30' in f) and not ('03-01' in f) and not ('03-02' in f) and not ('03-03' in f) and not ('03-04' in f)]

    ids = load_composite_ids_hhee(ds, res, var_ref, y0, y1, months, t_thresh, q_thresh, n_days, window, sw, tw, lat_range, lon_range, min_hw_size, max_hw_size, meth, cnty)

    file = CP4OUTPATH + '/' + ds + '/' + res_ + '/composites_' + var_ref + '/hhe_tables/lat={0},{1}_lon={2},{3}'.format(lat_min,lat_max,lon_min,lon_max) + '/' + months_ + '/' + space_scale + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_

    files_ = []
    for f in files:
        for id in ids:
            if f == file + '/' + id + '.p':
                files_.append(f)

    nfs = len(files_)
    print('>>> {0} events <<<'.format(nfs))

    var_ref_data = load_proc_var_roll_mean(ds, res, var_ref, 2000, window, months, lat_range, lon_range)

    varhclim = load_single_lvl_var_running_hclimato(ds, res, var, y0, y1, window, months, lat_range, lon_range)

    idayc = (months[0]-1) * 30 + int(dwindow/2)  # index of the first day of climatology

    #if var == 't2':
    #    varhclim = varhclim - 273.15
    #if var == 'lsRain':
    #    varhclim = varhclim * 3600
    if (var == 'u10') or (var == 'v10'):
        print('Regrid %s climato' % var)
        varhclim = varhclim.interp(latitude=var_ref_data.latitude.values, longitude=var_ref_data.longitude.values)


    lats = varhclim.latitude.values
    lons = varhclim.longitude.values
    lat_cen = lats[int(len(lats)/2)] #np.round()
    lon_cen = lons[int(len(lons)/2)] #np.round()


    print('-- Treat data --')

    var_hhee_vals = []
    var_hhee_clims = []
    coords_hhee = []
    dates_hhee = []


    for f in files_:
        hhee = load_hhee_data(f)

        datestr = hhee['date']
        y = int(datestr[0:4])
        m = int(datestr[5:7])
        d = int(datestr[8:10])
        tmaxlat = hhee['tmaxlat']
        tmaxlon = hhee['tmaxlon']
        tmax = hhee['tmax']

        iday = (m-1)*30+d-1
        iday_ = iday - idayc

        data = load_var_roll_mean(ds, res, var, y, window, months, lat_range, lon_range)

        if (var == 'u10') or (var == 'v10'):
            print('Regrid %s data' % var)
            data = data.interp(latitude=var_ref_data.latitude.values, longitude=var_ref_data.longitude.values)

        boxshape = data.sel(latitude=slice(lat_cen-swlat/2, lat_cen+swlat/2), longitude=slice(lon_cen-swlon/2, lon_cen+swlon/2)).isel(time=0).shape   # get shape of a complete box

        xs = np.arange(0.5, boxshape[1], 1)
        ys = np.arange(0.5, boxshape[0], 1)

        imsdata = data.groupby('time.month').groups
        mdata = data.isel(time=imsdata[m])
        idsdata = mdata.groupby('time.day').groups
        ddata = mdata.isel(time=idsdata[d])

        idata = np.where(data.time.values == ddata.time[0].values)[0][0]

        data_hhee = data.sel(latitude=slice(tmaxlat-swlat/2, tmaxlat+swlat/2), longitude=slice(tmaxlon-swlon/2, tmaxlon+swlon/2)).isel(time=slice(idata+tw_before-24, idata+tw_after+24))

        if (data_hhee.isel(time=0).shape == boxshape) and (iday_+int(tw_before/24)-1 >= 0.) and (iday_+int(tw_after/24)+1 < varhclim.shape[0]) and (np.isnan(ddata.sel(latitude=tmaxlat, longitude=tmaxlon, method='nearest').values).all() == False):
            data_hhee_clim = varhclim.sel(latitude=slice(tmaxlat-swlat/2, tmaxlat+swlat/2), longitude=slice(tmaxlon-swlon/2, tmaxlon+swlon/2)).isel(days=slice(int(iday_+tw_before/24-1), int(iday_+tw_after/24+1)))

            assert (data_hhee_clim.time_level_0.values[int(tw[1]/24)+1] == m) and (data_hhee_clim.time_level_1.values[int(tw[1]/24)+1] == d)
            assert data_hhee_clim.isel(days=0, hour=0).shape == boxshape, 'Shape issue'

            # convert UTC to local time
            tz = tf.timezone_at(lng=tmaxlon, lat=tmaxlat)
            loc_zone = pytz.timezone(tz)
            times = data_hhee.time.to_numpy()
            dts = [t.strftime(t.format) for t in times]
            dts = [datetime.strptime(dt, t.format) for dt, t in zip(dts, times)]
            dts = [dt.replace(tzinfo=utc_zone) for dt in dts]
            dts_loc = [dt.astimezone(loc_zone) for dt in dts]
            dts_loc = pd.to_datetime(dts_loc)

            it0 = np.where(dts_loc.day == d)[0][0]
            its = time_arr[200-it0:200-it0+(-tw_before+tw_after+24*2)]
            its_ = its[it0+tw_before:it0+tw_after]  # reduce time window from +/- 4 days to +/- 3 days

            data_hhee = data_hhee.assign_coords(time=its, latitude=ys, longitude=xs)

            data_hhee_clim_ = data_hhee_clim.values.reshape(data_hhee.shape)
            data_hhee_clim_ = xr.DataArray(data=data_hhee_clim_, dims=['time', 'latitude', 'longitude'], coords=dict(time=(['time'], its), latitude=(['latitude'], ys), longitude=(['longitude'], xs)))

            data_hhee_ = data_hhee.sel(time=its_)
            data_hhee_clim_ = data_hhee_clim_.sel(time=its_)

            var_hhee_vals.append(data_hhee_.values)
            var_hhee_clims.append(data_hhee_clim_.values)

            print('\n  -- {0},{1} {2}: {3} --'.format(tmaxlat, tmaxlon, loc_zone, datestr))
            print(np.unravel_index(np.nanargmax(data_hhee_.values), data_hhee_.values.shape)[0], data_hhee_.max().values)

    var_hhee_vals = np.asarray(var_hhee_vals)
    var_hhee_clims = np.asarray(var_hhee_clims)


    print('\n-- Save --')

    outfile_vals = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_
    outfile_clim = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_climato'

    with open(outfile_vals, 'wb') as pics:
        pickle.dump(obj=var_hhee_vals, file=pics)

    with open(outfile_clim, 'wb') as pics:
        pickle.dump(obj=var_hhee_clims, file=pics)


print('Done')

