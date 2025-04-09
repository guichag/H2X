"""Make Twb/HI composite fields"""

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

from config import *
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean
from CP4.make_climato.a3_compute_rolling_mean_multi_level_var import load_multi_level_var_roll_mean
from CP4.make_climato.c2_compute_running_hourly_climato_proc_var import load_proc_var_running_hclimato
from CP4.make_composites.make_hhee_data import get_files_hhee_data, load_hhee_data


### CST ###

tf = TimezoneFinder()
utc_zone = pytz.timezone('UTC')


### FUNC ###


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--var_ref", help='variable used to make HHE', type=str, default='twb')
    parser.add_argument("--variable", help='variable', type=str, default='twb')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--daily_window", type=int, default=11)
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

    files = get_files_hhee_data(ds, res, var_ref, months, lat_range, lon_range, min_hw_size, max_hw_size, t_thresh, q_thresh, n_days, meth, connectivity=cnty)
    files.sort()
    files = [f for f in files if not ('02-26' in f) and not ('02-27' in f) and not ('02-28' in f) and not ('02-29' in f) and not ('02-30' in f) and not ('03-01' in f) and not ('03-02' in f) and not ('03-03' in f) and not ('03-04' in f)]

    if not months == [1,2,3,4,5,6,7,8,9,10,11,12]:
        print('Remove dates')
        m0 = months[0]
        m1 = months[-1]
        files = [f for f in files if not ('%02d-01'%m0 in f) and not ('%02d-02'%m0 in f) and not ('%02d-03'%m0 in f) and not ('%02d-04'%m0 in f) and not ('%02d-05'%m0 in f)]

    varhclim = load_proc_var_running_hclimato(ds, res, var, window, y0, y1, months, lat_range, lon_range)


    idayc = (months[0]-1) * 30 + int(dwindow/2)  # index of the first day of climatology

    lats = varhclim.latitude.values
    lons = varhclim.longitude.values
    lat_cen = lats[int(len(lats)/2)]
    lon_cen = lons[int(len(lons)/2)]

    lsmask = xr.open_dataset(CP4landseamaskfile, decode_times=False)
    lsmask = lsmask.assign_coords(x=lsmask.x - 360)
    lsmask = lsmask.sel(surface=0, t=0).lsm

    vegfrac = xr.open_dataset(CP4vegfracfile, decode_times=False)
    vegfrac = vegfrac.assign_coords(x=vegfrac.x - 360)
    vegfrac = vegfrac.sel(pseudo=7, t=0, x=slice(lon_min, lon_max), y=slice(lat_min, lat_max)).field1391


    print('-- Treat data --')

    var_hhee_vals = []
    var_hhee_clims = []
    hhee_times = []
    clim_times = []
    coords_hhee = []
    dates_hhee = []
    durs_hhee = []
    areas_hhee = []
    tmax_hhee = []
    hmax_hhee = []
    ids_hhee = []

    for f in files:
        hhee = load_hhee_data(f)

        datestr = hhee['date']
        y = int(datestr[0:4])
        m = int(datestr[5:7])
        d = int(datestr[8:10])
        hmax = hhee['hmax']
        tmaxlat = hhee['tmaxlat']
        tmaxlon = hhee['tmaxlon']
        tmax = hhee['tmax']
        dur = hhee['duration']
        area = int(hhee['area'])
        id = hhee['HW_ID']

        if (vegfrac.sel(y=tmaxlat, x=tmaxlon, method='nearest') != np.nan) and (lsmask.sel(y=tmaxlat, x=tmaxlon, method='nearest') == 1.):  # globe.is_land(tmaxlat, tmaxlon):
            iday = (m-1)*30+d-1
            iday_ = iday - idayc

            data = load_proc_var_roll_mean(ds, res, var, y, window, months, lat_range, lon_range)
            datasm = load_multi_level_var_roll_mean(ds, res, 'SM', y, window, months, lat_range, lon_range, 0.05)

            boxshape = data.sel(latitude=slice(lat_cen-swlat/2, lat_cen+swlat/2), longitude=slice(lon_cen-swlon/2, lon_cen+swlon/2)).isel(time=0).shape   # get shape of a complete box

            xs = np.arange(0.5, boxshape[1], 1)
            ys = np.arange(0.5, boxshape[0], 1)

            imsdata = data.groupby('time.month').groups
            mdata = data.isel(time=imsdata[m])
            imsdatasm = datasm.groupby('time.month').groups
            mdatasm = datasm.isel(time=imsdatasm[m])
            idsdata = mdata.groupby('time.day').groups
            idsdatasm = mdatasm.groupby('time.day').groups
            ddata = mdata.isel(time=idsdata[d])
            ddatasm = mdatasm.isel(time=idsdatasm[d])

            idata = np.where(data.time.values == ddata.time[0].values)[0][0]

            data_hhee = data.sel(latitude=slice(tmaxlat-swlat/2, tmaxlat+swlat/2), longitude=slice(tmaxlon-swlon/2, tmaxlon+swlon/2)).isel(time=slice(idata+tw_before-24, idata+tw_after+24))

            if (data_hhee.isel(time=0).shape == boxshape) and (iday_+int(tw_before/24)-1 >= 0.) and (iday_+int(tw_after/24)+1 < varhclim.shape[0]) and (np.isnan(ddatasm.sel(latitude=tmaxlat, longitude=tmaxlon, method='nearest').values).all() == False):
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

                # get local time of Twb max
                date = datestr + '-' + str(hmax).zfill(2)
                dmax = datetime.strptime(date, '%Y-%m-%d-%H'.format(y, m, d, hmax))
                dmax = dmax.replace(tzinfo=utc_zone)
                dmax_loc = dmax.astimezone(loc_zone)
                hmax_loc = dmax_loc.hour
                datestr_loc = dmax_loc.strftime('%Y-%m-%d')
                offset = dmax_loc.utcoffset().seconds /3600

                data_hhee = data_hhee.assign_coords(time=its, latitude=ys, longitude=xs)

                data_hhee_clim_ = data_hhee_clim.values.reshape(data_hhee.shape)
                data_hhee_clim_ = xr.DataArray(data=data_hhee_clim_, dims=['time', 'latitude', 'longitude'], coords=dict(time=(['time'], its), latitude=(['latitude'], ys), longitude=(['longitude'], xs)))

                data_hhee_ = data_hhee.sel(time=its_)
                data_hhee_clim_ = data_hhee_clim_.sel(time=its_)

                var_hhee_vals.append(data_hhee_.values)
                var_hhee_clims.append(data_hhee_clim_.values)
                coords_hhee.append((tmaxlat, tmaxlon))
                dates_hhee.append(datestr_loc)
                durs_hhee.append(dur)
                areas_hhee.append(area)
                tmax_hhee.append(tmax)
                hmax_hhee.append(hmax_loc)
                ids_hhee.append(id)

                print('\n  -- {0},{1} ({2}:+{3}) {4}: {5}, {6}km2 --'.format(tmaxlat, tmaxlon, loc_zone, offset, datestr_loc, round(tmax, 1), area))


    var_hhee_vals = np.asarray(var_hhee_vals)
    var_hhee_clims = np.asarray(var_hhee_clims)


    #~ Sample daily maxima

    features = pd.DataFrame({'coords': coords_hhee, 'date': dates_hhee, 'duration': durs_hhee, 'area': areas_hhee, 'tmax': tmax_hhee, 'hmax': hmax_hhee, 'id': ids_hhee})

    new_dates = list(set(dates_hhee))
    new_dates.sort()

    idxs = []

    for d in new_dates:
        features_ = features[features['date']==d]
        idxs.append(features_.idxmax()['tmax'])

    features = features.iloc[idxs]

    var_hhee_vals = var_hhee_vals[idxs]
    var_hhee_clims = var_hhee_clims[idxs]

    coords_hhee = list(features['coords'].values)
    dates_hhee = list(features['date'].values)
    ids_hhee = list(features['id'].values)


    #sys.exit()

    #~ Save

    print('\n-- Save --')

    outfile_vals = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_
    outfile_clim = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_climato'
    outfile_features_hhee = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_  + '_features'
    outfile_coords_hhee = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_ + '_coords'
    outfile_dates_hhee = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_  + '_dates'
    outfile_ids_hhee = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_' + tw_  + '_ids'


    with open(outfile_vals, 'wb') as pics:
        pickle.dump(obj=var_hhee_vals, file=pics)

    with open(outfile_clim, 'wb') as pics:
        pickle.dump(obj=var_hhee_clims, file=pics)

    with open(outfile_coords_hhee, 'wb') as pics:
        pickle.dump(obj=coords_hhee, file=pics)

    with open(outfile_dates_hhee, 'wb') as pics:
        pickle.dump(obj=dates_hhee, file=pics)

    with open(outfile_ids_hhee, 'wb') as pics:
        pickle.dump(obj=ids_hhee, file=pics)

    features.to_pickle(outfile_features_hhee)


print('Done')

