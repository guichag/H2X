"""Make variable composite fields for significance assessment"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import pytz

from datetime import datetime
from timezonefinder import TimezoneFinder

from config import CP4OUTPATH
from CP4.make_climato.a1_compute_rolling_mean_var import load_var_roll_mean
from CP4.make_climato.c1_compute_running_hourly_climato_single_lvl_var import load_single_lvl_var_running_hclimato
from CP4.make_composites.make_hhee_data import get_files_hhee_data, load_hhee_data
from CP4.make_composites.a1_make_var_field import get_path_composite_var, load_composite_dates_hhee, get_path_composite_var_merge, load_composite_ids_hhee


### CST ###

tf = TimezoneFinder()
utc_zone = pytz.timezone('UTC')


### FUNC ###

def get_path_significance(ds='CP4', res=4, var_ref='twb', var='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, method='cc3d', connectivity=18, lat_range=(9., 18.), lon_range=(-8., 10.)):
    "Get path of HHE events significance files"
    outdir = get_path_composite_var(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outdir = outdir + '/significance'

    return outdir


def load_composite_significance_fields(ds='CP4', res=4, var_ref='twb', var='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], lat_range=(9., 18.), lon_range=(-8., 10.), t_thresh=26., q_thresh=0.95, n_days=3, window=6, spatial_window=[4., 4.], time_window=[-72, 72], sampling_time=[20, 20], min_hw_size=100., max_hw_size=1000000., method='cc3d', connectivity=18):
    "Load variable HHE composite significance"
    space_scale = str(min_hw_size) + '-' + str(max_hw_size)
    sw_ = str(spatial_window[0]) + 'x' + str(spatial_window[1])
    tw_ = str(time_window[0]) + '_to_' + str(time_window[1])

    datapath = get_path_significance(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(time_window[0], time_window[1]) + '_mean_ano_field_' + str(sampling_time[0]) + 'h_' + str(sampling_time[1]) + 'h.nc'

    out = xr.open_dataarray(outfile)

    return out


def get_path_significance_merge(ds='CP4', res=4, var_ref='twb', var='twb', year0=1997, year1=2006, months=[1,2,3,4,5,6,7,8,9,10,11,12], t_thresh=26., q_thresh=0.95, n_days=3, method='cc3d', connectivity=18, regions=['WSahel', 'ESahel']):
    "Get path of HHE events significance files"
    outdir = get_path_composite_var_merge(ds, res, var_ref, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, regions)
    outdir = outdir + '/significance'

    return outdir


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
    parser.add_argument("--connectivity", help='connectivity', type=int, default=26)
    parser.add_argument("--sampling_time", nargs="+", type=int, default=[19, 19])

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
    samtime = opts.sampling_time

    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

    swlat = sw[0]
    swlon = sw[1]
    sw_ = str(swlat) + 'x' + str(swlon)

    space_scale = str(min_hw_size) + '-' + str(max_hw_size)

    tw_ = str(tw[0]) + '_to_' + str(tw[1])

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

    if var == 'lsRain':
        mul_fac = 3600  # kg / m2 / s -> mm / h
    else:
        mul_fac = 1


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

    print(outdir)


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

    varhclim = load_single_lvl_var_running_hclimato(ds, res, var, y0, y1, window, months, lat_range, lon_range)
    idayc = (months[0]-1) * 30 + int(dwindow/2)  # index of the first day of climatology

    lats = varhclim.latitude.values
    lons = varhclim.longitude.values
    lat_cen = lats[int(len(lats)/2)]
    lon_cen = lons[int(len(lons)/2)]

    boxshape = varhclim.sel(latitude=slice(lat_cen-swlat/2, lat_cen+swlat/2), longitude=slice(lon_cen-swlon/2, lon_cen+swlon/2)).isel(days=0, hour=0).shape   # get shape of a complete box

    xs = np.arange(0.5, boxshape[1], 1)
    ys = np.arange(0.5, boxshape[0], 1)


    print('-- Treat data --')

    ds_anos = []

    for f_ in files_:
        hhee = load_hhee_data(f_)

        datestr = hhee['date']
        y = int(datestr[0:4])
        m = int(datestr[5:7])
        d = int(datestr[8:10])
        tmaxlat = hhee['tmaxlat']
        tmaxlon = hhee['tmaxlon']
        tmax = hhee['tmax']

        tz = tf.timezone_at(lng=tmaxlon, lat=tmaxlat)
        loc_zone = pytz.timezone(tz)

        print('\n>>> {0} <<<'.format(datestr))

        iday = (m-1)*30+d-1
        iday_ = iday - idayc

        var_hhee_anos = []

        yrs_ = np.delete(years, np.where(years==y)[0][0])
        for y_ in yrs_:
            print(y_, end=' : ', flush=True)

            data = load_var_roll_mean(ds, res, var, y_, window, months, lat_range, lon_range)

            imsdata = data.groupby('time.month').groups
            mdata = data.isel(time=imsdata[m])
            idsdata = mdata.groupby('time.day').groups
            ddata = mdata.isel(time=idsdata[d])

            idata = np.where(data.time.values == ddata.time[0].values)[0][0]

            data_hhee = data.sel(latitude=slice(tmaxlat-swlat/2, tmaxlat+swlat/2), longitude=slice(tmaxlon-swlon/2, tmaxlon+swlon/2)).isel(time=slice(idata+tw[0]-24, idata+tw[1]+24)) * mul_fac
            data_hhee_clim = varhclim.sel(latitude=slice(tmaxlat-swlat/2, tmaxlat+swlat/2), longitude=slice(tmaxlon-swlon/2, tmaxlon+swlon/2)).isel(days=slice(int(iday_+tw[0]/24-1), int(iday_+tw[1]/24+1))) * mul_fac

            assert (data_hhee_clim.time_level_0.values[int(tw[1]/24)+1] == m) and (data_hhee_clim.time_level_1.values[int(tw[1]/24)+1] == d)

            times = data_hhee.time.to_numpy()
            dts = [t.strftime(t.format) for t in times]
            dts = [datetime.strptime(dt, t.format) for dt, t in zip(dts, times)]
            dts = [dt.replace(tzinfo=utc_zone) for dt in dts]
            dts_loc = [dt.astimezone(loc_zone) for dt in dts]
            dts_loc = pd.to_datetime(dts_loc)

            it0 = np.where(dts_loc.day == d)[0][0]
            its = time_arr[200-it0:200-it0+(-tw[0]+tw[1]+24*2)]
            its_ = its[it0+tw[0]:it0+tw[1]]  # reduce time window from +/- 4 days to +/- 3 days

            data_hhee_clim_ = data_hhee_clim.values.reshape(data_hhee.shape)  # 6*24 -> 144

            data_hhee = data_hhee.assign_coords(time=its)
            data_hhee_clim_ = xr.DataArray(data=data_hhee_clim_, dims=['time', 'latitude', 'longitude'], coords=dict(time=(['time'], its), latitude=(['latitude'], data_hhee_clim.latitude.values), longitude=(['longitude'], data_hhee_clim.longitude.values)))

            data_hhee_ = data_hhee.sel(time=its_)
            data_hhee_clim_ = data_hhee_clim_.sel(time=its_)

            varhclim_time = data_hhee_clim_.sel(time=slice(samtime[0], samtime[1]))
            vardata_time = data_hhee_.sel(time=slice(samtime[0], samtime[1]))

            if var == 'lsRain':
                varhclim_time = varhclim_time.mean(dim='time') * abs((samtime[1] - samtime[0]))
                vardata_time = vardata_time.mean(dim='time') * abs((samtime[1] - samtime[0]))
            else:
                varhclim_time = varhclim_time.mean(dim='time')
                vardata_time = vardata_time.mean(dim='time')

            var_hhee_ano = vardata_time - varhclim_time
            var_hhee_anos.append(var_hhee_ano)

            #print('\n  -- HHEE: {0},{1},{2} --'.format(tmaxlat, tmaxlon, datestr))

        var_hhee_anos = np.asarray(var_hhee_anos)
        ds_ano = xr.DataArray(data=var_hhee_anos, dims=['year', 'y', 'x'], coords=dict(year=yrs_, x=(['x'], xs), y=(['y'], ys)))
        ds_anos.append(ds_ano)

    out = xr.concat(ds_anos, dim='n')


    #~ Save

    print('\n-- Save --')

    outfile = outdir + '/' + space_scale + '_' + str(window) + 'h_' + sw_ + '_{0}-to-{1}'.format(tw[0], tw[1]) + '_mean_ano_field_' + str(samtime[0]) + 'h_' + str(samtime[1]) + 'h.nc'

    if not os.path.isfile(outfile):
        out.to_netcdf(outfile)
    else:
        print('File already exist!')


print('Done')

