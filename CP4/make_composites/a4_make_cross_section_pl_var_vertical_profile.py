"""Make pressure level variable composite vertical cross-section"""

import sys
import os
import argparse
import dill
import pickle
import numpy as np

from config import CP4OUTPATH
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean
from CP4.make_climato.a4_compute_rolling_mean_pl_var import load_pl_var_roll_mean
from CP4.make_climato.c4_compute_running_hourly_climato_pl_var import load_pl_var_running_hclimato
from CP4.make_composites.make_hhee_data import get_files_hhee_data, load_hhee_data
from CP4.make_composites.a1_make_var_field import load_composite_dates_hhee


### CST ###


### FUNC ###

def get_path_composite_pl_var(ds='CP4A', res=4, var='t_pl', year0=1997, year1=2006, months=[3, 4, 5], t_thresh=24., q_thresh=0.95, n_days=3, method='cc3d', connectivity=26, lat_range=(4., 10.), lon_range=(-14., 10.)):
    "Get path of variable composite fields files"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    res_ = str(res) + 'km'

    if meth == 'cc3d':
        meth_ = '_' + meth + '=' + str(cnty)
    else:
        meth_ = meth

    outdir = CP4OUTPATH + '/' + ds + '/' + res_ + '/composites/' + var + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/twb_thres=' + str(t_thresh) + '_q_thres=' + str(q_thresh) + '_n_days=' + str(n_days) + meth_ + '/' + years_ + '/' + months_

    return outdir


def load_composite_cross_section_pl_var(ds='CP4A', res=4, var='t_pl', year0=1997, year1=2006, months=[3, 4, 5], t_thresh=24., q_thresh=0.95, n_days=3, window=6, lat_window=[-0.1, 0.1], lon_window=[-1.0, 1.0], lat_range=(4., 10.), lon_range=(-14., 10.), min_hw_size=1000., max_hw_size=5000., method='cc3d', connectivity=26, levels=None):
    "Load variable HHE composite"
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

    datapath = get_path_composite_pl_var(ds, res, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_)

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


def load_composite_cross_section_pl_var_clim(ds='CP4A', res=4, var='t_pl', year0=1997, year1=2006, months=[3, 4, 5], t_thresh=24., q_thresh=0.95, n_days=3, window=6, lat_window=[-0.1, 0.1], lon_window=[-1.0, 1.0], lat_range=(4., 10.), lon_range=(-14., 10.), min_hw_size=1000., max_hw_size=5000., method='cc3d', connectivity=26, levels=None):
    "Load variable climato HHE composite"
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

    datapath = get_path_composite_pl_var(ds, res, var, year0, year1, months, t_thresh, q_thresh, n_days, method, connectivity, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_climato'

    with open(outfile, 'rb') as pics:
        out = dill.load(pics)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--var_ref", help='variable used to make HHE', type=str, default='twb')
    parser.add_argument("--var", help='variable', type=str, default='t_pl')
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--daily_window", type=int, default=11)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[5., 10.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-18., 10.])
    parser.add_argument("--q_thresh", type=float, default=0.95)
    parser.add_argument("--t_thresh", type=float, default=26.)
    parser.add_argument("--min_hw_size", type=float, default=100.)  # km2
    parser.add_argument("--max_hw_size", type=float, default=1000000.)  # km2
    parser.add_argument("--spatial_window", nargs="+", type=float, default=[4., 4.])
    parser.add_argument("--time_window", nargs="+", type=int, default=[-72, 72])
    parser.add_argument("--lat_window", nargs="+", type=float, default=[-0.1, 0.1])
    parser.add_argument("--lon_window", nargs="+", type=float, default=[-2., 2.])
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--method", help='method', type=str, default='cc3d')
    parser.add_argument("--connectivity", help='connectivity', type=int, default=26)
    parser.add_argument("--levels", nargs="+", type=float, default=None)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    var_ref = opts.var_ref
    var = opts.var
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
    latw = opts.lat_window
    lonw = opts.lon_window
    n_days = opts.n_days
    meth = opts.method
    cnty = opts.connectivity
    lvls = opts.levels

    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])

    res_ = str(res) + 'km'

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

    if meth == 'cc3d':
        meth_ = '_' + meth + '=' + str(cnty)
    else:
        meth_ = meth

    if lvls == None:
        lvls_ = 'all'
    else:
        lvls_ = '-'.join([str(int(l)) for l in lvls])


    #~ Outdir

    if not os.path.isdir(CP4OUTPATH + '/' + ds):
        os.mkdir(CP4OUTPATH + '/' + ds)
    outdir = CP4OUTPATH + '/' + ds

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/' + ds):
        os.mkdir(outdir + '/' + ds)
    outdir = outdir + '/' + ds

    if not os.path.isdir(outdir + '/composites'):
        os.mkdir(outdir + '/composites')
    outdir = outdir + '/composites'

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

    dates = load_composite_dates_hhee(ds, res, var_ref, y0, y1, months, t_thresh, q_thresh, n_days, window, sw, tw, lat_range, lon_range, min_hw_size, max_hw_size. meth, cnty)

    files = get_files_hhee_data(ds, res, var_ref, months, lat_range, lon_range, min_hw_size, max_hw_size, t_thresh, q_thresh, n_days, meth, cnty)
    files.sort()

    files_ = []
    for f in files:
        for date in dates:
            if (date in f) and (date[5:7] != '6') and (date[8:10] != '21'):
                files_.append(f)

    nfs = len(files_)
    print('>>> {0} events <<<'.format(nfs))

    var_ref_data = load_proc_var_roll_mean(ds, res, var_ref, 2000, window, months, lat_range, lon_range)

    varhclim = load_pl_var_running_hclimato(ds, res, var, y0, y1, window, months, lat_range, lon_range, lvls)
    print('Regrid climato')
    varhclim = varhclim.interp(latitude=var_ref_data.latitude.values, longitude=var_ref_data.longitude.values)

    idayc = (months[0]-1) * 30 + int(dwindow/2)  # index of the first day of climatology

    lats = varhclim.latitude.values
    lons = varhclim.longitude.values
    lat_cen = lats[int(len(lats)/2)]
    lon_cen = lons[int(len(lons)/2)]


    print('-- Treat data --')

    var_hhee_vals = []
    var_hhee_clims = []

    for f in files_:
        hhee = load_hhee_data(f)

        i = np.argmax(hhee['tmax'])  # -> take the strongest value !!!   range(len(hhee['date'])):
        datestr = hhee['date'][i]
        y = int(datestr[0:4])
        m = int(datestr[5:7])
        d = int(datestr[8:10])
        tmaxlat = hhee['tmaxlat'][i]
        tmaxlon = hhee['tmaxlon'][i]
        tmax = hhee['tmax'][i]

        iday = (m-1)*30+d-1
        iday_ = iday - idayc

        data = load_pl_var_roll_mean(ds, res, var, y, window, months, lat_range, lon_range)

        print('Regrid %s data' % var)

        imsdata = data.groupby('time.month').groups
        mdata = data.isel(time=imsdata[m])
        idsdata = mdata.groupby('time.day').groups
        ddata = mdata.isel(time=idsdata[d])
        ddata = ddata.interp(latitude=var_ref_data.latitude.values, longitude=var_ref_data.longitude.values)

        boxshape = ddata.sel(latitude=slice(lat_cen+latwmin, lat_cen+latwmax), longitude=slice(lon_cen+lonwmin, lon_cen+lonwmax)).isel(time=0, pressure=0).shape   # get shape of a complete box

        data_hhee = ddata.sel(latitude=slice(tmaxlat+latwmin, tmaxlat+latwmax), longitude=slice(tmaxlon+lonwmin, tmaxlon+lonwmax))

        if data_hhee.isel(time=0, pressure=0).shape == boxshape:  # take only square (full) domains
            data_hhee_clim = varhclim.sel(latitude=slice(tmaxlat+latwmin, tmaxlat+latwmax), longitude=slice(tmaxlon+lonwmin, tmaxlon+lonwmax)).isel(days=iday_)

            assert (data_hhee_clim.time_level_0.values == m) and (data_hhee_clim.time_level_1.values == d)
            assert data_hhee_clim.isel(hour=0, pressure=0).shape == boxshape, 'Shape issue'  # data_hhee.isel(time=0, pressure=0).shape

            var_hhee_vals.append(data_hhee)
            var_hhee_clims.append(data_hhee_clim)

            print('\n  -- HHEE: {0},{1},{2} --'.format(tmaxlat, tmaxlon, datestr))

    #~ Remove event with incomplete time steps
    #igood = [i for i, var in enumerate(var_hhee_vals) if var.shape[0] == 8]
    #var_hhee_vals = [val for val in var_hhee_vals if val.shape[0] == 8]
    #var_hhee_clims = [var_hhee_clims[i] for i in igood]
    #~ Remove event with incomplete time steps

    var_hhee_vals = np.asarray(var_hhee_vals)
    var_hhee_clims = np.asarray(var_hhee_clims)


    print('\n-- Save --')

    outfile_vals = outdir + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_)
    outfile_clim = outdir + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_climato'

    with open(outfile_vals, 'wb') as pics:
        pickle.dump(obj=var_hhee_vals, file=pics)

    with open(outfile_clim, 'wb') as pics:
        pickle.dump(obj=var_hhee_clims, file=pics)


print('Done')

