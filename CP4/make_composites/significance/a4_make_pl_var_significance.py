"""Make pressure levels variable composite for significance assessment"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import CP4OUTPATH
from CP4.make_climato.a6bis_compute_rolling_mean_theta_wb_pl_var import lvls
from CP4.make_climato.a2_compute_rolling_mean_proc_var import load_proc_var_roll_mean
from CP4.make_climato.a4_compute_rolling_mean_pl_var import load_pl_var_roll_mean
from CP4.make_climato.c4_compute_running_hourly_climato_pl_var import load_pl_var_running_hclimato
from CP4.make_composites.make_hhee_data import get_files_hhee_data, load_hhee_data
from CP4.make_composites.a1_make_var_field import load_composite_dates_hhee, get_path_composite_var_merge
from CP4.make_composites.a4_make_cross_section_pl_var_vertical_profile import get_path_composite_pl_var
from CP4.make_composites.significance.a_make_var_field_significance import get_path_significance


### CST ###

levels = lvls
mean_iaxis = {'latitude': 3, 'longitude': 2}  # long cross-section -> mean across latitude (2)


### FUNC ###

def load_composite_significance_pl(ds='CP4', var='theta_wb_pl', year0=1997, year1=2006, months=[3, 4, 5], lat_range=(4., 10.), lon_range=(-14., 10.), t_thresh=24., q_thresh=0.95, n_days=3, window=6, lat_window=(-0.1,0.1), lon_window=(-2.,2.), min_hw_size=1000., max_hw_size=5000., levels=None):
    "Load pressure levels variable HHE composite significance"
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

    datapath = get_path_significance(ds, var, year0, year1, months, t_thresh, q_thresh, n_days, lat_range, lon_range)
    outfile = datapath + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_mean_ano_pl.nc'

    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
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
    parser.add_argument("--n_days", type=int, default=3)
    parser.add_argument("--sampling_time", nargs="+", type=int, default=[-24, 6])
    parser.add_argument("--lat_window", nargs="+", type=float, default=[-0.1, 0.1])
    parser.add_argument("--lon_window", nargs="+", type=float, default=[-2., 2.])
    parser.add_argument("--cross_section_axis", type=str, default='longitude')  # cross section direction
    parser.add_argument("--levels", nargs="+", type=float, default=None)

    opts = parser.parse_args()

    ds = opts.dataset
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
    n_days = opts.n_days
    samtime = opts.sampling_time
    latw = opts.lat_window
    lonw = opts.lon_window
    cross_section_axis = opts.cross_section_axis
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

    files = get_files_hhee_data(ds, months, lat_range, lon_range, min_hw_size, max_hw_size, t_thresh, q_thresh, n_days)
    files.sort()

    dates = load_composite_dates_hhee(ds, 'twb', y0, y1, months, t_thresh, q_thresh, n_days, window, sw, tw, lat_range, lon_range, min_hw_size, max_hw_size)

    files_ = []
    for f in files:
        for date in dates:
            if (date in f) and (date[5:7] != '6') and (date[8:10] != '21'):
                files_.append(f)

    nfs = len(files_)
    print('>>> {0} events <<<'.format(nfs))

    twbdata = load_proc_var_roll_mean(months=months, lat_range=lat_range, lon_range=lon_range)

    varhclim = load_pl_var_running_hclimato(ds, var, y0, y1, window, months, lat_range, lon_range, lvls)
    print('Regrid climato')
    varhclim = varhclim.interp(latitude=twbdata.latitude.values, longitude=twbdata.longitude.values)

    idayc = (months[0]-1) * 30 + int(dwindow/2)  # index of the first day of climatology

    lats = varhclim.latitude.values
    lons = varhclim.longitude.values
    lat_cen = lats[int(len(lats)/2)]
    lon_cen = lons[int(len(lons)/2)]

    boxshape = varhclim.sel(latitude=slice(lat_cen+latwmin, lat_cen+latwmax), longitude=slice(lon_cen+lonwmin, lon_cen+lonwmax)).isel(days=0, hour=0, pressure=0).shape   # get shape of a complete box

    xs = np.arange(0.5, boxshape[1], 1)
    ys = np.arange(0.5, boxshape[0], 1)


    print('-- Treat data --')

    iaxis = mean_iaxis[cross_section_axis]

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

        print('\n>>> {0} <<<'.format(datestr))

        iday = (m-1)*30+d-1
        iday_ = iday - idayc

        var_hhee_anos = []

        yrs_ = np.delete(years, np.where(years==y)[0][0])
        for y_ in yrs_:
            print(y_, end=' : ', flush=True)

            data = load_pl_var_roll_mean(ds, var, y_, window, months, lat_range, lon_range)

            imsdata = data.groupby('time.month').groups
            mdata = data.isel(time=imsdata[m])
            idsdata = mdata.groupby('time.day').groups
            ddata = mdata.isel(time=idsdata[d])
            ddata = ddata.interp(latitude=twbdata.latitude.values, longitude=twbdata.longitude.values)

            idata = np.where(data.time.values == ddata.time[0].values)[0][0]

            data_hhee = ddata.sel(latitude=slice(tmaxlat+latwmin, tmaxlat+latwmax), longitude=slice(tmaxlon+lonwmin, tmaxlon+lonwmax))
            data_hhee_clim = varhclim.sel(latitude=slice(tmaxlat+latwmin, tmaxlat+latwmax), longitude=slice(tmaxlon+lonwmin, tmaxlon+lonwmax)).isel(days=iday_)

            assert (data_hhee_clim.time_level_0.values == m) and (data_hhee_clim.time_level_1.values == d)
            assert data_hhee_clim.isel(hour=0, pressure=0).shape == boxshape, 'Shape issue'

            #sys.exit()

            #data_hhee_clim = data_hhee_clim.values.reshape(data_hhee.shape)  # 6*24 -> 144
            #varhclim_time = data_hhee_clim[tw_before:tw_after+1,:,:].mean(axis=0)
            #vardata_time = data_hhee[tw_before:tw_after+1,:,:].mean(axis=0)

            data_hhee = np.nanmean(data_hhee, axis=iaxis)
            data_hhee_clim = np.nanmean(data_hhee_clim, axis=iaxis)

            var_hhee_ano = data_hhee - data_hhee_clim
            var_hhee_anos.append(var_hhee_ano)

            #print('\n  -- HHEE: {0},{1},{2} --'.format(tmaxlat, tmaxlon, datestr))

        var_hhee_anos = np.asarray(var_hhee_anos)

        ds_ano = xr.DataArray(data=var_hhee_anos, dims=['year', 'time', 'pressure', 'x'], coords=dict(year=yrs_, time=(['time'], np.arange(0, 21+3, 3)), pressure=(['pressure'], levels), x=(['x'], np.arange(0.5, var_hhee_anos.shape[-1], 1))))
        ds_anos.append(ds_ano)

    out = xr.concat(ds_anos, dim='n')


    #~ Save

    print('\n-- Save --')

    outfile = outdir + '/' + space_scale + '_' + str(window) + 'h_lat:' + latw_ + '_lon:' + lonw_ + '_levels=' + str(lvls_) + '_mean_ano_pl.nc'

    if not os.path.isfile(outfile):
        out.to_netcdf(outfile)
    else:
        print('File already exist!')


print('Done')


