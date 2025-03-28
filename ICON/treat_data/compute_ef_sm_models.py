"""Compute SM-EF models"""
"""ICON data"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import itertools
import matplotlib.pyplot as plt

from global_land_mask import is_land

from config import DATADIR, FIGDIR
from functions.landatmospherecoupling import LACR
from ICON.make_data.a_compute_data import load_data_ICON
from ICON.make_data.a2_compute_data_multi_level import load_data_multi_level_ICON


### CST ###


### FUNC ###

def load_sm_ef_model_number(dataset='ICON', experiment='ngc4008', zoom=9, year0=2020, year1=2029, lat_range=(-30., 30.), lon_range=(-180., 180.), months=[1,2,3,4,5,6,7,8,9,10,11,12], level=0):
    """Load SM-EF coupling model numbers"""
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])

    datapath = DATADIR + '/' + dataset + '/' + experiment + '/ef/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/z' + str(zoom) + '/' + years_ + '/' + months_
    outfile = datapath + '/ef_sm_model_number_level=' + str(level) + '.nc'
    out = xr.open_dataarray(outfile)

    return out


def load_wilting_point(dataset='ICON', experiment='ngc4008', zoom=9, year0=2020, year1=2029, lat_range=(-30., 30.), lon_range=(-180., 180.), months=[1,2,3,4,5,6,7,8,9,10,11,12], level=0):
    """Load wilting point"""
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])

    datapath = DATADIR + '/' + dataset + '/' + experiment + '/ef/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/z' + str(zoom) + '/' + years_ + '/' + months_
    outfile = datapath + '/wilting_point_level=' + str(level) + '.nc'
    out = xr.open_dataarray(outfile)

    return out


def load_critical_point(dataset='ICON', experiment='ngc4008', zoom=9, year0=2020, year1=2029, lat_range=(-30., 30.), lon_range=(-180., 180.), months=[1,2,3,4,5,6,7,8,9,10,11,12], level=0):
    """Load critical point"""
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])

    datapath = DATADIR + '/' + dataset + '/' + experiment + '/ef/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/z' + str(zoom) + '/' + years_ + '/' + months_
    outfile = datapath + '/critical_point_level=' + str(level) + '.nc'
    out = xr.open_dataarray(outfile)

    return out


def load_slope(dataset='ICON', experiment='ngc4008', zoom=9, year0=2020, year1=2029, lat_range=(-30., 30.), lon_range=(-180., 180.), months=[1,2,3,4,5,6,7,8,9,10,11,12], level=0):
    """Load dEF / dSM slope"""
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])

    datapath = DATADIR + '/' + dataset + '/' + experiment + '/ef/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/z' + str(zoom) + '/' + years_ + '/' + months_
    outfile = datapath + '/slope_level=' + str(level) + '.nc'
    out = xr.open_dataarray(outfile)

    return out


def load_time_trans(dataset='ICON', experiment='ngc4008', zoom=9, year0=2020, year1=2029, lat_range=(-30., 30.), lon_range=(-180., 180.), months=[1,2,3,4,5,6,7,8,9,10,11,12], level=0):
    """Load fractional time spent in transitional regime"""
    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    years_ = str(year0) + '-' + str(year1)
    months_ = "-".join([str(m) for m in months])

    datapath = DATADIR + '/' + dataset + '/' + experiment + '/ef/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max) + '/z' + str(zoom) + '/' + years_ + '/' + months_
    outfile = datapath + '/time_trans_level=' + str(level) + '.nc'
    out = xr.open_dataarray(outfile)

    return out


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ICON')
    parser.add_argument("--experiment", type=str, default='ngc4008')
    parser.add_argument("--zoom", type=int, default=9)
    parser.add_argument("--year0", type=int, default=2020)
    parser.add_argument("--year1", type=int, default=2029)
    parser.add_argument("--months", nargs="+", type=int, default=[1,2,3,4,5,6,7,8,9,10,11,12])
    parser.add_argument("--lat_range", nargs="+", type=float, default=[-30., 30.])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[-180., 180.])
    parser.add_argument("--level", type=int, default=0)

    opts = parser.parse_args()

    ds = opts.dataset
    exp = opts.experiment
    zoom = opts.zoom
    y0 = opts.year0
    y1 = opts.year1
    months = opts.months
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    level = opts.level

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    #assert lat_min < lat_max, "incorrect latitude range"
    lon_min = lon_range[0]
    lon_max = lon_range[1]
    #assert lon_min < lon_max, "incorrect longitude range"
    
    years = np.arange(y0, y1+1, 1)
    years_ = str(y0) + '-' + str(y1)
    months_ = "-".join([str(m) for m in months])


    #~ Outdir

    if not os.path.isdir(DATADIR + '/' + ds):
        os.mkdir(DATADIR + '/' + ds)
    outdir = DATADIR + '/' + ds

    if not os.path.isdir(outdir + '/' + exp):
        os.mkdir(outdir + '/' + exp)
    outdir = outdir + '/' + exp

    if not os.path.isdir(outdir + '/ef'):
        os.mkdir(outdir + '/ef')
    outdir = outdir + '/ef'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)

    if not os.path.isdir(outdir + '/z' + str(zoom)):
        os.mkdir(outdir + '/z' + str(zoom))
    outdir = outdir + '/z' + str(zoom)

    if not os.path.isdir(outdir + '/' + years_):
        os.mkdir(outdir + '/' + years_)
    outdir = outdir + '/' + years_

    if not os.path.isdir(outdir + '/' + months_):
        os.mkdir(outdir + '/' + months_)
    outdir = outdir + '/' + months_


    #~ Get data

    print("-- Get data --")

    efs = []
    sms = []

    for y in years:
        print(y, end=' : ', flush=True)

        ef = load_data_ICON(dataset=ds, experiment=exp, zoom=zoom, variable='ef', year=y, lat_range=(lat_min, lat_max), lon_range=(lon_min, lon_max))
        sm = load_data_multi_level_ICON(dataset=ds, experiment=exp, zoom=zoom, variable='sm', year=y, lat_range=(lat_min, lat_max), lon_range=(lon_min, lon_max), level=level)

        if (len(ef.time) > 366):  # resample EF to daily
            ef = efs.resample(time='1D').mean()

        efs.append(ef)
        sms.append(sm)

    efs = xr.concat(efs, dim='time')
    sms = xr.concat(sms, dim='time')

    imths = efs.groupby('time.month').groups
    lmonths = [imths[m] for m in months]
    lmonths = list(itertools.chain(*lmonths))

    efs = efs.isel(time=lmonths)
    sms = sms.isel(time=lmonths)

    lats = efs.lat.values
    lons = efs.lon.values


    '''print("\n-- Make land mask --")

    lons_, lats_ = np.meshgrid(lons, lats)
    land_mask = is_land(lats_, lons_)
    efs_land = efs.where(land_mask).isel(time=lmonths)
    sms_land = sms.where(land_mask).isel(time=lmonths)'''

    out_nums = xr.DataArray(data=None, dims=['lat', 'lon'], coords=dict(lat=(['lat'], lats), lon=(['lon'], lons)))
    out_wilts = xr.DataArray(data=None, dims=['lat', 'lon'], coords=dict(lat=(['lat'], lats), lon=(['lon'], lons)))
    out_crits = xr.DataArray(data=None, dims=['lat', 'lon'], coords=dict(lat=(['lat'], lats), lon=(['lon'], lons)))
    out_slopes = xr.DataArray(data=None, dims=['lat', 'lon'], coords=dict(lat=(['lat'], lats), lon=(['lon'], lons)))
    out_t_trans = xr.DataArray(data=None, dims=['lat', 'lon'], coords=dict(lat=(['lat'], lats), lon=(['lon'], lons)))


    print("\n-- Compute piecewise linear regression --")

    for ilat, lat in enumerate(lats):
        print('\n>>> {0} <<<'.format(lat))

        for ilon, lon in enumerate(lons):
            if is_land(lat, lon):
                efs_pt = efs.sel(lat=lat, lon=lon, method='nearest')
                sms_pt = sms.sel(lat=lat, lon=lon, method='nearest')

                #if not (np.isnan(efs_pt.values).all()) and not (np.isnan(sms_pt.values).all()):
                df = pd.DataFrame(data={'sm': sms_pt, 'ef': efs_pt})
                df_ = df.dropna(axis=0, how='any')
                df_ = df_.sort_values('sm')

                models = LACR(df_.sm.values, df_.ef.values)

                if not df_.sm.std() == 0:
                    out_nums.loc[lat, lon] = models.get_best_model_number()
                    out_wilts.loc[lat, lon] = models.get_wilting_point()
                    out_crits.loc[lat, lon] = models.get_critical_point()
                    out_slopes.loc[lat, lon] = models.get_slope()
                    out_t_trans.loc[lat, lon] = models.get_transitional_time_frac()
                    print(models.get_best_model(), end=' : ', flush=True)
                else:
                    out_nums.loc[lat, lon] = np.nan
                    out_wilts.loc[lat, lon] = np.nan
                    out_crits.loc[lat, lon] = np.nan
                    out_slopes.loc[lat, lon] = np.nan
                    out_t_trans.loc[lat, lon] = np.nan
                #sys.exit()

            else:
                out_nums.loc[lat, lon] = np.nan
                out_wilts.loc[lat, lon] = np.nan
                out_crits.loc[lat, lon] = np.nan
                out_slopes.loc[lat, lon] = np.nan
                out_t_trans.loc[lat, lon] = np.nan


            '''plt.figure()
            xd = np.linspace(df_.sm.min(), df_.sm.max(), len(df_.sm))
            y_lr = models.predicted_lr()
            y_dt = models.predicted_dt()
            y_tw = models.predicted_tw()
            y_dtw = models.predicted_dtw()

            plt.scatter(df_.sm , df_.ef, color='k', alpha=0.5, linewidths=0.)
            plt.plot(xd, y_lr, color='r', label='linear')
            plt.plot(xd, y_dt, color='b', label='dry-trans')
            plt.plot(xd, y_tw, color='g', label='trans-wet')
            plt.plot(xd, y_dtw, color='yellow', label='dry-trans-wet')
            plt.legend()
            plt.title((lat, lon))
            plt.savefig(FIGDIR + '/ef_sm_models_ICON_' + str(lat) + '-' + str(lon) + '.pdf')
            plt.close()
            sys.exit()'''

    #sys.exit()

    #~ Save

    outfile_nums = outdir + '/ef_sm_model_number_level=' + str(level) + '.nc'
    out_nums.to_netcdf(outfile_nums)

    outfile_wilt = outdir + '/wilting_point_level=' + str(level) + '.nc'
    out_wilts.to_netcdf(outfile_wilt)

    outfile_crit = outdir + '/critical_point_level=' + str(level) + '.nc'
    out_crits.to_netcdf(outfile_crit)

    outfile_slope = outdir + '/slope_level=' + str(level) + '.nc'
    out_slopes.to_netcdf(outfile_slope)

    outfile_t_trans = outdir + '/time_trans_level=' + str(level) + '.nc'
    out_t_trans.to_netcdf(outfile_t_trans)


print('Done')

