"""Calculation of wet bulb temperature on pressure levels using the corrected Goodman-Raymond function"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import *
from DATA.read_variables import get_var_filenames, create_CP4_filename
from DATA.read_variables import load_proc_var_data
from utils.wetbulb_dj08_spedup import WetBulb_all, QSat_2, WetBulb
from utils.meteo_constants import vkp, lambd_a


### CST ###

tvar = 't_pl'
qvar = 'q_pl'
pvar = 'p_srfc'

p0 = np.float64(100000)
lvls = [1000., 950., 925., 900., 850., 800., 700.]


### FUNC ###


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--lat_range", nargs="+", type=float, default=[None, None])
    parser.add_argument("--lon_range", nargs="+", type=float, default=[None, None])
    parser.add_argument("--humtype", type=int, default=0)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    y0 = opts.year0
    y1 = opts.year1
    lat_range = opts.lat_range
    lon_range = opts.lon_range
    humtype = opts.humtype

    years = np.arange(y0, y1+1, 1)

    res_ = str(res) + 'km'

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    lon_min = lon_range[0]
    lon_max = lon_range[1]

    if humtype == 1:
        sfx = '_with-rh'
    elif humtype == 0:
        sfx = ''
    else:
        print('Humidity type must be 0 (specific) or 1 (relative)')

    tvar_id = create_CP4_filename(tvar)
    qvar_id = create_CP4_filename(qvar)
    pvar_id = create_CP4_filename(pvar)


    #~ Outdir

    if not (H2XPATH + '/' + ds):
        os.mkdir(H2XPATH + '/' + ds)
    outdir = H2XPATH + '/' + ds

    if not (outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not (outdir + '/twb_pl'):
        os.mkdir(outdir + '/twb_pl')
    outdir = outdir + '/twb_pl'

    if not os.path.isdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)):
        os.mkdir(outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max))
    outdir = outdir + '/lat={0},{1}_lon={2},{3}'.format(lat_min, lat_max, lon_min, lon_max)


    #~ Get data

    tfiles = get_var_filenames(ds, res, tvar)
    tfiles = [f for f in tfiles if f[-2:] == 'nc']
    tfiles.sort()
    qfiles = get_var_filenames(ds, res, qvar)
    qfiles = [f for f in qfiles if f[-2:] == 'nc']
    qfiles.sort()
    pfiles = get_var_filenames(ds, res, pvar)
    pfiles = [f for f in pfiles if f[-2:] == 'nc']
    pfiles.sort()


    for y in years:
        y_ = '_' + str(y)  # + '0430' # -> make April only
        ytfiles = [f for f in tfiles if y_ in f]
        yqfiles = [f for f in qfiles if y_ in f]
        ypfiles = [f for f in pfiles if y_ in f]

        for tf, qf, pf in zip(ytfiles, yqfiles, ypfiles):
            print(tf)
            date = tf[-28:-16]
            m = int(date[4:6])
            d = int(date[6:8])

            tdata = xr.open_dataset(tf)[tvar_id]
            tdata = tdata.assign_coords(longitude=tdata.longitude - 360)  # - 273.15   # EN CELSIUS !!!!
            tdata = tdata.where(tdata > 0, np.nan)
            tdata = tdata - 273.15
            tdata = tdata.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

            pdata = xr.open_dataset(pf)[pvar_id]
            pdata = pdata.assign_coords(longitude=pdata.longitude - 360)
            print('interpolate pressure grid')
       	    pdata = pdata.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            pdata = pdata.interp(latitude=tdata.latitude, longitude=tdata.longitude)

            if humtype == 0:
                hdata = xr.open_dataset(qf)[qvar_id]
                hdata = hdata.assign_coords(longitude=hdata.longitude - 360)
            elif humtype == 1:
                hdata = load_proc_var_data(ds, res, 'rh', y, m, d)
       	    hdata = hdata.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

            time = tdata.time.values
            lats = tdata.latitude
            lons = tdata.longitude

            outfile = outdir + '/' + tf[-28:-3] + sfx + '.nc'

            twbout = []
            for it, t in enumerate(time):
                print(t, end=' : ', flush=True)

                htvals = tdata.sel(time=t, pressure=lvls).values.flatten()
                inant = np.where(np.isnan(htvals))[0]
                htvals[inant] = 100  # make impossible twb values

                hpvals = pdata.sel(time=t)
                hpvals = np.stack([hpvals for i in range(len(lvls))])
                hpvals = hpvals.flatten()
                inanp = np.where(np.isnan(hpvals))[0]
                hpvals[inanp] = 9999

                hhvals = hdata.isel(time=it).sel(pressure=lvls).values.flatten()
                hhvals[hhvals <= 0] =  np.float64(1e-6)
                hhvals[hhvals >= 100] = 99.999
                inanh = np.where(np.isnan(hhvals))[0]
                hhvals[inanh] = 0.5  # make impossible twb values

                twb = WetBulb_all(htvals, hpvals, hhvals, humtype)
                twb = np.reshape(twb, (len(lvls), len(lats), len(lons)))
                twb[twb < -300] = np.nan

                twbout.append(np.round(twb, 5))


            twbout = np.asarray(twbout)
            twbout = xr.DataArray(data=twbout, dims=["time", "pressure", "latitude", "longitude"], coords=dict(time=time, pressure=lvls, latitude=(["latitude"], lats.values), longitude=(["longitude"], lons.values)))

            twbout.to_netcdf(outfile)


print('Done')

