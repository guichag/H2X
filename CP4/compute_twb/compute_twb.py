"""Calculation of wet bulb temperature using the corrected Goodman-Raymond function"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import H2XPATH
from DATA.read_variables import get_var_filenames, create_CP4_filename
from CP4.compute_twb.compute_rh import load_rh_data
from utils.wetbulb_dj08_spedup import WetBulb_all


### CST ###

tvar = 't2'
qvar = 'q2'
pvar = 'p_srfc'


### FUNC ###


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CP4A')
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--year0", type=int, default=1997)
    parser.add_argument("--year1", type=int, default=2006)
    parser.add_argument("--humtype", type=int, default=0)

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    y0 = opts.year0
    y1 = opts.year1
    humtype = opts.humtype

    years = np.arange(y0, y1+1, 1)
    res_ = str(res) + 'km'

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

    if not (outdir + '/twb'):
        os.mkdir(outdir + '/twb')
    outdir = outdir + '/twb'


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
        y_ = '_' + str(y)
        ytfiles = [f for f in tfiles if y_ in f]
        yqfiles = [f for f in qfiles if y_ in f]
        ypfiles = [f for f in pfiles if y_ in f]

        for tf, qf, pf in zip(ytfiles, yqfiles, ypfiles):
            print(tf)
            date = tf[-28:-16]
            m = int(date[4:6])
            d = int(date[6:8])

            tdata = xr.open_dataset(tf)[tvar_id] - 273.15
            pdata = xr.open_dataset(pf)[pvar_id]
            if humtype == 0:
                hdata = xr.open_dataset(qf)[qvar_id]
            elif humtype == 1:
                hdata = load_rh_data(y, m, d)

            time = tdata.time.values
            lats = tdata.latitude
            lons = tdata.longitude

            outfile = outdir + '/' + tf[-28:-3] + sfx + '.nc'

            twbout = []
            for it, t in enumerate(time):
                print(t, end=' : ', flush=True)

                htvals = tdata.sel(time=t).values.flatten()
                hpvals = pdata.sel(time=t).values.flatten()
                hhvals = hdata.isel(time=it).values.flatten()
                hhvals[hhvals <= 0] =  np.float64(1e-6)
                hhvals[hhvals >= 100] = 99.999

                twb = WetBulb_all(htvals, hpvals, hhvals, humtype)
                twb = np.reshape(twb, (len(lats), len(lons)))
                twbout.append(twb)

            twbout = np.asarray(twbout)
            twbout = xr.DataArray(data=twbout, dims=["time", "latitude", "longitude"], coords=dict(time=time, latitude=(["latitude"], lats.values), longitude=(["longitude"], lons.values)))
            twbout.to_netcdf(outfile)

print('Done')

