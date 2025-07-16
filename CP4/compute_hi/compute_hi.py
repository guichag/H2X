"""Calculation of HI using NOAA Equation
See HI Table on Wikipedia: https://en.wikipedia.org/wiki/Heat_index"""

import sys
import os
import argparse
import numpy as np
import xarray as xr

from config import H2XPATH
from DATA.read_variables import get_var_filenames, create_CP4_filename
from utils.hi_formula import *


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

    opts = parser.parse_args()

    ds = opts.dataset
    res = opts.resolution
    y0 = opts.year0
    y1 = opts.year1

    res_ = str(res) + 'km'

    years = np.arange(y0, y1+1, 1)

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

    if not (outdir + '/hi'):
        os.mkdir(outdir + '/hi')
    outdir = outdir + '/hi'


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
            #print(tf)

            date = tf[-28:-16]
            m = int(date[4:6])
            d = int(date[6:8])

            tdata = xr.open_dataset(tf)[tvar_id] - 273.15   # EN CELSIUS !!!!
            pdata = xr.open_dataset(pf)[pvar_id]
            hdata = xr.open_dataset(qf)[qvar_id]

            time = tdata.time.values
            lats = tdata.latitude
            lons = tdata.longitude

            outfile = outdir + '/' + tf[-28:-3] + '.nc'

            hiout = []

            for it, t in enumerate(time):
                print(t, end=' : ', flush=True)

                htvals = tdata.sel(time=t).values.flatten()
                hpvals = pdata.sel(time=t).values.flatten()
                hhvals = hdata.isel(time=it).values.flatten()
                hhvals[hhvals <= 0] = np.float64(1e-6)
                hhvals[hhvals >= 100] = 99.999

                rh = get_rh(htvals, hhvals, hpvals, "pa")
                rh[rh > 100.] = 100.
                htvalsfh = Tcelsius_to_farhenheit(htvals)   # hourly T values in farhenheit -> for HI calculation

                hi = compute_hi_all(htvals, rh)

                hiout.append(hi)

            hiout = np.asarray(hiout)
            hiout = np.reshape(hiout, (len(time), len(lats), len(lons)))
            hiout = np.round(hiout, 5)

            hiout = xr.DataArray(data=hiout, dims=["time", "latitude", "longitude"], coords=dict(time=time, latitude=(["latitude"], lats.values), longitude=(["longitude"], lons.values)))
            hiout = hiout.where(hiout < 60.)

            hiout.to_netcdf(outfile)

            print('\n{0}: HImin={1} ; HImax={2}'.format(tf, hiout.min().values, hiout.max().values))


print('Done')

