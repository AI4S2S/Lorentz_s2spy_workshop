#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal.windows as spwin
import xarray as xr
from netCDF4 import num2date
from typing import Union

flatten = lambda l: list(itertools.chain.from_iterable(l))

def remove_leapdays(datetime):
    mask_lpyrfeb = np.logical_and((datetime.month == 2), (datetime.day == 29))

    dates_noleap = datetime[mask_lpyrfeb==False]
    return dates_noleap

def detrend_anom_ncdf3D(infile, outfile, loadleap=False, detrend=True, anomaly=True,
                        apply_fft=True, n_harmonics=6, encoding={}):
    '''
    Function for preprocessing
    - Calculate anomalies (removing seasonal cycle); requires full years of data.
        for daily data, a 25-day rolling mean is applied, afterwards this is
        further smoothened by a FFT with 6 annual harmonics.
    - linear or loess long-term detrend, see core_pp.detrend_wrapper?
    '''
    # loadleap=False; seldates=None; selbox=None; format_lon='east_west';
    # auto_detect_mask=False; detrend=True; anomaly=True;
    # apply_fft=True; n_harmonics=6; encoding=None
    #%%
    ds = xr.open_dataarray(infile)
    if loadleap==False:
        ds = ds.sel(time=remove_leapdays(pd.to_datetime(ds.time.values)))
        
    # check if 3D data (lat, lat, lev) or 2D
    check_dim_level = any([level in ds.dims for level in ['lev', 'level']])

    if check_dim_level:
        key = ['lev', 'level'][any([level in ds.dims for level in ['lev', 'level']])]
        levels = ds[key]
        output = np.empty( (ds.time.size,  ds.level.size, ds.latitude.size, ds.longitude.size), dtype='float32' )
        output[:] = np.nan
        for lev_idx, lev in enumerate(levels.values):
            ds_2D = ds.sel(level=lev)

            output[:,lev_idx,:,:] = detrend_xarray_ds_2D(ds_2D, detrend=detrend, anomaly=anomaly,
                                      apply_fft=apply_fft, n_harmonics=n_harmonics)
    else:
        output = detrend_xarray_ds_2D(ds, detrend=detrend, anomaly=anomaly,
                                      apply_fft=apply_fft, n_harmonics=n_harmonics)

    print(f'\nwriting ncdf file to:\n{outfile}')
    output = xr.DataArray(output, name=ds.name, dims=ds.dims, coords=ds.coords)
    # copy original attributes to xarray
    output.attrs = ds.attrs
    pp_dict = {'anomaly':str(anomaly), 'fft':str(apply_fft), 'n_harmonics':n_harmonics,
               'detrend':str(detrend)}
    output.attrs.update(pp_dict)
    # ensure mask
    output = output.where(output.values != 0.).fillna(-9999)
    encoding.update({'_FillValue': -9999})
    encoding_var = ( {ds.name : encoding} )
    mask =  (('latitude', 'longitude'), (output.values[0] != -9999) )
    output.coords['mask'] = mask

    # save netcdf
    output.to_netcdf(outfile, mode='w', encoding=encoding_var)
    #%%
    return

def detrend_xarray_ds_2D(ds, detrend, anomaly, apply_fft=False, n_harmonics=6,
                         kwrgs_NaN_handling: dict=None):
    #%%

    if type(ds.time[0].values) != type(np.datetime64()):
        numtime = ds['time']
        dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
        if numtime.attrs['calendar'] != 'gregorian':
            dates = [d.strftime('%Y-%m-%d') for d in dates]
        dates = pd.to_datetime(dates)
    else:
        dates = pd.to_datetime(ds['time'].values)
    stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')
    ds['time'] = dates

    def _detrendfunc2d(arr_oneday, arr_oneday_smooth):

        # get trend of smoothened signal
        no_nans = np.nan_to_num(arr_oneday_smooth)
        detrended_sm = sp.signal.detrend(no_nans, axis=0, type='linear')
        nan_true = np.isnan(arr_oneday)
        detrended_sm[nan_true.values] = np.nan
        # subtract trend smoothened signal of arr_oneday values
        trend = (arr_oneday_smooth - detrended_sm)- np.mean(arr_oneday_smooth, 0)
        detrended = arr_oneday - trend
        return detrended, detrended_sm


    def detrendfunc2d(arr_oneday):
        return xr.apply_ufunc(_detrendfunc2d, arr_oneday,
                              dask='parallelized',
                              output_dtypes=[float])

    if kwrgs_NaN_handling is not None:
        ds = NaN_handling(ds, **kwrgs_NaN_handling)

    if anomaly:
        if (stepsyr.day== 1).all() == True or int(ds.time.size / 365) >= 120:
            print('\nHandling time series longer then 120 yrs or monthly data, no smoothening applied')
            data_smooth = ds.values
            if (stepsyr[1] - stepsyr[0]).days in [28,29,30,31]:
                window_s = False

        elif (stepsyr.day== 1).all() == False and int(ds.time.size / 365) < 120:
            window_s = max(min(25,int(stepsyr.size / 12)), 1)
            # print('Performing {} day rolling mean'
            #       ' to get better interannual statistics'.format(window_s))
            # from time import time
            # start = time()
            print('applying rolling mean, beware: memory intensive')
            data_smooth =  rolling_mean_np(ds.values, window_s, win_type='boxcar')
            # data_smooth_xr = ds.rolling(time=window_s, min_periods=1,
            #                             center=True).mean(skipna=False)
            # passed = time() - start / 60

        output_clim3d = np.zeros((stepsyr.size, ds.latitude.size, ds.longitude.size),
                                   dtype='float32')

        for i in range(stepsyr.size):

            sliceyr = np.arange(i, ds.time.size, stepsyr.size)
            arr_oneday_smooth = data_smooth[sliceyr]

            if i==0: print('using absolute anomalies w.r.t. climatology of '
                            'smoothed concurrent day accross years\n')
            output_clim2d = arr_oneday_smooth.mean(axis=0)
            # output[i::stepsyr.size] = arr_oneday - output_clim3d
            output_clim3d[i,:,:] = output_clim2d

            progress = int((100*(i+1)/stepsyr.size))
            print(f"\rProcessing {progress}%", end="")

        if apply_fft:
            # beware, mean by default 0, add constant = False
            list_of_harm= [1/h for h in range(1,n_harmonics+1)]
            clim_rec = reconstruct_fft_2D(xr.DataArray(data=output_clim3d,
                                                       coords=ds.sel(time=stepsyr).coords,
                                                       dims=ds.dims),
                                          list_of_harm=list_of_harm,
                                          add_constant=False)
            # Adding mean of origninal ds
            clim_rec += ds.values.mean(0)
            output = ds - np.tile(clim_rec, (int(dates.size/stepsyr.size), 1, 1))
        elif apply_fft==False:
            output = ds - np.tile(output_clim3d, (int(dates.size/stepsyr.size), 1, 1))
    else:
        output = ds


    # =============================================================================
    # test gridcells:
    # =============================================================================
    if anomaly:
        la1 = int(ds.shape[1]/2)
        lo1 = int(ds.shape[2]/2)
        la2 = int(ds.shape[1]/3)
        lo2 = int(ds.shape[2]/3)

        tuples = [[la1, lo1], [la1+1, lo1],
                  [la2, lo2], [la2+1, lo2]]
        if apply_fft:
            fig, ax = plt.subplots(4,2, figsize=(16,8))
        else:
            fig, ax = plt.subplots(2,2, figsize=(16,8))
        ax = ax.flatten()
        for i, lalo in enumerate(tuples):
            ts = ds[:,lalo[0],lalo[1]]
            while bool(np.isnan(ts).all()):
                lalo[1] += 5
                ts = ds[:,lalo[0],lalo[1]]
            lat = int(ds.latitude[lalo[0]])
            lon = int(ds.longitude[lalo[1]])
            print(f"\rVisual test latlon {lat} {lon}", end="")

            if window_s == False: # no daily data
                rawdayofyear = ts.groupby('time.month').mean('time')
            else:
                rawdayofyear = ts.groupby('time.dayofyear').mean('time')

            ax[i].set_title(f'latlon coord {lat} {lon}')
            for yr in np.unique(dates.year):
                singleyeardates = get_oneyr(dates, yr)
                ax[i].plot(ts.sel(time=singleyeardates), alpha=.1, color='purple')

            if window_s is not None:
                ax[i].plot(output_clim3d[:,lalo[0],lalo[1]], color='green', linewidth=2,
                     label=f'clim {window_s}-day rm')
            ax[i].plot(rawdayofyear, color='black', alpha=.6,
                       label='clim mean dayofyear')
            if apply_fft:
                ax[i].plot(clim_rec[:,lalo[0],lalo[1]][:365], 'r-',
                           label=f'fft {n_harmonics}h on (smoothened) data')
                diff = clim_rec[:,lalo[0],lalo[1]][:singleyeardates.size] - output_clim3d[:,lalo[0],lalo[1]]
                diff = diff / ts.std(dim='time').values
                ax[i+len(tuples)].plot(diff)
                ax[i+len(tuples)].set_title(f'latlon coord {lat} {lon} diff/std(alldata)')
            ax[i].legend()
        ax[-1].text(.5,1.2, 'Visual analysis',
                transform=ax[0].transAxes,
                ha='center', va='bottom')
        plt.subplots_adjust(hspace=.4)
        fig, ax = plt.subplots(1, figsize=(5,3))
        std_all = output[:,lalo[0],lalo[1]].std(dim='time')
        monthlymean = output[:,lalo[0],lalo[1]].groupby('time.month').mean(dim='time')
        (monthlymean/std_all).plot(ax=ax)
        ax.set_ylabel('standardized anomaly [-]')
        ax.set_title(f'climatological monthly means anomalies latlon coord {lat} {lon}')
        fig, ax = plt.subplots(1, figsize=(5,3))
        summer = output.sel(time=get_subdates(dates, start_end_date=('06-01', '08-31')))
        summer.name = f'std {summer.name}'
        (summer.mean(dim='time') / summer.std(dim='time')).plot(ax=ax,
                                                                vmin=-3,vmax=3,
                                                                cmap=plt.cm.bwr)
        ax.set_title('summer composite mean [in std]')
    print('\n')

    if detrend==True: # keep old workflow working with linear detrending
        output = detrend_wrapper(output, kwrgs_detrend={'method':'linear'})
    elif type(detrend) is dict:
        output = detrend_wrapper(output, kwrgs_detrend=detrend)

    #%%
    return output

def rolling_mean_np(arr, win, center=True, win_type='boxcar'):


    df = pd.DataFrame(data=arr.reshape( (arr.shape[0], arr[0].size)))

    if win_type == 'gaussian':
        w_std = win/3.
        print('Performing {} day rolling mean with gaussian window (std={})'
              ' to get better interannual statistics'.format(win,w_std))
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(range(-int(win/2),+round(win/2+.49)), spwin.gaussian(win, w_std))
        plt.title('window used for rolling mean')
        plt.xlabel('timesteps')
        rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='gaussian').mean(std=w_std)
    elif win_type == 'boxcar':
        fig, ax = plt.subplots(figsize=(3,3))
        plt.plot(spwin.boxcar(win))
        plt.title('window used for rolling mean')
        plt.xlabel('timesteps')
        rollmean = df.rolling(win, center=center, min_periods=1,
                          win_type='boxcar').mean()

    return rollmean.values.reshape( (arr.shape))

def detrend_lin_longterm(ds, plot=True, return_trend=False):
    offset_clim = np.mean(ds, 0)
    dates = pd.to_datetime(ds.time.values)
    detrended = sp.signal.detrend(np.nan_to_num(ds), axis=0, type='linear')
    detrended[np.repeat(np.isnan(offset_clim).expand_dims('t').values,
                        dates.size, 0 )] = np.nan # restore NaNs
    detrended += np.repeat(offset_clim.expand_dims('time'), dates.size, 0 )
    detrended = detrended.assign_coords(
                coords={'time':dates})
    if plot:
        _check_trend_plot(ds, detrended)

    if return_trend:
        out = ( detrended,  (ds - detrended)+offset_clim )
    else:
        out = detrended
    return out

def _check_trend_plot(ds, detrended):
    if len(ds.shape) > 2:
        # plot single gridpoint for visual check
        always_NaN_mask = np.isnan(ds).all(axis=0)
        lats, lons = np.where(~always_NaN_mask)
        tuples = np.stack([lats, lons]).T
        tuples = tuples[::max(1,int(len(tuples)/3))]
        # tuples = np.stack([lats, lons]).T
        fig, ax = plt.subplots(len(tuples), figsize=(8,8))
        for i, lalo in enumerate(tuples):
            ts = ds[:,lalo[0],lalo[1]]
            la = lalo[0]
            lo = lalo[1]
            # while bool(np.isnan(ts).all()):
            #     lo += 5
            #     try:
            #         ts = ds[:,la,lo]
            #     except:

            lat = int(ds.latitude[la])
            lon = int(ds.longitude[lo])
            print(f"\rVisual test latlon {lat} {lon}", end="")

            ax[i].set_title(f'latlon coord {lat} {lon}')
            ax[i].plot(ts)
            ax[i].plot(detrended[:,la,lo])
            trend1d = ts - detrended[:,la,lo]
            linregab = np.polyfit(np.arange(trend1d.size), trend1d, 1)
            linregab = np.insert(linregab, 2, float(trend1d[-1] - trend1d[0]))
            ax[i].plot(trend1d+ts.mean(axis=0))
            ax[i].text(.05, .05,
            'y = {:.2g}x + {:.2g}, max diff: {:.2g}'.format(*linregab),
            transform=ax[i].transAxes)
        plt.subplots_adjust(hspace=.5)
        ax[-1].text(.5,1.2, 'Visual analysis of trends',
                    transform=ax[0].transAxes,
                    ha='center', va='bottom')
    elif len(ds.shape) == 1:
        fig, ax = plt.subplots(1, figsize=(8,4))
        ax.set_title('detrend 1D ts')
        ax.plot(ds.values)
        ax.plot(detrended)
        trend1d = ds - detrended
        linregab = np.polyfit(np.arange(trend1d.size), trend1d, 1)
        linregab = np.insert(linregab, 2, float(trend1d[-1] - trend1d[0]))
        ax.plot(trend1d + (ds.mean(axis=0)) )
        ax.text(.05, .05,
        'y = {:.2g}x + {:.2g}, max diff: {:.2g}'.format(*linregab),
        transform=ax.transAxes)
    else:
        pass

def to_np(data):
    if type(data) is pd.DataFrame:
        kwrgs = {'columns':data.columns, 'index':data.index};
        input_dtype = pd.DataFrame
    elif type(data) is xr.DataArray:
        kwrgs= {'coords':data.coords, 'dims':data.dims, 'attrs':data.attrs,
                'name':data.name}
        input_dtype = xr.DataArray
    if type(data) is not np.ndarray:
        data = data.values # attempt to make np.ndarray (if xr.DA of pd.DF)
    else:
        input_dtype = np.ndarray ; kwrgs={}
    return data, kwrgs, input_dtype

def back_to_input_dtype(data, kwrgs, input_dtype):
    if input_dtype is pd.DataFrame:
        data = pd.DataFrame(data, **kwrgs)
    elif input_dtype is xr.DataArray:
        data = xr.DataArray(data, **kwrgs)
    return data

def NaN_handling(data, inter_method: str='spline', order=2, inter_NaN_limit: float=None,
                 extra_NaN_limit: float=.05, final_NaN_to_clim: bool=True,
                 missing_data_ts_to_nan: Union[bool, float, int]=False):
    '''
    4 options to deal with NaNs (performed in this order):
        1. mask complete timeseries if more then % of dp are missing
        2. Interpolation (by default via order 2 spine)
        3. extrapolation NaNs at edged of timeseries
        4. fills the left-over NaNs with the mean over the
           time-axis.


    Parameters
    ----------
    data : xr.DataArray, pd.DataFrame or np.ndarray
        input data.
    inter_method : str, optional
        inter_method used to interpolate NaNs, build upon
        pd.DataFrame().interpolate(method=inter_method).

        Interpolation technique to use:
        ‘linear’: Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
        ‘time’: Works on daily and higher resolution data to interpolate given length of interval.
        ‘index’, ‘values’: use the actual numerical values of the index.
        ‘pad’: Fill in NaNs using existing values.
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’: Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index. Both ‘polynomial’ and ‘spline’ require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
        ‘krogh’, ‘piecewise_polynomial’, ‘spline’, ‘pchip’, ‘akima’, ‘cubicspline’: Wrappers around the SciPy interpolation methods of similar names. See Notes.
        If str is None of the valid option, will raise an ValueError.
        The default is 'quadratic'.
    order : int, optional
        order of spline fit
    inter_NaN_limit : float, optional
        Limit the % amount of consecutive NaNs for interpolation.
        The default is None (no limit)
    extra_NaN_limit : float, optional
        Limit the % amount of consecutive NaNs for extrapolation using linear method.
    missing_data_ts_to_nan : bool, float, int
        Will mask complete timeseries to np.nan if more then a percentage (if float)
        or more then integer (if int) of NaNs are present in timeseries.
    final_NaN_to_clim : bool (optional)
        If NaNs are still left in data after interpolation, extrapolation and
        masking timeseries completely due to too many NaNs

    Raises
    ------
    ValueError
        If NaNs are not allowed (method=False).

    Returns
    -------
    data : xr.DataArray, pd.DataFrame or np.ndarray
        data with inter- / extrapolated / masked NaNs.

    references:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html

    '''
    # inter_method='spline';order=2;inter_NaN_limit=None;extra_NaN_limit=.05;
    # final_NaN_to_clim=True;missing_data_ts_to_nan=False
    #%%
    data, kwrgs_dtype, input_dtype = to_np(data)

    orig_shape = data.shape
    data = data.reshape(orig_shape[0], -1)

    if type(missing_data_ts_to_nan) is float: # not allowed more then % of NaNs
        NaN_mask = np.isnan(data).sum(0) >= missing_data_ts_to_nan * orig_shape[0]
    elif type(missing_data_ts_to_nan) is int: # not allowed more then int NaNs
        NaN_mask = np.isnan(data).sum(0) >= missing_data_ts_to_nan
    elif missing_data_ts_to_nan == True:
        NaN_mask = np.isnan(data).any(axis=0) # mask if any NaN is present in ts
    else:
        NaN_mask = np.isnan(data).all(axis=0) # mask when all datapoints are NaN
    data[:,NaN_mask] = np.nan

    # interpolating NaNs
    t, o = np.where(np.isnan(data[:,~NaN_mask])) # NaNs some timesteps
    if t.size > 0:

        # interpolation
        print(f'Warning: {t.size} NaNs found at {np.unique(o).size} location(s)',
              '')
        if inter_method is not False:
            print(f'Sparse NaNs will be interpolated using {inter_method} (pandas)\n')
            if inter_NaN_limit is not None:
                limit = int(orig_shape[0]*inter_NaN_limit)
            else:
                limit=None
            try:
                data[:,~NaN_mask] = pd.DataFrame(data[:,~NaN_mask]).interpolate(method=inter_method, limit=limit,
                                                                                limit_direction='both',
                                                                                limit_area='inside',
                                                                                order=order).values
            except:
                print(f'{inter_method} spline gave error, reverting to linear interpolation')
                data[:,~NaN_mask] = pd.DataFrame(data[:,~NaN_mask]).interpolate(method='linear', limit=limit,
                                                                                limit_area='inside',
                                                                                limit_direction='both').values
            ti, _ = np.where(np.isnan(data[:,~NaN_mask]))
            print(f'{t.size - ti.size} values are interpolated')
            if ti.size != 0 and limit is None:
                print('Warning: NaNs left at edges of timeseries')
            elif ti.size !=0 and limit is not None:
                print('Warning: NaNs left. could be interpolation was '
                      'insufficient due to limit, or NaNs are located at edges '
                      'of timeseries')
        else:
            ti, _ = t, o

        # extrapolation (only linear method possible)
        if ti.size != 0 and extra_NaN_limit is not False:
            print(f'Extrapolating up to {int(100*extra_NaN_limit)}% of datapoints'
                  ' using linear method')
            if extra_NaN_limit is not None:
                limit = int(orig_shape[0]*extra_NaN_limit)
            else:
                limit=None
            data[:,~NaN_mask] = pd.DataFrame(data[:,~NaN_mask]).interpolate(method='linear',
                                                                            limit=limit,
                                                            limit_direction='both').values
            te, oe = np.where(np.isnan(data[:,~NaN_mask]))
            print(f'{ti.size - te.size} values are extrapolated')
            if te.size != 0:
                print('Not all NaNs allowed to extrapolate, more then '
                  f'{int(100*extra_NaN_limit)}% of first or last dps are missing')
                print('Warning: extrapolation was insufficient')
        tf, of = np.where(np.isnan(data[:,~NaN_mask]))
        if tf.size != 0:
            print(f'Warning: {tf.size} NaNs left after inter/extrapolation '
                  'and masking.')
            if final_NaN_to_clim:
                print('Since final_NaN_to_clim==True, will fill other '
                      'outlier NaNs with mean over time axis')
                data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
                data[:,NaN_mask] = np.nan # restore always NaN mask
            else:
                print('Since final_NaN_to_clim==False, no other method to '
                      'handle NaNs. Warning: NaNs still present')
        else:
            raise ValueError('NaNs not allowed')

    data = back_to_input_dtype(data.reshape(orig_shape), kwrgs_dtype, input_dtype)
    return data

