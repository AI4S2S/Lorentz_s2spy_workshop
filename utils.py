#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Aug 4 2022

@author: semvijverberg
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
import itertools

flatten = lambda l: list(itertools.chain.from_iterable(l))

def get_oneyr(dt_pdf_pds_xr, *args):
    if type(dt_pdf_pds_xr) == pd.DatetimeIndex:
        pddatetime = dt_pdf_pds_xr
    if type(dt_pdf_pds_xr) == pd.DataFrame or type(dt_pdf_pds_xr) == pd.Series:
        pddatetime = dt_pdf_pds_xr.index # assuming index of df is DatetimeIndex
    if type(dt_pdf_pds_xr) == xr.DataArray:
        pddatetime = pd.to_datetime(dt_pdf_pds_xr.time.values)

    dates = []
    pddatetime = pd.to_datetime(pddatetime)
    year = pddatetime.year[0]

    for arg in args:
        year = arg
        dates.append(pddatetime.where(pddatetime.year.values==year).dropna())
    dates = pd.to_datetime(flatten(dates))
    if len(dates) == 0:
        dates = pddatetime.where(pddatetime.year.values==year).dropna()
    return dates

def get_selbox(ds, selbox, verbosity=0):
    '''
    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    # test selbox assumes [west_lon, east_lon, south_lat, north_lat]
    '''

    except_cross180_westeast = test_periodic(ds)==False and 0 not in ds.longitude

    if except_cross180_westeast:
        # convert selbox to degrees east
        selbox = np.array(selbox)
        selbox[:2][selbox[:2] < 0] += 360
        selbox = list(selbox)

    if ds.latitude[0] > ds.latitude[1]:
        slice_lat = slice(max(selbox[2:]), min(selbox[2:]))
    else:
        slice_lat = slice(min(selbox[2:]), max(selbox[2:]))

    east_lon = selbox[0]
    west_lon = selbox[1]
    if (east_lon > west_lon and east_lon > 180) or (east_lon < 0 and east_lon!=-180):
        if verbosity > 0:
            print('east lon > 180 and cross GW meridional, converting to west '
                  'east longitude format because lons must be sorted by value')
        zz = convert_longitude(ds, to_format='east_west')
        zz = zz.sortby('longitude')
        if east_lon <= 0:
            e_lon =east_lon
        elif east_lon > 180:
            e_lon = east_lon - 360
        ds = zz.sel(longitude=slice(e_lon, west_lon))
    else:
        ds = ds.sel(longitude=slice(east_lon, west_lon))
    ds = ds.sel(latitude=slice_lat)
    return ds

def convert_longitude(data, to_format='west_east'):
    if to_format == 'east_west':
        data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))
    elif to_format == 'only_east':
        data = data.assign_coords(longitude=((data.longitude + 360) % 360))
    return data

def _check_format(ds):
    longitude = ds.longitude.values
    if longitude[longitude > 180.].size != 0:
        format_lon = 'only_east'
    else:
        format_lon = 'west_east'
    return format_lon

def test_periodic(ds):
    dlon = ds.longitude[1] - ds.longitude[0]
    return (360 / dlon == ds.longitude.size).values

def crossing0lon(ds):
    dlon = ds.longitude[1] - ds.longitude[0]
    return ds.sel(longitude=0, method='nearest').longitude < dlon

def formatting_ds(ds, format_lon : str='only_east'):
    
    if 'latitude' and 'longitude' not in ds.dims:
        ds = ds.rename({'lat':'latitude',
                        'lon':'longitude'})
        if 'time' in ds.squeeze().dims and len(ds.squeeze().dims) == 3:
            ds = ds.transpose('time', 'latitude', 'longitude')

    if format_lon is not None:
        if test_periodic(ds)==False and crossing0lon(ds)==False:
            format_lon = 'only_east'
        if _check_format(ds) != format_lon:
            ds = convert_longitude(ds, format_lon)

    # ensure longitude in increasing order
    minidx = np.where(ds.longitude == ds.longitude.min())[0]
    maxidx = np.where(ds.longitude == ds.longitude.max())[0]
    if bool(minidx > maxidx):
        print('sorting longitude')
        ds = ds.sortby('longitude')

    # ensure latitude is in increasing order
    minidx = np.where(ds.latitude == ds.latitude.min())[0]
    maxidx = np.where(ds.latitude == ds.latitude.max())[0]
    if bool(minidx > maxidx):
        print('sorting latitude')
        ds = ds.sortby('latitude')    
    return ds

def view_or_replace_labels(xarr: xr.DataArray, regions: Union[int,list],
           replacement_labels: Union[int,list]=None):
    '''
    View or replace a subset of labels.

    Parameters
    ----------
    xarr : xr.DataArray
        xarray with precursor region labels.
    regions : Union[int,list]
        region labels to select (for replacement).
    replacement_labels : Union[int,list], optional
        If replacement_labels given, should be same length as regions.
        The default is that no labels are replaced.

    Returns
    -------
    xarr : xr.DataArray
        xarray with precursor labels defined by argument regions, if
        replacement_labels are given; region labels are replaced by values
        in replacement_labels.

    '''
    if replacement_labels is None:
        replacement_labels = regions
    if type(regions) is int:
        regions = [regions]
    if type(replacement_labels) is int:
        replacement_labels = [replacement_labels]
    xarr = xarr.copy() # avoid replacement of init prec_labels xarray
    shape = xarr.shape
    df = pd.Series(np.round(xarr.values.flatten(), 0), dtype=float)
    d = dict(zip(regions, replacement_labels))
    out = df.map(d).values
    xarr.values = out.reshape(shape)
    return xarr

def regrid_xarray(xarray_in, to_grid_res, periodic=True):
    import xesmf as xe
    #%%
    '''
    Only supports 2 (lat, lon) or 3 (time, lat, lon) xr.DataArrays
    '''
    method_list = ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']
    method = method_list[0]


    ds = xr.Dataset({'data':xarray_in})
    ds = xarray_in

    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon',
                        'latitude' : 'lat'})

    lats = ds.lat
    lons = ds.lon
    orig_grid = float(abs(ds.lat[1] - ds.lat[0] ))

    if method == 'conservative':
        # add lon_b and lat_b
        lat_b = np.concatenate(([lats.max()+orig_grid/2.], (lats - orig_grid/2.).values))
        lon_b = np.concatenate(([lons.max()+orig_grid/2.], (lons - orig_grid/2.).values))
        ds['lat_b'] = xr.DataArray(lat_b, dims=['lat_b'], coords={'lat_b':lat_b})
        ds['lon_b'] = xr.DataArray(lon_b, dims=['lon_b'], coords={'lon_b':lon_b})

        lat0_b = lat_b.min()
        lat1_b = lat_b.max()
        lon0_b = lon_b.min()
        lon1_b = lon_b.max()
    else:
        lat0_b = lats.min()
        lat1_b = lats.max()
        lon0_b = lons.min()
        lon1_b = lons.max()
    to_grid = xe.util.grid_2d(lon0_b, lon1_b, to_grid_res, lat0_b, lat1_b, to_grid_res)
#    to_grid = xe.util.grid_global(2.5, 2.5)
    try:
        regridder = xe.Regridder(ds, to_grid, method, periodic=periodic, reuse_weights=True)
    except:
        regridder = xe.Regridder(ds, to_grid, method, periodic=periodic, reuse_weights=False)
    try:
        xarray_out = regridder(ds)
    except:
        xarray_out  = regridder.regrid_dataarray(ds)
    regridder.clean_weight_file()
    xarray_out = xarray_out.rename({'lon':'longitude',
                                    'lat':'latitude'})
    if len(xarray_out.shape) == 2:
        xarray_out = xr.DataArray(xarray_out.values[::-1],
                                  dims=['latitude', 'longitude'],
                                  coords={'latitude':xarray_out.latitude[:,0].values[::-1],
                                  'longitude':xarray_out.longitude[0].values})
    elif len(xarray_out.shape) == 3:
        xarray_out = xr.DataArray(xarray_out.values[:,::-1],
                                  dims=['time','latitude', 'longitude'],
                                  coords={'time':xarray_out.time,
                                          'latitude':xarray_out.latitude[:,0].values[::-1],
                                          'longitude':xarray_out.longitude[0].values})
    xarray_out.attrs = xarray_in.attrs
    xarray_out.name = xarray_in.name
    if 'is_DataArray' in xarray_out.attrs:
        del xarray_out.attrs['is_DataArray']
    xarray_out.attrs['regridded'] = f'{method}_{orig_grid}d_to_{to_grid_res}d'
#    xarray_out['longitude'] -= xarray_out['longitude'][0] # changed 17-11-20
    #%%
    return xarray_out

def match_coords_xarrays(wanted_coords_arr, *to_match):
    dlon = float(wanted_coords_arr.longitude[:2].diff('longitude'))
    dlat = float(wanted_coords_arr.latitude[:2].diff('latitude'))
    lonmin = wanted_coords_arr.longitude.min()
    lonmax = wanted_coords_arr.longitude.max()
    latmin = wanted_coords_arr.latitude.min()
    latmax = wanted_coords_arr.latitude.max()
    return [tomatch.sel(longitude=np.arange(lonmin, lonmax+dlon,dlon),
                       latitude=np.arange(latmin, latmax+dlat,dlat),
                       method='nearest') for tomatch in to_match]
