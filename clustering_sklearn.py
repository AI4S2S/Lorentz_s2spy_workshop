# -*- coding: utf-8 -*-
"""
Created on Wed Aug 4 2022

@author: semvijverberg
"""
from typing import Optional, Tuple, List, Dict # for nice documentation
import numpy as np
import sklearn.cluster as cluster
import xarray as xr

def skclustering(time_space_3d: xr.DataArray, spatial_mask: Optional[np.ndarray] = None,
                 clustermethodkey: str = 'AgglomerativeClustering',
                 kwrgs: Dict ={'n_clusters':4}, dimension: str = 'temporal'):
    '''
    Is build upon sklearn clustering. Algorithms available are listed in cluster.__dict__,
    e.g. KMeans, or AgglomerativeClustering, kwrgs are algorithms dependend.

    Parameters
    ----------
    time_space_3d : xarray.DataArray
        input xarray with observations to be clustered, must contain only one variable

    spatial_mask : 2-d numpy.ndarray, optional
        mask, only coordinates == True will taken into account by the clustered algorithm, 
        the default is None (no mask).

    clustermethodkey : str
        name of a sklearn.cluster algorithm

    kwrgs : dict
        (algorithm dependent) dictionary of clustering parameters

    dimension: str
        clustering dimension, "temporal" or "spatial"

    Returns
    -------
    if dimension == 'temporal':
        xrclustered: xarray.DataArray with with additional coordinate 'cluster' attached to time for coordinates in mask 2d (temporal)
    elif dimension == 'spatial':
        xrclustered: xarray.DataArray with clustering labels as values for coordinates in mask 2d (spatial)
    results: sklearn.cluster instance
    '''

    # ensure that the number of clusters is an integer
    if 'n_clusters' in kwrgs.keys():
        assert isinstance(kwrgs['n_clusters'], int), 'Number of clusters is not an integer'
        
    if dimension not in ['temporal', 'spatial']:
        raise ValueError("dimension should be 'temporal' or 'spatial'.")
    
    # initialize algorithm from sklearn.cluster
    algorithm = cluster.__dict__[clustermethodkey]
    cluster_method = algorithm(**kwrgs)
    
    # reshape time,lat,lon xarray (taking into account spatial_mask).
    space_time_vec, output_space_time, indices_mask = create_vector(time_space_3d, spatial_mask)

    if dimension == 'temporal':
        results = cluster_method.fit(space_time_vec.swapaxes(0, 1))
        labels = results.labels_ + 1
        # assigning cluster label to time dimension (now cluster dimension is attached to time)
        xrclustered = time_space_3d.assign_coords(cluster = ("time", labels))
    elif dimension == 'spatial':
        results = cluster_method.fit(space_time_vec)
        labels = results.labels_ + 1
        xrclustered = labels_to_latlon(time_space_3d, labels, output_space_time, indices_mask, spatial_mask)
    return xrclustered, results

def create_vector(time_space_3d : xr.DataArray, spatial_mask: Optional[np.ndarray] = None):
    """
    Converts time, lat, lon xarray object to (space, time) shape.
    """
    if spatial_mask is None: # no mask will be applied, all values set to True.
        spatial_mask = time_space_3d.isel(time=0).copy()
        spatial_mask.values = np.ones_like(spatial_mask)

    time_space_3d = time_space_3d.where(spatial_mask == True)

    # create mask for to-be-clustered time_space_3d
    n_space = time_space_3d.longitude.size*time_space_3d.latitude.size #

    # reshape 2d mask into 1d mask; 1d numpy array
    mask_1d = np.reshape( spatial_mask, (1, n_space))

    # create numpy array with each entry as a list; e.g. [[1], [2], [3]]
    mask_1d = np.swapaxes(mask_1d, 1,0 )

    # Construct an array by repeating each element of mask_1d for each datetime of the data; e.g. [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mask_space_time = np.array(np.tile(mask_1d, (1,time_space_3d.time.size)), dtype=int)

    # track location of mask to store output; take only the first element of each entry; e.g. [1, 2, 3]
    output_space_time = np.array(mask_space_time[:,0].copy(), dtype=int)

    # find indices where mask has value 1
    indices_mask = np.argwhere(mask_space_time[:,0] == 1)[:,0]

    # convert all space_time_3d gridcells to time_space_2d_all
    # create numpy array with TV values with shape (time.size, longitude.size*latitude.size)
    time_space_2d_all = np.reshape( time_space_3d.values,
                                   (time_space_3d.time.size, n_space) )

    # create numpy array with TV values with shape (longitude.size*latitude.size, time.size)
    space_time_2d_all = np.swapaxes(time_space_2d_all, 1,0)

    # only keep the mask gridcells for clustering
    space_time_2d = space_time_2d_all[mask_space_time == 1]
    space_time_vec = space_time_2d.reshape( (indices_mask.size, time_space_3d.time.size)  ) # shape(# 1 in mask, time.size)
    space_time_vec[np.isnan(space_time_vec)] = -32767.0 #replace nans
    return space_time_vec, output_space_time, indices_mask

def labels_to_latlon(time_space_3d : xr.DataArray, labels : np.ndarray, 
                     output_space_time : np.ndarray, indices_mask : np.ndarray, 
                     spatial_mask : np.ndarray):
    '''
    Translates observations into clustering labels on a spatial mask

    Parameters
    ----------
    time_space_3d : xarray.DataArray
        input xarray with observations to be clustered, must contain only one variable

    labels : 1-d numpy.ndarray
        clustering labels returned by sklearn.clustering algorithm

    output_space_time : 1-d numpy.ndarray
        1-d mask array, size is the number of spatial points (time_space_3d.longitude*time_space_3d.latitude)

    indices_mask : 1-d numpy.ndarray
        1-d array with indices of the mask

    spatial_mask : 2-d numpy.ndarray
        mask with spatial coordinates to be clustered

    Returns
    -------
    xrspace : xarray.DataArray
        xarray with clustering labels as values, spatial points that are now in the mask are asigned value NaN
    '''
    # apply mask if given
    if spatial_mask is None: # no mask will be applied, all values set to True.
        spatial_mask = time_space_3d.isel(time=0).copy()
        spatial_mask.values = np.ones_like(spatial_mask)

    # chooses all coordinates for the first datetime of the data
    xrspace = time_space_3d[0].copy()

    # only choose those coordinate for which mask is 1
    output_space_time[indices_mask] = labels
    # array with dim (#lat, #lon):
    output_space_time = output_space_time.reshape((time_space_3d.latitude.size, time_space_3d.longitude.size)) 

    # add data to xarray
    xrspace.values = output_space_time
    xrspace = xrspace.where(spatial_mask==True)
    return xrspace

def binary_occurences_quantile(xarray, q=95):
    '''
    creates binary occuences of 'extreme' events defined as exceeding or being below the qth percentile
    '''
    assert type(q) is int, ('q should be integer between 1 and 99')
    
    np.warnings.filterwarnings('ignore')
    perc = xarray.reduce(np.percentile, dim='time', keep_attrs=True, q=q)
    rep_perc = np.tile(perc, (xarray.time.size,1,1))
    if q >= 50:
        indic = xarray.where(np.squeeze(xarray.values) >= rep_perc)
    elif q < 50: 
        indic = xarray.where(np.squeeze(xarray.values) < rep_perc)
    indic.values = np.nan_to_num(indic)
    indic.values[indic.values > 0 ] = 1
    return indic