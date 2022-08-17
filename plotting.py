#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Aug 5 2022

@author: semvijverberg
"""

from typing import List, Union
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_maps(corr_xr, mask_xr=None, map_proj=None, row_dim='split',
                   col_dim='lag', clim='relaxed', hspace=-0.6, wspace=0.02,
                   size=2.5, cbar_vert=-0.01, units='units', cmap=None,
                   clevels=None, clabels=None, cticks_center=None,
                   cbar_tick_dict: dict={},
                   kwrgs_cbar = {'orientation':'horizontal', 'extend':'neither'},
                   title=None, title_fontdict: dict=None, subtitles: np.ndarray=None,
                   subtitle_fontdict: dict=None, zoomregion=None,
                   aspect=None, n_xticks=5, n_yticks=3, x_ticks: Union[bool, np.ndarray]=None,
                   y_ticks: Union[bool, np.ndarray]=None, add_cfeature: str=None,
                   col_wrap: int=None,
                   kwrgs_mask: dict={}):

    '''
    zoomregion = tuple(east_lon, west_lon, south_lat, north_lat)
    '''
    #%%
    # default parameters
    # mask_xr=None ; row_dim='split'; col_dim='lag'; clim='relaxed'; wspace=.03;
    # size=2.5; cbar_vert=-0.01; units='units'; cmap=None; hspace=-0.6;
    # clevels=None; clabels=None; cticks_center=None; cbar_tick_dict={}; map_proj=None ;
    # drawbox=None; subtitles=None; title=None; lat_labels=True; zoomregion=None
    # aspect=None; n_xticks=5; n_yticks=3; title_fontdict=None; x_ticks=None;
    # y_ticks=None; add_cfeature=None; col_wrap=None
    # kwrgs_cbar = {'orientation':'horizontal', 'extend':'neither'}

    if map_proj is None:
        cen_lon = int(corr_xr.longitude.mean().values)
        map_proj = ccrs.LambertCylindrical(central_longitude=cen_lon)

    if row_dim not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims(row_dim, 0)
        if mask_xr is not None and row_dim not in mask_xr.dims:
            mask_xr = mask_xr.expand_dims(row_dim, 0)
    if col_dim not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims(col_dim, 0)
        if mask_xr is not None and col_dim not in mask_xr.dims:
            mask_xr = mask_xr.expand_dims(col_dim, 0)

    var_n   = corr_xr.name
    rows    = corr_xr[row_dim].values
    cols    = corr_xr[col_dim].values

    rename_dims = {row_dim:'row', col_dim:'col'}
    rename_dims_inv = {'row':row_dim, 'col':col_dim}
    plot_xr = corr_xr.rename(rename_dims)
    if mask_xr is not None:
        plot_mask = mask_xr.rename(rename_dims)
    dim_coords = plot_xr.squeeze().dims
    dim_coords = [d for d in dim_coords if d not in ['latitude', 'longitude']]
    rename_subs = {d:rename_dims_inv[d] for d in dim_coords}

    lat = plot_xr.latitude
    lon = plot_xr.longitude
    zonal_width = abs(lon[-1] - lon[0]).values
    if aspect is None:
        aspect = (lon.size) / lat.size

    if col_wrap is None:
        g = xr.plot.FacetGrid(plot_xr, col='col', row='row',
                              subplot_kws={'projection': map_proj},
                              sharex=True, sharey=True,
                              aspect=aspect, size=size)
    else:
        g = xr.plot.FacetGrid(plot_xr, col='col',
                      subplot_kws={'projection': map_proj},
                      sharex=True, sharey=True,
                      aspect=aspect, size=size,
                      col_wrap=col_wrap)
    figheight = g.fig.get_figheight()
    g.fig.subplots_adjust(hspace=hspace, wspace=wspace)
    # =============================================================================
    # Coordinate labels
    # =============================================================================
    import cartopy.mpl.ticker as cticker
    g.set_ticks(fontsize='large')
    if x_ticks is None or x_ticks is False: #auto-ticks, if False, will be masked
        longitude_labels = np.linspace(np.min(lon), np.max(lon), n_xticks, dtype=int)
        longitude_labels = np.array(sorted(list(set(np.round(longitude_labels, -1)))))
    else:
        longitude_labels = x_ticks # if x_ticks==False -> no ticklabels
    if y_ticks is None or y_ticks is False: #auto-ticks, if False, will be masked
        latitude_labels = np.linspace(lat.min(), lat.max(), n_yticks, dtype=int)
        latitude_labels = sorted(list(set(np.round(latitude_labels, -1))))
    else:
        latitude_labels = y_ticks # if y_ticks==False -> no ticklabels

    # =============================================================================
    # clevels and colormap
    # =============================================================================
    if clevels is None:
        vmin_ = np.nanpercentile(plot_xr, 1) ; vmax_ = np.nanpercentile(plot_xr, 99)
        vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    else:
        vmin_ = np.nanpercentile(plot_xr, 1) ; vmax_ = np.nanpercentile(plot_xr, 99)
        vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
        clevels=clevels

    if cmap is None:
        cmap = plt.cm.RdBu_r
    else:
        cmap=cmap
    # =============================================================================
    # loop over subplots in Facetgrid
    # =============================================================================
    for col, c_label in enumerate(cols):
        xrdatavar = plot_xr.sel(col=c_label)
        dlon = abs(lon[1] - lon[0])
        if abs(lon[-1] - 360) <= dlon and lon[0] < dlon:
            xrdatavar = extend_longitude(xrdatavar)

        for row, r_label in enumerate(rows):
            if col_wrap is not None:
                row = np.repeat(list(range(g.axes.shape[0])), g.axes.shape[1])[col]
                col = (list(range(col_wrap))*g.axes.shape[0])[col]

            print(f"\rPlotting Corr maps {var_n}, {row_dim} {r_label}, {col_dim} {c_label}", end="\n")
            plotdata = xrdatavar.sel(row=r_label).rename(rename_subs).squeeze()
            # =============================================================================
            # Plot contour mask
            # =============================================================================         
            if mask_xr is not None:
                xrmaskvar = plot_mask.sel(col=c_label)
                if abs(lon[-1] - 360) <= (lon[1] - lon[0]) and lon[0]==0:
                    xrmaskvar = extend_longitude(xrmaskvar)
                plotmask = xrmaskvar.sel(row=r_label)
                _kwrgs_mask = {'linestyles':['solid'],
                              'colors':['black'],
                              'linewidths':np.round(zonal_width/150, 1)+0.3}
                _kwrgs_mask.update(kwrgs_mask)
                # field not completely masked?
                all_masked = (plotmask.values==False).all()
                if all_masked == False:
                    # if plotdata is already masked (with nans):
                    p_nans = int(100*plotdata.values[np.isnan(plotdata.values)].size / plotdata.size)
                    if p_nans != 100:
                        plotmask.plot.contour(ax=g.axes[row,col],
                                              transform=ccrs.PlateCarree(),
                                              levels=[float(vmin),float(vmax)],
                                              add_colorbar=False,
                                              **_kwrgs_mask)
            # =============================================================================
            # Plot colourmap
            # =============================================================================  
            # if no signifcant regions, still plot corr values, but the causal plot must remain empty
            if mask_xr is None or all_masked==False or (all_masked and 'tigr' not in str(c_label)):
                im = plotdata.plot.pcolormesh(ax=g.axes[row,col], transform=ccrs.PlateCarree(),
                                              center=0, levels=clevels,
                                              cmap=cmap,add_colorbar=False)
            elif all_masked and 'tigr' in c_label:
                g.axes[row,col].text(0.5, 0.5, 'No regions significant',
                      horizontalalignment='center', fontsize='x-large',
                      verticalalignment='center', transform=g.axes[row,col].transAxes)
            # =============================================================================
            # Subtitles
            # =============================================================================
            if subtitles is not None:
                if subtitle_fontdict is None:
                    subtitle_fontdict = dict({'fontsize' : 16})
                if subtitles is not False:
                    subtitle = np.array(subtitles)[row,col]
                else:
                    subtitle  = ''
                g.axes[row,col].set_title(subtitle,
                                          fontdict=subtitle_fontdict,
                                          loc='center')
            # =============================================================================
            # Format coordinate ticks
            # =============================================================================
            if map_proj.proj4_params['proj'] in ['merc', 'eqc', 'cea']:
                ax = g.axes[row,col]
                # x-ticks and labels
                ax.set_xticks(longitude_labels[:], crs=ccrs.PlateCarree())
                if x_ticks is not False:
                    ax.set_xticklabels(longitude_labels[:], fontsize=12)
                    lon_formatter = cticker.LongitudeFormatter()
                    ax.xaxis.set_major_formatter(lon_formatter)
                else:
                    fake_labels = [' ' * len( str(l) ) for l in longitude_labels]
                    g.axes[row,col].set_xticklabels(fake_labels, fontsize=12)
                # y-ticks and labels
                g.axes[row,col].set_yticks(latitude_labels, crs=ccrs.PlateCarree())
                if y_ticks is not False:
                    g.axes[row,col].set_yticklabels(latitude_labels, fontsize=12)
                    lat_formatter = cticker.LatitudeFormatter()
                    g.axes[row,col].yaxis.set_major_formatter(lat_formatter)
                else:
                    fake_labels = [' ' * len( str(l) ) for l in latitude_labels]
                    g.axes[row,col].set_yticklabels(fake_labels, fontsize=12)
            # =============================================================================
            # Gridlines
            # =============================================================================
                if type(y_ticks) is bool and type(x_ticks) is bool:
                    if np.logical_and(y_ticks==False, x_ticks==False):
                        # if no ticks, then also no gridlines
                        pass
                else:
                    gl = g.axes[row,col].gridlines(crs=ccrs.PlateCarree(),
                                              linewidth=.5, color='black', alpha=0.15,
                                              linestyle='--', zorder=4)
                    gl.xlocator = mticker.FixedLocator((longitude_labels % 360 + 540) % 360 - 180)
                    gl.ylocator = mticker.FixedLocator(latitude_labels)
                g.axes[row,col].set_ylabel('')
                g.axes[row,col].set_xlabel('')
                
            g.axes[row,col].coastlines(color='black', alpha=0.3, linewidth=2, facecolor='white')
            # black outline subplot
            g.axes[row,col].spines['geo'].set_edgecolor('black')
            if corr_xr.name is not None:
                if corr_xr.name[:3] == 'sst':
                    g.axes[row,col].add_feature(cfeature.LAND, facecolor='grey',
                                                alpha=0.1, zorder=0)
            if add_cfeature is not None:
                g.axes[row,col].add_feature(cfeature.__dict__[add_cfeature],
                                            facecolor='white', alpha=0.1,
                                            zorder=4)
            if zoomregion is not None:
                g.axes[row,col].set_extent(zoomregion, crs=ccrs.PlateCarree())
            else:
                g.axes[row,col].set_extent([lon[0], lon[-1],
                                       lat[0], lat[-1]], crs=ccrs.PlateCarree())
    # =============================================================================
    # lay-out settings FacetGrid and colorbar
    # =============================================================================
    # height colorbor 1/10th of height of subfigure
    height = g.axes[-1,0].get_position().height / 10
    bottom_ysub = (figheight/40)/(rows.size*2) + cbar_vert
    cbar_ax = g.fig.add_axes([0.25, bottom_ysub,
                              0.5, height]) #[left, bottom, width, height]

    if units == 'units' and 'units' in corr_xr.attrs:
        clabel = corr_xr.attrs['units']
    elif units != 'units' and units is not None:
        clabel = units
    else:
        clabel = ''

    if cticks_center is None:
        if clabels is None:
            clabels = clevels[::2]
        plt.colorbar(im, cax=cbar_ax,
                     label=clabel, ticks=clabels, **kwrgs_cbar)
    else:
        cbar = plt.colorbar(im, cbar_ax, label=clabel, **kwrgs_cbar)
        cbar.set_ticks(clevels[:-1] + 0.5)
        cbar.set_ticklabels(np.array(clevels[1:], dtype=int))
    cbar_ax.tick_params(**cbar_tick_dict)

    if title is not None:
        if title_fontdict is None:
            title_fontdict = dict({'fontsize'     : 18,
                                   'fontweight'   : 'bold'})
        g.fig.suptitle(title, **title_fontdict)
    return g

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def extend_longitude(data):
    plottable = xr.concat([data, data.sel(longitude=data.longitude[:1])], dim='longitude').to_dataset(name="ds")
    plottable["longitude"] = np.linspace(0,360, len(plottable.longitude))
    plottable = plottable.to_array(dim='ds').squeeze(dim='ds').drop_vars('ds')
    return plottable

def plot_labels(prec_labels,
                kwrgs_plot={},
                labelsintext=False):

    xrlabels = prec_labels.copy()
    # if 'cmap' not in kwrgs_plot.keys():
    #     # default cmap is plt.cm.tab20, which does not have more than 20 colours
    #     xrlabels = xrlabels.where(~(xrlabels.values>20))

    kwrgs_labels = _get_kwrgs_labels(xrlabels, kwrgs_plot, labelsintext)
    xrlabels.values = prec_labels.values - 0.5
    return plot_maps(xrlabels, **kwrgs_labels)

def _get_kwrgs_labels(prec_labels, kwrgs_plot={}, labelsintext=False):

    # default dims such that I can use dims to ensure position textinmap
    if 'row_dim' not in kwrgs_plot.keys():
        kwrgs_plot['row_dim'] = 'split'
    if 'col_dim' not in kwrgs_plot.keys():
        kwrgs_plot['col_dim'] = 'lag'

    kwrgs_labels = {'size':3, 'cticks_center':True, 'units': None}
    if labelsintext:
        textinmap = []
        min_lat = float(np.min(prec_labels.latitude))
        max_lat = float(np.max(prec_labels.latitude))
        spatdim = ['latitude', 'longitude', 'lat', 'lon', 'mask']
        dims = [d for d in prec_labels.dims if d not in spatdim]
        coords = [list(np.array(prec_labels[d], dtype=str)) for d in dims]
        if len(coords) == 1:
            coords.append(['fake'])
        elif len(coords) == 0:
            coords = ['fake', 'fake'] ; dims = ['fake']
        combs = np.array(np.meshgrid(coords[0], coords[1])).T.reshape(-1,2)



        for i, (c1, c2) in enumerate(combs):
            idx1 = coords[0].index(c1)
            if c2 != 'fake':
                idx2 = coords[1].index(c2)
                labelsmap = prec_labels[idx1, idx2]
            elif c1 != 'fake' and c2 == 'fake':
                idx2 = 0
                labelsmap = prec_labels[idx1]
            else:
                idx1 = 0; idx2 = 0
                labelsmap = prec_labels

            df_labelloc = labels_to_df(labelsmap,
                                       return_mean_latlon=True)

            labels = np.unique(labelsmap)
            labels = labels[~np.isnan(labels)]


            if kwrgs_plot['col_dim'] == dims[0]:
                rowdim = (idx2, idx1)
            else:
                rowdim = (idx1, idx2)

            temp = []
            for q, l in enumerate(labels):
                if l == 0: # pattern cov
                    lat, lon = df_labelloc.mean(0)[:2]
                else:
                    lat, lon = df_labelloc.loc[l].iloc[:2].values.round(1)
                if lon > 180: lon-360
                temp.append([lon,max(min_lat,min(max_lat,lat)),
                             str(int(l)),
                             {'fontsize':10}]),
                              # 'bbox':dict(facecolor='pink', alpha=0.01)}])
                textinmap.append([rowdim, temp])
        kwrgs_labels['textinmap'] = textinmap

    if np.isnan(prec_labels.values).all() == False:
        max_N_regs = min(20, int(prec_labels.max() + 0.5))
    else:
        max_N_regs = 20
    label_weak = np.nan_to_num(prec_labels.values) >=  max_N_regs

    prec_labels.values[label_weak] = max_N_regs
    steps = max_N_regs+1
    prec_labels.values = prec_labels.values-0.5
    clevels = np.linspace(0, max_N_regs,steps)

    if 'cmap' not in kwrgs_plot:
        cmap = plt.cm.tab20
    else:
        cmap = kwrgs_plot['cmap']

    kwrgs_labels.update({'clevels':clevels,
                         'cmap':cmap})

    if len(prec_labels.shape) == 2 or prec_labels.shape[0] == 1:
        kwrgs_labels['cbar_vert'] = -0.1

    kwrgs_labels.update(kwrgs_plot)

    return kwrgs_labels

def labels_to_df(prec_labels, return_mean_latlon=True):
    dims = [d for d in prec_labels.dims if d not in ['latitude', 'longitude']]
    df = prec_labels.mean(dim=tuple(dims)).to_dataframe().dropna()
    label_coord = [c for c in df.columns if 'label' in c][0]
    if return_mean_latlon:
        labels = np.unique(prec_labels)[~np.isnan(np.unique(prec_labels))]
        mean_coords_area = np.zeros( (len(labels), 3))
        for i,l in enumerate(labels):
            latlon = np.array(df[(df[label_coord]==l).values].index)
            latlon = np.array([list(l) for l in latlon])
            if latlon.size != 0:
                mean_coords_area[i][:2] = np.median(latlon, 0)
                mean_coords_area[i][-1] = latlon.shape[0]
        df = pd.DataFrame(mean_coords_area, index=labels,
                     columns=['latitude', 'longitude', 'n_gridcells'])
    return df