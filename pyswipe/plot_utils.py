""" 
This module contains tools for plotting certain quantities in polar coordinates
(mlt/mlat grid).

Example:
--------
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)   
pax = Polarsubplot(ax)
pax.plot(np.ones(20) * 60, np.linspace(18, 24+6, 20), color = 'red', linewidth = 5)
pax.scatter([70, 70], [15, 9], s = 200, c = 'red')
plt.show()

In addition to `scatter` and `plot`, it is also possible to use matplotlib's 
`contour` and `contourf` functions with mlt/mlat coordinates. A function 
`featherplot` is also added for plotting vectors. A function `write` is also 
provided which replicates the beahaviour of matplotlib's `text`.



MIT License

Copyright (c) 2017 Karl M. Laundal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from __future__ import absolute_import, division
import numpy as np
from scipy.interpolate import griddata
from builtins import range
from matplotlib.collections import LineCollection


class Polarsubplot(object):
    def __init__(self, ax, minlat = 60, plotgrid = True, **kwargs):
        """ class which can be used for easy plotting in polar coordinates (mlt/mlat)

            Parameters
            ----------
            ax : matplotlib.axes._subplots.AxesSubplot
                matplotlib axes object which to plot 
            minlat : int (optional)
                minimum latitude of the plot. Should be positive, so if you want
                to plot in the southern hemisphere, convert mlats to positive and 
                just change the labels. Default 60 degrees
            plotgrid : bool (optional)
                whether or not to plot a grid. Default is true
            **kwargs : dict (optional)
                keyword parameters that are passed to matplotlib's `plot` when
                plotting the grid (if `plotgrid` == True). By default, the grid 
                will have linestyle ':' and color 'lightgray'.


            Example
            -------
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)   
            pax = Polarsubplot(ax)
            pax.METHOD(...)
            plt.show()

            where `METHOD` is one of the following:

            Methods
            -------
            plot(mlat, mlt, **kwargs) 
                works like `matplotlib`'s `plot`, except that it uses mlat/mlt
            write(mlat, mlt, text, **kwargs)
                works like `matplotlib`'s `text`, except that it uses mlat/mlt
            scatter(mlat, mlt, **kwargs)    
                works like `matplotlib`'s `scatter`, except that it uses mlat/mlt
            contour(mlat, mlt, f, **kwargs)
                works like `matplotlib`'s `contour`, except that it uses mlat/mlt.
                (note that it uses `scipy.interpolate.griddata` to interpolate,
                which may give some unexpected behaviour)
            contourf(mlat, mlt, f, **kwargs)
                works like `matplotlib`'s `contour`, except that it uses mlat/mlt.
                (note that it uses `scipy.interpolate.griddata` to interpolate,
                which may give some unexpected behaviour)
            featherplot(mlats, mlts, north, east, ...)
                for plotting vector fields, where the vectors are represented by
                a circular marker and a line (see funciton docstring for more details)
    """

        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.minlat = minlat

        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_axis_off()

        if plotgrid:
            self.plotgrid(**kwargs)

    def plot(self, mlat, mlt, **kwargs):
        """ plot curve based on mlat, mlt. **kwargs are passed to `matplotlib`'s `plot`. """

        x, y = self._mlat_mlt_to_xy(mlat, mlt)
        return self.ax.plot(x, y, **kwargs)

    def write(self, mlat, mlt, text, **kwargs):
        """ write text on specified mlat, mlt. **kwargs are passed to `matplotlib`'s `text`"""
        
        x, y = self._mlat_mlt_to_xy(mlat, mlt) 
        return self.ax.text(x, y, text, **kwargs)

    def scatter(self, mlat, mlt, **kwargs):
        """ scatterplot using mlat mlt. **kwargs go to `matplotlib`'s `scatter` """

        x, y = self._mlat_mlt_to_xy(mlat, mlt)
        return self.ax.scatter(x, y, **kwargs)

    def plotgrid(self, **kwargs):
        """ plot mlt, mlat-grid """

        # set default linestyle and color
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = ':'

        if 'color' not in kwargs.keys():
            kwargs['color'] = 'lightgray'

        self.ax.plot([-1, 1], [0 , 0], **kwargs)
        self.ax.plot([0, 0], [-1, 1] , **kwargs)

        latgrid = (90 - np.r_[self.minlat:90:10])/(90. - self.minlat)

        angles = np.linspace(0, 2*np.pi, 360)

        for lat in latgrid:
            self.ax.plot(lat*np.cos(angles), lat*np.sin(angles), **kwargs)

    def writeMLTlabels(self, mlat = 60, degrees = False, **kwargs):
        """ write MLT labels at given latitude 
            if degrees is true, the longitude will be written instead of hour (with 0 at midnight)
        """

        if degrees:
            self.write(mlat, 0,    '0$^\circ$', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            self.write(mlat, 6,   '90$^\circ$', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs) 
            self.write(mlat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs)
            self.write(mlat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs)            
        else:
            self.write(mlat, 0,  '00', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            self.write(mlat, 6,  '06', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs) 
            self.write(mlat, 12, '12', verticalalignment = 'bottom' , horizontalalignment = 'center', **kwargs)
            self.write(mlat, 18, '18', verticalalignment = 'center' , horizontalalignment = 'right' , **kwargs)

    def featherplot(self, mlats, mlts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', color = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ Plot a vector field

            Parameters
            ----------
            mlats : array
                array of latitudes (degrees) describing the location of the vector
            mlts : array
                array of magnetic local times (hours) describing the location of the vector.
                Must have same number of elements as mlats
            north : array
                Array of northward components of the vectors.
                Must have same number of elements as mlats
            east : array
                Array of eastward components of the vectors.
                Must have same number of elements as mlats
            rotation : scalar (optional)
                Number which describes a rotation to be applied to each vector. Default is zero.
                This may be useful when plotting equivalent currents based on magnetic field measurements, 
                SuperMAG style.
            SCALE : number (optional)
                This number determines the length of the vectors, AND whether or not a reference 
                vector shall be shown in the top right corner of the plot. By default, it is not, and the 
                scale is 1, which means that vectors that have length 1 will have a sensible length on 
                the plot. Larger SCALE leads to shorter vectors.
            size : int (optional)
                Font size for the unit (if set)
            unit : string (optional)
                Unit of the vector, which will be written beside the reference vector. Default is
                an empty string
            color : string (optional)
                color of the vector lines. Default is 'black'
            markercolor : string (optional)
                color of the markers at the vector bases. Default 'black'
            marker: string (optional)
                the marker used for vector bases. Default is 'o' (see `matplotlib` `scatter` for 
                other options)
            markersize: int (optional)
                size of the markers. Default is 20.
            **kwargs : dict (optional)
                keywords passed to `matplotlib` `add_collection`

        """

        mlts = mlts.flatten()
        mlats = mlats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                self.ax.plot([0.9, 1], [0.95, 0.95], color = color, linestyle = '-', linewidth = 2)
                self.ax.text(0.9, 0.95, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size)

            scale = 0.1/SCALE

        # segments = []
        for i in range(len(mlats)):

            mlt = mlts[i]
            mlat = mlats[i]

            x, y = self._mlat_mlt_to_xy(mlat, mlt)
            dx, dy = R.dot(self._north_east_to_cartesian(north[i], east[i], mlt).reshape((2, 1))).flatten()

            # segments.append([(x, y), (x + dx*scale, y + dy*scale)])
            self.ax.plot([x, x + dx*scale], [y, y + dy*scale], color = color, **kwargs)
            if markersize != 0:
                self.ax.scatter(x, y, marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)

                
        # self.ax.add_collection(LineCollection(segments, colors = color, **kwargs))

        # if markersize != 0:
        #     self.scatter(mlats, mlts, marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


    def contour(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are passed to `matplotlib`'s `contour` """

        xea, yea = self._mlat_mlt_to_xy(mlat.flatten(), mlt.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contour(xx, yy, gridf, **kwargs)


    def contourf(self, mlat, mlt, f, **kwargs):
        """ plot filled contour on grid, **kwargs are passed to `matplotlib`'s `contourf` """

        xea, yea = self._mlat_mlt_to_xy(mlat.flatten(), mlt.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contourf(xx, yy, gridf, **kwargs)


    def _mlat_mlt_to_xy(self, mlat, mlt):
        """ convert mlt and mlat to x and y """
        r = (90. - np.abs(mlat))/(90. - self.minlat)
        a = (np.array(mlt) - 6.)/12.*np.pi

        return r*np.cos(a), r*np.sin(a)

    def _xy_to_mlat_mlt(self, x, y):
        """ convert x, y to mlt, mlat """
        x, y = np.array(x, ndmin = 1), np.array(y, ndmin = 1) # conver to array to allow item assignment

        lat = 90 - np.sqrt(x**2 + y**2)*(90. - self.minlat)
        mlt = np.arctan2(y, x)*12/np.pi + 6
        mlt[mlt < 0] += 24
        mlt[mlt > 24] -= 24

        return lat, mlt


    def _north_east_to_cartesian(self, north, east, mlt):
        """ convert north, east to cartesian (adapted to plot axis) """
        a = (np.array(mlt) - 6)/12*np.pi # convert MLT to angle with x axis (pointing from pole towards dawn)
        
        x1 = np.array([-north*np.cos(a), -north*np.sin(a)]) # arrow direction towards origin (northward)
        x2 = np.array([-east*np.sin(a),  east*np.cos(a)])   # arrow direction eastward

        return x1 + x2


def equal_area_grid(dr = 2, K = 0, M0 = 8, N = 20):
    """ function for calculating an equal area grid in polar coordinates

    Parameters
    ----------
    dr : float (optional)
        The latitudinal resolution of the grid. Default 2 degrees
    K : int (optional)
        This number determines the colatitude of the inner radius of the
        post poleward ring of grid cells (the pole is not inlcluded!). 
        It relates to this colatitude (r0) as r0/dr = (2K + 1)/2 => K = (2r0/dr - 1)/2.
        Default value is 0
    M0 : int (optional)
        The number of sectors in the most poleward ring. Default is 8
    N : int (optional)
        The number of rings to be included. This determiend how far 
        equatorward the grid extends. Typically if dr is changed from 2 to 1, 
        N should be doubled to reach the same latitude. Default is 20, which means
        that the equatorward edge of the grid is 89 - 20*2 = 49 degrees
        (the most poleward latitude is 89 with default values)

    Returns
    -------
    mlat : array
        Array of latitudes for the equatorward west ("lower left") corners of the grid
        cells.
    mlt : array
        Array of magnetic local times for the equatorward west ("lower left") corner
        of the grid cells.
    mltres : array
        width, in magnetic local time, of the cells with lower left corners described 
        by mlat and mlt. Notice that this width changes with latitude, while the 
        latitudinal width is fixed, determined by the `dr` parameter

    
    """

    r0 = dr * (2*K + 1)/2.

    assert M0 % (K + 1) == 0 # this must be fulfilled

    grid = {}

    M = M0
    grid[90 - r0 - dr] = np.linspace(0, 24 - 24./M, M) # these are the lower limits in MLT

    for i in range(N - 1):

        M = int(M *  (1 + 1./(K + i + 1.))) # this is the partion for i + 1

        grid[90 - (r0 + (i + 1)*dr) - dr] = np.linspace(0, 24 - 24./M, M) # these are the lower limits in MLT

    mlats = []
    mlts = []
    mltres = []
    for key in sorted(grid.keys()):
        mltres_ = sorted(grid[key])[1] - sorted(grid[key])[0]
        for mlt in sorted(grid[key]):
            mlats.append(key)
            mlts.append(mlt)
            mltres.append(mltres_)

    return np.array(mlats), np.array(mlts), np.array(mltres)


def get_h2d_bin_areas(minlats, maxlats, minlons, maxlons, haversine=True,
                      rearth=6.370949e3,
                      altitude=0,
                      spherical_rectangle=True,
                      do_extra_width_calc=True):
    # +
    # NAME:                        GET_H2D_BIN_AREAS
    #
    # PURPOSE:                     And now you wanna know the area, too?!?!?!
    #                              Well, you can have it--in km^2
    # CATEGORY:
    #
    # MODIFICATION HISTORY:      2016/03/12 Barnebarn
    #
    # Notáte bien—se dan las áreas en kilometros cuadrados
    # -

    ## BINEDGE1=Binedge1, BINEDGE2=Binedge2

    # breakpoint()

    # rearth = 6.370949e3+altitude
    # print("rearth:", rearth)

    # print("Want more precision? Try https://math.stackexchange.com/questions/1205927/how-to-calculate-the-area-covered-by-any-spherical-rectangle")
    areas = latlong_pair_area(
        minlons, minlats, maxlons, maxlats, rearth=rearth, haversine=haversine,
        altitude=altitude,
        spherical_rectangle=spherical_rectangle,
        do_extra_width_calc=do_extra_width_calc)  # use radius at 100 km

    return areas


def latlong_pair_area(lon1, lat1,
                      lon2, lat2,
                      rearth=6.370949e3,
                      altitude=0,
                      haversine=True,
                      spherical_rectangle=True,
                      do_extra_width_calc=True):
    """
    Get area (in km^2) of a rectangle on a sphere defined by two lat/lon pairs
    """
    # 2016/03/12 Spence needs this
    # Example: Surf. Area of Utah in two squares
    # ##:Upper square:
    # lonLat1  = [-114.043579, 41.951320]
    # lonLat2  = [-111.055298, 41.021355]
    # upArea   = latlong_pair_area(lonLat1[0],lonLat1[1],lonLat2[0],lonLat2[1])
    #
    # ##:Lower square:
    # lonLat1  = [-114.043579, 41.021355]
    # lonLat2  = [-109.046173, 37.009133]
    # downArea = latlong_pair_area(lonLat1[0],lonLat1[1],lonLat2[0],lonLat2[1])
    #
    # uparea+downarea = 212565.86 km^2
    # actual area of Utah = 84899 sq. mi. * (1.60934 km/mi)^2 = 219886.3 sq. km.
    # Error = 3.33% (7320 sq. km.). Not bad!

    # rearth += altitude

    height = geo_dist(lon1, lat1,
                      lon1, lat2,
                      rearth=rearth,
                      altitude=altitude,
                      haversine=haversine)

    if do_extra_width_calc:
        width1 = geo_dist(lon1, lat1,
                          lon2, lat1,
                          rearth=rearth,
                          altitude=altitude,
                          haversine=haversine)
        width2 = geo_dist(lon1, lat2,
                          lon2, lat2,
                          rearth=rearth,
                          altitude=altitude,
                          haversine=haversine)

        # width = np.max([width1, width2])

        # width = np.mean([width1, width2])
        width = np.mean(np.vstack([width1,width2]),axis=0)
    else:
        width = geo_dist(lon1, lat1,
                         lon2, lat1,
                         rearth=rearth,
                         altitude=altitude,
                         haversine=haversine)

    if spherical_rectangle:

        return 4*rearth**2.*np.arcsin(np.tan(width/rearth/2.)*np.tan(height/rearth/2.))
    else:
        # print("Vanlig")
        return height*width


def geo_dist(lon1, lat1,
             lon2, lat2,
             deg=True,
             rearth=6.370949e3,
             altitude=0,
             haversine=True):  # default to earth radius in km
    # MODIFICATION HISTORY:
    # 	Written by:	Daithi A. Stone (stoned@uvic.ca), 2000-06-29.
    #	Modified:	DAS, 2000-07-06 (removed LENGTH.pro, added
    #			DIMENSION.pro).
    #	Modified:	DAS, 2000-07-24 (added Degrad constant).
    #	Modified:	DAS, 2000-08-17 (coverted from Cosine to Haversine
    #			Haversine formula, added SINGLE keyword).
    #	Modified:	DAS, 2002-08-12 (complies with CONSTANTS.pro revision)
    #   Modified:       SMH, 2016-03-12 Made my own stuff
    #   Modified:       SMH, 2019-11-18 From IDL code

    rearth += altitude

    if deg:
        lat1, lat2, lon1, lon2 = map(np.deg2rad, (lat1, lat2, lon1, lon2))

    # Difference coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # breakpoint()

    if haversine:

        # ***********************************************************************
        # Haversine Formula

        # Main calculation
        a = (np.sin(dlat / 2.))**2 + np.cos(lat1) * \
            np.cos(lat2) * (np.sin(dlon / 2.))**2
        a = np.sqrt(a)

        # A fix if a>1
        id = np.where(a > 1)[0]

        if id.size > 0:
            a[id] = 1

        # Convert to distance
        dist = rearth * 2 * np.arcsin(a)

    else:
        # ***********************************************************************
        # Cosine Formula (not used, but I thought I would leave it in)

        # Convert to Cartesian coordinates
        x1 = np.cos(lon1) * np.cos(lat1)
        y1 = np.sin(lon1) * np.cos(lat1)
        z1 = np.sin(lat1)
        x2 = np.cos(lon2) * np.cos(lat2)
        y2 = np.sin(lon2) * np.cos(lat2)
        z2 = np.sin(lat2)

        # Direction cosine
        dx = x1*x2+y1*y2+z1*z2

        # A fix if the |cosine| > 1
        id = np.where(dx > 1)[0]
        if id.size > 0:
            dx[id] = 1.

        #Output (distance)
        dist = rearth*np.arccos(dx)

    return dist
