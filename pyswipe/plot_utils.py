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
import matplotlib.pyplot as plt
import warnings
from matplotlib import rc
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PolyCollection, LineCollection


d2r = np.pi / 180

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
rc('text', usetex=True)

class Polarplot(object):
    def __init__(self, ax, minlat = 50, plotgrid = True, sector = 'all', lt_label='lt', lat_label='lat', **kwargs):
        """ pax = Polarsubplot(axis, minlat = 50, plotgrid = True, **kwargs)

            **kwargs are the plot parameters for the grid

            this is a class which handles plotting in polar coordinates, specifically a latitude-local time grid

            Example:
            --------
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pax = Polarsubplot(ax)
            pax.MEMBERFUNCTION()
            plt.show()


            where memberfunctions include:
            plotgrid()                           - called by __init__
            plot(lat, lt, **kwargs)            - works like plt.plot
            text(lat, lt, text, **kwargs)      - works like plt.text
            scatter(lat, lt, **kwargs)         - works like plt.scatter
            contour(lat, lt, f)                - works like plt.contour
            contourf(lat, lt, f)               - works like plt.contourf


        Parameters
        ----------
        ax : matplotlib AxesSubplot object
        minlat : scalar, optional
            low latitude boundary of the plot. The default is 50.
        plotgrid : bool, optional
            Set True to plot a grid. The default is True.
        sector : string, optional
            Used to generate portions of polar plot.
            Can either use one of the following strings: 'all', 'dusk', 'dawn', 'night', 'day'. Or can be defined using numbers for example '18-6' will produce the same as night while '6-18' will produce the same as day. The default is 'all'.
        lt_label : string, optional
            Name of the local time co-ordinate used for displaying the hover co-ordinates. The default is 'lt'.
        lat_label : string, optional
            Name of the latitude co-ordinate used for displaying the hover co-ordinates. The defualt is 'lat'.
        **kwargs : dict
            Keywords passed to the plot function to control grid lines.

        """
        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.sector = sector

        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = .5

        if 'color' not in kwargs.keys():
            kwargs['color'] = 'lightgrey'

        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'

        # define the lt limits
        if sector.lower() == 'all':
            self.ltlims = 'lt >= 0'
        elif sector.lower() == 'dusk':
            self.ltlims = '(lt >= 12) | (lt == 0)'
        elif sector.lower() == 'dawn':
            self.ltlims = '(lt <= 12) | (lt == 24)'
        elif sector.lower() == 'night':
            self.ltlims = '(lt >= 18) | (lt <= 6)'
        elif sector.lower() == 'day':
            self.ltlims = '(lt >= 6) & (lt <=18)'
        else:
            sector = [float(s) for s in sector.split('-')]
            if sector[0]> sector[-1]:
                self.ltlims= f'(lt>={sector[0]})|(lt<={sector[-1]})'
            else:
                self.ltlims= f'(lt>={sector[0]})&(lt<={sector[-1]})'

        self.ax.set_axis_off()

        self.ax.format_coord= self.make_format(lt_label, lat_label)
        if plotgrid:
            self.plotgrid(**kwargs)

        # set suitable plot limits by drawing a circle at minlat:
        x, y = self._latlt2xy(np.full(100, self.minlat), np.linspace(0, 24, 100))
        x, y = np.hstack((x, 0)), np.hstack((y, 0)) # add origin
        self.ax.set_xlim(np.nanmin(x) - 0.1, np.nanmax(x) + 0.1)
        self.ax.set_ylim(np.nanmin(y) - 0.1, np.nanmax(y) + 0.1)


    def plot(self, lat, lt, **kwargs):
        """
        Wrapper for matplotlib's plot function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's plot
        """

        x, y = self._latlt2xy(lat, lt)
        return self.ax.plot(x, y, **kwargs)


    def text(self, lat, lt, text, ignore_plot_limits=False, **kwargs):
        """
        Wrapper for matplotlib's text function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's text
        """

        x, y = self._latlt2xy(lat, lt, ignore_plot_limits=ignore_plot_limits)

        if np.isfinite(x + y):
            return self.ax.text(x, y, text, **kwargs)
        else:
            print('text outside plot limit - set "ignore_plot_limits = True" to override')


    def write(self, lat, lt, text, ignore_plot_limits=False, **kwargs):
        """ Alias for text, a wapper for matplotlib's text function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's text
        """

        return self.text(lat, lt, text, ignore_plot_limits, **kwargs)


    def scatter(self, lat, lt, **kwargs):
        """
        Wrapper for matplotlib's scatter function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's scatter
        """

        x, y = self._latlt2xy(lat, lt)
        c = self.ax.scatter(x, y, **kwargs)
        return c


    def plotgrid(self, labels=False, **kwargs):
        """ plot lt, lat-grid on self.ax

        parameters
        ----------
        labels: bool
            set to True to include lat/lt labels
        **kwarsgs: dictionary
            passed to matplotlib's plot function (for linestyle etc.)

        """
        returns= []
        for lt in [0, 6, 12, 18]:
            returns.append(self.plot([self.minlat, 90], [lt, lt], **kwargs))

        lts = np.linspace(0, 24, 100)
        for lat in np.r_[90: self.minlat -1e-12 :-10]:
            returns.append(self.plot(np.full(100, lat), lts, **kwargs))

        # add LAT and LT labels to axis
        if labels:
            returns.append(self.writeLATlabels())
            returns.append(self.writeLTlabels())
        return tuple(returns)

    def writeLTlabels(self, lat = None, degrees = False, **kwargs):
        """ write local time labels at given latitude (default minlat - 2)
            if degrees is true, the longitude will be written instead of hour (with 0 at midnight)
        """
        if lat is None:
            lat = self.minlat - 2
        labels=[]
        if degrees:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                labels.append(self.write(lat, 0,    '0$^\circ$', verticalalignment = 'top'   , horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            if self.sector in ['all', 'night', 'dawn', 'day']:
                labels.append(self.write(lat, 6,   '90$^\circ$', verticalalignment = 'center', horizontalalignment = 'left'  , ignore_plot_limits=True, **kwargs))
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                labels.append(self.write(lat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            if self.sector in ['all', 'night', 'dusk', 'day']:
                labels.append(self.write(lat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , ignore_plot_limits=True, **kwargs))
        else:
            lt=np.array([0, 24])
            if any(eval(self.ltlims)):
                labels.append(self.write(lat, 0, '00', verticalalignment = 'top'    , horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            lt=6
            if eval(self.ltlims):
                labels.append(self.write(lat, 6, '06', verticalalignment = 'center' , horizontalalignment = 'left'  , ignore_plot_limits=True, **kwargs))
            lt=12
            if eval(self.ltlims):
                labels.append(self.write(lat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            lt=18
            if eval(self.ltlims):
                labels.append(self.write(lat, 18, '18', verticalalignment = 'center', horizontalalignment = 'right' , ignore_plot_limits=True, **kwargs))

            return labels


    # added by AØH 20/06/2022 to plot latitude labels
    def writeLATlabels(self, lt = None, **kwargs):
        """ write latitude labels """
        if lt == None:
            lt = 3
        if kwargs is not None:
            latkwargs = {'rotation':45, 'color':'lightgrey', 'backgroundcolor':'white', 'zorder':2, 'alpha':1.}
            latkwargs.update(kwargs)
        labels = []
        for lat in np.r_[self.minlat:81:10]:
            labels.append(self.write(lat, lt, str(lat)+'$^{\circ}$', ignore_plot_limits = False, **latkwargs))

        return labels


    def plotpins(self, lats, lts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', colors = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ plot vector field. Each vector is a dot with a line pointing in the vector direction

            kwargs go to ax.plot

            the markers at each pin can be modified by the following keywords, that go to ax.scatter:
            marker (default 'o')
            markersize (defult 20 - size in points^2)
            markercolor (default black)

        """
        returns= []
        lts = lts.flatten()
        lats = lats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))
        
        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                returns.append(self.ax.plot([0.9, 1], [0.95, 0.95], color = colors, linestyle = '-', linewidth = 2))
                returns.append(self.ax.text(0.9, 0.95, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size))

            scale = 0.1/SCALE


        x, y = self._latlt2xy(lats, lts)
        dx, dy = R.dot(self._northEastToCartesian(north, east, lts))
        segments = np.dstack((np.vstack((x, x + dx * scale)).T, np.vstack((y, y + dy * scale)).T))


        returns.append(self.ax.add_collection(LineCollection(segments, colors = colors, **kwargs)))

        if markersize != 0:
            returns.append(self.scatter(lats, lts, marker = marker, c = markercolor, s = markersize, edgecolors = markercolor))
        return tuple(returns)

    def quiver(self, lat, lt, north, east, rotation = 0, qkeyproperties = None, **kwargs):
        """ wrapper for matplotlib's quiver function, just for lat/lt coordinates and
            east/north components. Rotation specifies a rotation angle for the vectors,
            which could be convenient if you want to rotate magnetic field measurements
            to an equivalent current direction

            rotation in radians

            qkeyproperties: set to dict to pass to matplotlib quiverkey. The dict must contain
            'label': the label written next to arrow
            'U': length of the arrow key in same unit as north/east components

        """

        lt = lt.flatten()
        lat = lat.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        x, y = self._latlt2xy(lat, lt)
        dx, dy = R.dot(self._northEastToCartesian(north, east, lt))

        q = self.ax.quiver(x, y, dx, dy, **kwargs)

        if qkeyproperties is not None:
            qkeykwargs = {'X':.8, 'Y':.9, 'labelpos':'E'}
            qkeykwargs.update(qkeyproperties)

            self.ax.quiverkey(q, **qkeykwargs)

        return q



    def plottrack(self, lats, lts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', color = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ like plotpins, only it's not a line pointing in the arrow direction but a dot at the tip of the arrow

            kwargs go to ax.plot

            the markers at each pin can be modified by the following keywords, that go to ax.scatter:
            marker (default 'o')
            markersize (defult 20 - size in points^2)
            markercolor (default black)

        """

        lts = lts.flatten()
        lats = lats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))
        if 'key_coords' in kwargs:
            key_x, key_y= kwargs['key_coords']
            kwargs.pop('key_coords', None)
        else:
            key_x, key_y= 0.9, 0.95
        key=[]
        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                key.append(self.ax.plot([key_x, key_x+.1], [key_y]*2, color = color, linestyle = '-', linewidth = 2))
                key.append(self.ax.text(key_x, key_y, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size))

            scale = 0.1/SCALE

        xs, ys, dxs, dys = [],[],[],[]
        for i in range(len(lats)):

            lt = lts[i]
            lat = lats[i]

            x, y = self._latlt2xy(lat, lt)
            dx, dy = R.dot(self._northEastToCartesian(north[i], east[i], lt).reshape((2, 1))).flatten()
            xs.append(x)
            ys.append(y)
            dxs.append(dx)
            dys.append(dy)

        xs, ys, dxs, dys = np.array(xs),np.array(ys),np.array(dxs)*scale,np.array(dys)*scale
        if len(key)!=0:
            return self.ax.scatter(xs+dxs,ys+dys,marker = marker, c = markercolor, s = markersize, edgecolors = markercolor), key
        else:
            return self.ax.scatter(xs+dxs,ys+dys,marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


    def contour(self, lat, lt, f, **kwargs):
        """
        Wrapper for matplotlib's contour function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's contour
        """
        xea, yea = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits = True)
        mask, _  = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits = False)
        f = f.flatten()
        f[~np.isfinite(mask)] = np.nan

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f, (xx, yy))

        # ... and plot
        return self.ax.contour(xx, yy, gridf, **kwargs)


    def contourf(self, lat, lt, f, **kwargs):
        """
        Wrapper for matplotlib's contourf function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's contourf
        """

        xea, yea = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits = True)
        mask, _  = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits = False)
        f = f.flatten()
        f[~np.isfinite(mask)] = np.nan

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f, (xx, yy))

        # ... and plot
        return self.ax.contourf(xx, yy, gridf, **kwargs)


    def fill(self, lat, lt, **kwargs):
        """ Fill polygon defined in lat/lt, **kwargs are given to self.ax.contour. LT in hours - no rotation
        """

        xx, yy = self._latlt2xy(lat.flatten(), lt.flatten())

        # plot
        return self.ax.fill(xx, yy, **kwargs)


    def plot_terminator(self, time, sza = 90, north = True, apex = None, shadecolor = None, shade_kwargs={}, **kwargs):
        """ shade the area antisunward of the terminator

            Parameters
            ----------
            time : float or datetime
                if datetime, time describes the time for which to calculate the
                subsolar point and associated sunlight terminator. If float, it
                desribes where to draw a horizontal terminator line - in
                degrees from the pole towards the night side (think of it
                approximately as dipole tilt angle)
            sza : float, optional
                the terminator is defined as contours of this solar zenith angle
                default 90 degrees
            north : bool, optional
                set to True for terminator in the north (default), False for south
            apex : apexpy.Apex object, optional
                give an apexpy.Apex object to convert the terminator to magnetic
                apex coordinates. By default it is plotted in geographic
            shadecolor : string, optional
                color of a shade to be drawn on the night side of the
                terminator. Default is None, which means that no shade will be
                drawn
            shade_kwargs : dict
                Used if shadecolor is not None. keywords passed to matplotlib.patches'
                Pylogon function
            kwargs : dict
                keywords passed to matplotlib's plot function for the terminator line
        """
        returns= []
        hemisphere = 1 if north else -1

        if np.isscalar(time): # time is interpreted as dipole tilt angle
            y0 = -(hemisphere * time + sza - 90) / (90. - self.minlat) # noon-midnight coordinate
            xr = np.sqrt(1 - y0**2)

            x = np.array([-xr, xr]).flatten()
            y = np.array([y0, y0]).flatten()

        else: # time should be a datetime object

            sslat, sslon = subsol(time)

            # make cartesian vector
            x = np.cos(sslat * d2r) * np.cos(sslon * d2r)
            y = np.cos(sslat * d2r) * np.sin(sslon * d2r)
            z = np.sin(sslat * d2r)
            ss = np.array([x, y, z]).flatten()

            # construct a vector pointing roughly towards dawn, and normalize
            t0 = np.cross(ss, np.array([0, 0, 1]))
            t0 = t0/np.linalg.norm(t0)

            # make a new vector pointing northward at the 90 degree SZA contour:
            sza90 = np.cross(t0, ss)

            # rotate this about the dawnward vector to get specified SZA contour
            rotation_angle = (sza - 90) * d2r

            sza_vector = sza90 * np.cos(rotation_angle) + np.cross(t0, sza90) * np.sin(rotation_angle) + t0 * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle)) # (rodrigues formula)
            sza_vector = sza_vector.flatten()

            # rotate this about the sun-Earth line to trace out the trajectory:
            angles = np.r_[-np.pi/2 : 3*np.pi/2: 2*np.pi / 360][np.newaxis, :]
            r = sza_vector[:, np.newaxis] * np.cos(angles) + np.cross(ss, sza_vector)[:, np.newaxis] * np.sin(angles) + ss[:, np.newaxis] * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle))

            # convert to spherical
            t_lat = 90 - np.arccos(r[2]) / d2r
            t_lon = (np.arctan2(r[1], r[0])*180/np.pi) % 360

            londiff = (t_lon - sslon + 180) % 360 - 180 # signed difference in longitude
            t_lt = (180. + londiff)/15. # convert to lt with ssqlon at noon

            if apex is not None:
                mlat, mlon = apex.geo2apex(t_lat, t_lon, apex.refh)
                mlt = apex.mlon2mlt(mlon, time)
                t_lat, t_lt = mlat, mlt

            # limit contour to correct hemisphere:
            iii = (t_lat >= self.minlat) if north else (t_lat <= -self.minlat)
            if len(iii) == 0:
                return 0 # terminator is outside plot
            t_lat = t_lat[iii] #* hemisphere
            t_lt = t_lt[iii]

            x, y = self._latlt2xy(t_lat, t_lt)


        returns.append(self.ax.plot(x, y, **kwargs))

        x0, x1 = np.min([x[0], x[-1]]), np.max([x[0], x[-1]])
        y0, y1 = np.min([y[0], y[-1]]), np.max([y[0], y[-1]])

        a0 = np.arctan2(y0, x0)
        a1 = np.arctan2(y1, x1)
        if a1 < a0: a1 += 2*np.pi

        a = np.linspace(a0, a1, 100)[::-1]

        xx, yy = np.cos(a), np.sin(a)

        if shadecolor is not None:
             shade = Polygon(np.vstack((np.hstack((x[::hemisphere], xx)), np.hstack((y[::hemisphere], yy)))).T, closed = True, color = shadecolor, linewidth = 0, **shade_kwargs)
             returns.append(self.ax.add_patch(shade))
        return tuple(returns)



    def filled_cells(self, lat, lt, latres, ltres, data, resolution = 10, crange = None, levels = None, bgcolor = None, verbose = False, **kwargs):
        """ specify a set of cells in lat and lt, along with a data array,
            and make a color plot of the cells

        """

        if not all([len(lat) == len(q) for q in [lt,ltres,data]]):
            print("WARNING: Input arrays to filled_cells of unequal length! Your plot is probably incorrect")

        lat, lt, latres, ltres, data = map(np.ravel, [lat, lt, latres, ltres, data])
        mask = np.ones_like(data)
        mask[(~eval(self.ltlims)) | (lat < self.minlat)] = np.nan
        data*=mask
        if verbose:
            print(lt.shape, lat.shape, latres.shape, ltres.shape)

        la = np.vstack([(lt - 6) / 12. * np.pi + i * ltres / (resolution - 1.) / 12. * np.pi for i in range(resolution)]).T
        if verbose:
            print (la.shape)
        ua = la[:, ::-1]

        vertslo = np.dstack(((90 - lat          )[:, np.newaxis] / (90. - self.minlat) * np.cos(la),
                             (90 - lat          )[:, np.newaxis] / (90. - self.minlat) * np.sin(la)))
        vertshi = np.dstack(((90 - lat - latres)[:, np.newaxis] / (90. - self.minlat) * np.cos(ua),
                             (90 - lat - latres)[:, np.newaxis] / (90. - self.minlat) * np.sin(ua)))
        verts = np.concatenate((vertslo, vertshi), axis = 1)

        if verbose:
            print( verts.shape, vertslo.shape, vertshi.shape)


        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
            kwargs.pop('cmap')
        else:
            cmap = plt.cm.viridis

        if levels is not None:
            # set up a function that maps data values to color levels:
            nlevels = len(levels)
            lmin, lmax = levels.min(), levels.max()
            self.colornorm = lambda x: plt.Normalize(lmin, lmax)(np.floor((x - lmin) / (lmax - lmin) * (nlevels - 1)) / (nlevels - 1) * (lmax - lmin) + lmin)
            coll = PolyCollection(verts, facecolors = cmap(self.colornorm(data.flatten())), **kwargs)

        else:
            coll = PolyCollection(verts, array = data, cmap = cmap, **kwargs)
            if crange is not None:
                coll.set_clim(crange[0], crange[1])



        if bgcolor != None:
            radius = 2*((90 - self.minlat)/ (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder = 0, facecolor = bgcolor)
            self.ax.add_artist(bg)


        return self.ax.add_collection(coll)


    def plotimg(self,lat,lt,image,corr=True , crange = None, bgcolor = None, **kwargs):
        """
        Displays an image in polar coordinates.
        lat,lt,image are m x n arrays.
        If corr == True, lat and lt are corrected to edge coordinates. The plotted image is (m-2) x (n-2)
        If corr == False, lat and lat are assumed to be edge coordinates (plotted with a small offset). The plotted image is (m-1) x (n-1).
        crange is a tuple giving lower and upper colormap limits.
        bgcolor is a str giving background color inside the polar region.
        """

        x,y =self._latlt2xy(lat, lt)
        if corr:
            xe = np.full(np.subtract(x.shape,(1,1)),np.nan)
            ye = np.full(np.subtract(y.shape,(1,1)),np.nan)
            for i in range(xe.shape[0]):
                for j in range(xe.shape[1]):
                    xe[i,j]=np.mean(x[i:i+2,j:j+2])
                    ye[i,j]=np.mean(y[i:i+2,j:j+2])
            x=xe
            y=ye
            data = image[1:-1,1 :-1]
        else:
            data = image[1:, :-1]

        ll = (x[1:,  :-1].flatten(), y[1:,  :-1].flatten())
        lr = (x[1:,   1:].flatten(), y[1:,   1:].flatten())
        ul = (x[:-1, :-1].flatten(), y[:-1, :-1].flatten())
        ur = (x[:-1,  1:].flatten(), y[:-1,  1:].flatten())

        vertsx = np.vstack((ll[0], lr[0], ur[0], ul[0])).T
        vertsy = np.vstack((ll[1], lr[1], ur[1], ul[1])).T

        iii = np.where(vertsx**2 + vertsy**2 <=1, True, False)
        iii = np.any(iii, axis = 1).nonzero()[0]

        vertsx = vertsx[iii]
        vertsy = vertsy[iii]

        verts = np.dstack((vertsx, vertsy))
        if 'cmap' in kwargs.keys():
            cmap = kwargs.pop('cmap')
        else:
            cmap = plt.cm.viridis
        data = np.ma.array(data, mask=np.isnan(data))
        coll = PolyCollection(verts, array=data.flatten()[iii], cmap = cmap, edgecolors='none', **kwargs)
        if crange is not None:
            coll.set_clim(crange[0], crange[1])

        if bgcolor is not None:
            radius = 2*((90 - self.minlat)/ (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder = 0, facecolor = bgcolor)
            self.ax.add_artist(bg)

        return self.ax.add_collection(coll)


    def ampereplot(self, keys, data, contour = False, crange = None, nlevels = 100, cbar = False, **kwargs):
        """ plot ampere data. keys is the columns of the ampere dataframes.
            data is an array of equal size, that will be plotted in grids of 1 degree by 1 hour size

            set cbar True to draw a color bar

            2021-7-1: Kalle: Added this function from an old version of polarsublot. It worked in 2016, but I have
                             no clue if it still works...
        """
        data = np.array(data)

        # MAKE VERTICES
        colats, lts = np.array(list(keys)).T
        lats = np.abs(90. - colats)

        ltres, latres = 1, 1

        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
            kwargs.pop('cmap')
        else:
            cmap = plt.cm.RdBu_r

        if not contour: # plot so that the grid cells are seen
            latl, ltl = lats, lts
            latu = latl + latres
            ltr  = ltl + ltres
            verts = np.empty((len(keys), 20, 2))

            for i in range(len(lts)):
                # convert corners to cartesian (lat 90 at origin, lat 0 at distance 1 from origin)
                la = np.linspace((ltl[i]-6)/12.*np.pi, (ltr[i]-6)/12.*np.pi, 10)
                ua = np.linspace((ltl[i]-6)/12.*np.pi, (ltr[i]-6)/12.*np.pi, 10)[::-1]
                verts[i, :10, :]  = (90 - latl[i])/(90 - self.minlat) * np.array([np.cos(la), np.sin(la)]).T
                verts[i,  10:, :] = (90 - latu[i])/(90 - self.minlat) * np.array([np.cos(ua), np.sin(ua)]).T

            # the sign of the data is changed to make the color scale consistent with AMPERE summary plots
            coll = PolyCollection(verts[np.isfinite(data)], array = data[np.isfinite(data)], cmap = cmap, edgecolors='none', **kwargs)
            if crange is not None:
                coll.set_clim(crange[0], crange[1])
            self.ax.add_collection(coll)

        else: # plot a smooth contour
            coll = self.contourf(lats, lts, data, cmap = cmap, levels = np.linspace(crange[0], crange[1], nlevels), extend = 'both', **kwargs)

        if cbar:

            clim = coll.get_clim()
            axpos = self.ax.get_position().bounds # position of current plot
            cbax = self.ax.figure.add_subplot(111)
            cbax.set_position([axpos[0] + axpos[2], axpos[1], 0.02, axpos[3]])
            #cbax.get_xaxis().set_visible(False)
            #cbax.get_yaxis().set_visible(False)
            RESOLUTION = 100
            colors = np.linspace(clim[0], clim[1], RESOLUTION + 1)
            colors -= colors.min()
            cbax.set_ylim(colors.min(), colors.max())
            verts = np.empty((RESOLUTION, 4, 2))
            for i in range(RESOLUTION):
                verts[i, :, :] = np.array([[0, 0, 1, 1], [colors[i], colors[i+1], colors[i+1], colors[i]]]).T

            cbarcoll = PolyCollection(verts, array = np.linspace(clim[0], clim[1], RESOLUTION), cmap = cmap, edgecolors='none')
            cbarcoll.set_clim(clim)
            cbax.add_collection(cbarcoll)

            cbax.set_xlabel(r'$\mu$A/m$^2$', labelpad = 1)
            cbax.set_yticks([colors.min(), 0 - clim[0], colors.max()])
            cbax.set_yticklabels([str(clim[0]), '0', str(clim[1])])
            cbax.set_xlim(0, 1)
            cbax.set_xticks([0, 1])
            cbax.set_xticklabels(['', ''])

        return coll


    def coastlines(self, time = None, mag = None, north = True, resolution = '50m', **kwargs):
        """ plot coastlines

        Coastline data are read from numpy savez files. These files are made
        with the download_coastlines.py script in the helper_scripts folder,
        which uses the cartopy module.

        Parameters
        ----------
        time: datetime, optional
            give a datetime to replace longitude with local time when
            plotting coastlines
        mag: apexpy.Apex, optional
            give an apexpy.Apex object to convert coastlines to magnetic
            apex coordinates. If None (default), coastlines will be
            plotted in geographic coordinates
        north: bool, optional
            set to False if you want coastlines plotted for the southern
            hemisphere. Default is True
        resolution: string, optional
            Set to '50m' or '110m' to adjust the resolution of the coastlines.
            These options correspond to options given to cartopy. Default is
            '50m', which is the highest resolution.
        **kwargs: dict, optional
            keywords passed to matplotlib's plot function
        """

        coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')

        segments = []
        for cl in coastlines:
            lat, lon = coastlines[cl]

            # convert lat and lon to magnetic if mag is given:
            if mag is not None:
                lat, lon = mag.geo2apex(lat, lon, mag.refh)

            if time is not None: # calculate local time if time is given
                if mag is None: # calculate geographic local time
                    sslat, sslon = subsol(time)
                    londiff = (lon - sslon + 180) % 360 - 180
                    lon = (180. + londiff)/15.
                else: # magnetic local time:
                    lon = mag.mlon2mlt(lon, time)
            else: # keep longitude - just divide by 15 to get unit hours
                lon = lon / 15

            if not north:
                lat = -1 * lat
            if not np.any(lat > self.minlat):
                continue

            iii = lat > self.minlat
            lat[~iii] = np.nan
            lon[~iii] = np.nan

            x, y = self._latlt2xy(lat, lon)

            segments.append(np.vstack((x, y)).T)

        collection = LineCollection(segments, **kwargs)
        return self.ax.add_collection(collection)



    def _latlt2xy(self, lat, lt, ignore_plot_limits=False):
        """

        Parameters
        ----------
        lt : TYPE
            DESCRIPTION.
        lat : TYPE
            DESCRIPTION.
        ignore_plot_limits : boolean, optional
            When True the conversion will ignore the limits of the plot allowing plotting outside
            the limits of the subplot, for example this is used in writeLTlabels. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        np.seterr(invalid='ignore', divide='ignore')
        lt = np.array(lt) % 24
        lat = np.abs(np.array(lat))

        if lt.size!= lat.size:
            raise ValueError('x and y must be the same size')

        r = (90. - np.abs(lat))/(90. - self.minlat)
        a = (lt - 6.)/12.*np.pi

        x, y = r*np.cos(a), r*np.sin(a)

        if ignore_plot_limits:
            return x, y
        else:
            mask = np.ones_like(x)
            mask[(~eval(self.ltlims)) | (lat < self.minlat)] = np.nan
            return x * mask, y * mask

    def _xy2latlt(self, x, y, ignore_plot_limits=False):
        """
        convert x, y to lt, lat, where x**2 + y**2 = 1 corresponds to self.minlat

        Parameters
        ----------
        x : float/integer/list/array
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        ignore_plot_limits : boolean, optional
            When True the conversion will ignore the limits of the plot allowing plotting outside
            the limits of the subplot, for example this is used in writeLTlabels. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        x, y = np.array(x, ndmin = 1, dtype='float64'), np.array(y, ndmin = 1, dtype='float64') # convert to array to allow item assignment

        lat = 90 - np.sqrt(x**2 + y**2)*(90. - self.minlat)
        lt = np.arctan2(y, x)*12/np.pi + 6
        lt = lt % 24

        if ignore_plot_limits:
            return lat, lt
        else:
            mask = np.ones_like(lt)
            mask[(~eval(self.ltlims)) | (lat < self.minlat)] = np.nan
            return lat * mask, lt * mask


    def _northEastToCartesian(self, north, east, lt):
        a = (lt - 6)/12*np.pi # convert LT to angle with x axis (pointing from pole towards dawn)

        x1 = np.array([-north*np.cos(a), -north*np.sin(a)]) # arrow direction towards origin (northward)
        x2 = np.array([-east*np.sin(a),  east*np.cos(a)])   # arrow direction eastward

        return x1 + x2


    def make_format(current, lt_label='lt', lat_label='lat'):
    # current and other are axes
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current._xy2latlt(x, y)[::-1]
            string_original= 'x={:.2f}, y={:.2f}'.format(x, y)
            string_magnetic= f'{lt_label}={float(display_coord[0]):.2f}, {lat_label}={float(display_coord[1]):.2f}'
            
            
            return (string_original.ljust(20)+string_magnetic)
        return format_coord





# Helper functions
# ----------------
def date_to_doy(month, day, leapyear = False):
    """ return day of year (DOY) at given month, day
    """

    # check that month in [1, 12]
    if month < 1 or month > 12:
        raise ValueError('month not in [1, 12]')

    # check if day < 1
    if day < 1:
        raise ValueError('date2doy: day must not be less than 1')

    # check if day exceeds days in months
    days_in_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    if day > days_in_month[month]:
        if not (leapyear & (month == 2) & (day == 29)):
            raise ValueError('date2doy: day must not exceed number of days in month')

    cumdaysmonth = np.cumsum(days_in_month[:-1])

    # day of year minus possibly leap day:
    doy = cumdaysmonth[month - 1] + day

    # add 1 if leapyear and month >= 3:
    return doy + leapyear if month >= 3 else doy


def is_leapyear(year):
    """ Check for leapyear

    """
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def subsol(time):
    """
    calculate subsolar point at given datetime(s)

    returns:
      subsol_lat  -- latitude of the subsolar point
      subsol_lon  -- longiutde of the subsolar point

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac,
    results are good to at least 0.01 degree latitude and 0.025 degree
    longitude between years 1950 and 2050.  Accuracy for other years
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.

    """

    year = time.year
    # day of year:
    doy  = date_to_doy(time.month, time.day, is_leapyear(year))
    # seconds since start of day:
    ut   = time.hour * 60.**2 + time.minute*60. + time.second

    yr = year - 2000

    if year >= 2100 or year <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = nleap - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    if year <= 1900: nleap = nleap + ncent

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat, sbsllon


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
