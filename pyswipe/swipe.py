""" 
Python interface for the Swarm Ionospheric Polar Electrodynamics (SWIPE) model

This module can be used to 
1) Calculate and plot the average convection electric field on a grid. 
   This is done through the SWIPE class. The parameters 
   that are available for calculation/plotting are:
    - convection velocity (vector)
    - convection electric field (vector)
    - potential (scalar)

MIT License

Copyright (c) 2023 Spencer M. Hatch and Karl M. Laundal

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
# import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# from pyswipe.plot_utils import equal_area_grid, Polarsubplot, get_h2d_bin_areas
from pyswipe.plot_utils import equal_area_grid, Polarplot, get_h2d_bin_areas
from .sh_utils import legendre, get_R_arrays, get_R_arrays__symm, get_A_matrix__Ephizero, get_A_matrix__potzero, SHkeys 
from .model_utils import get_model_vectors, get_coeffs, default_coeff_fn, get_truncation_levels
from .mlt_utils import mlt_to_mlon
import ppigrf
import apexpy
from datetime import datetime
from functools import reduce
from builtins import range
import warnings



rc('text', usetex=False)

MU0   = 4*np.pi*1e-7 # Permeability constant
REFRE = 6371.2 # Reference radius used in geomagnetic modeling
d2r = np.pi/180

# DEFAULT = object()

# Defaults for deciding whether the reconstructed picture of average electrodynamics
# is consistent with the assumed neutral wind pattern (corotation with Earth by default)
DEFAULT_MIN_EMWORK = 0.5        # mW/m²
DEFAULT_MIN_HALL = 0.05         # mho
DEFAULT_MAX_HALL = 100.         # mho

class SWIPE(object):
    """
    Calculate and plot maps of the Swarm HEmispherically resolved Ionospheric Convection (SWIPE) model

    Parameters
    ---------
    v : float
        solar wind velocity in km/s
    By : float
        IMF GSM y component in nT
    Bz : float
        IMF GSM z component in nT
    tilt : float
        dipole tilt angle in degrees
    f107 : float
        F10.7 index in s.f.u.
    minlat : float, optional
        low latitude boundary of grids  (default 60)
    maxlat : float, optional
        low latitude boundary of grids  (default 89.99)
    #height : float, optional (DO WE NEED THIS?)
    #    altitude of the ionospheric currents in km (default 110)
    dr : int, optional
        latitudinal spacing between equal area grid points (default 2 degrees)
    M0 : int, optional
        number of grid points in the most poleward circle of equal area grid points (default 4)
    resolution: int, optional
        resolution in both directions of the scalar field grids (default 100)
    coeff_fn: str, optional
        file name of model coefficients - must be in format produced by model_vector_to_txt.py
        (default is latest version)


    Examples
    --------
    >>> # initialize by supplying a set of external conditions:
    >>> m = SWIPE(solar_wind_velocity_in_km_per_s, 
                  IMF_By_in_nT, IMF_Bz_in_nT, 
                  dipole_tilt_in_deg, 
                  F107_index)
    
    >>> # make summary plot:
    >>> m.plot_potential()
        
    >>> # calculate field-aligned currents on a pre-defined grid
    >>> Ju = m.get_upward_current()

    >>> # Ju will be evaluated at the following coords:
    >>> mlat, mlt = m.scalargrid

    >>> # It is also possible to specify coordinates (can be slow with 
    >>> # repeated calls)
    >>> Ju = m.get_upward_current(mlat = my_mlats, mlt = my_mlts)

    >>> # get components of the total height-integrated horizontal current,
    >>> # calculated on a pre-defined grid
    >>> j_east, j_north = m.get_total_current()

    >>> # j_east, and j_north will be evaluated at the following coords 
    >>> # (default grids are different with vector quantities)
    >>> mlat, mlt = m.vectorgrid

    >>> # update model vectors (tor_c, tor_s, etc.) without 
    >>> # recalculating the other matrices:
    >>> m.update_model(new_v, new_By, new_Bz, new_tilt, new_f107)

    Attributes
    ----------
    tor_c : numpy.ndarray
        vector of cos term coefficents in the toroidal field expansion
    tor_s : numpy.ndarray
        vector of sin term coefficents in the toroidal field expansion
    # pol_c : numpy.ndarray
    #     vector of cos term coefficents in the poloidal field expansion
    # pol_s : numpy.ndarray
    #     vector of sin term coefficents in the poloidal field expansion
    # keys_P : list
    #     list of spherical harmonic wave number pairs (n,m) corresponding to elements of pol_c and pol_s 
    keys_T : list
        list of spherical harmonic wave number pairs (n,m) corresponding to elements of tor_c and tor_s 
    vectorgrid : tuple
        grid used to calculate and plot vector fields
    scalargrid : tuple
        grid used to calculate and plot scalar fields
                   
        The grid formats are as follows (see also example below):
        (np.hstack((mlat_north, mlat_south)), np.hstack((mlt_north, mlt_south)))
        
        The grids can be changed directly, but member function calculate_matrices() 
        must then be called for the change to take effect. 

    """



    def __init__(self, v, By, Bz, tilt, f107, minlat = 60, maxlat = 89.99,
                 height = 110., dr = 2, M0 = 4, resolution = 100,
                 coeff_fn=None,
                 # use_transpose_coeff_fn=False,
                 zero_lats=None,
                 min_Efield__mVm=None,
                 min_emwork=None,
                 min_hall=None,
                 max_hall=None,
    ):
        """ __init__ function for class SWIPE
        """

        if coeff_fn is None:
            # if use_transpose_coeff_fn:
            #     coeff_fn = default_transpose_coeff_fn
            # else:
            coeff_fn = default_coeff_fn

        self.coeff_fn = coeff_fn

        # In case we want to use AMPS 
        self.inputs = dict(v=v,
                           By=By,
                           Bz=Bz,
                           tilt=tilt,
                           f107=f107,
                           minlat=minlat,
                           maxlat=maxlat,
                           height=height,
                           dr=dr,
                           M0=M0,
                           resolution=resolution)

        # Holder for AMPS object
        # Needed for calculation of conductances, emwork, Poynting flux
        self.amps = None

        # self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = self.coeff_fn)
        self.tor_c, self.tor_s, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = self.coeff_fn)

        # For deciding which type of model we're working with.
        self.keys_T = [c for c in self.tor_keys]
        self.m_T = np.array(self.keys_T).T[1][np.newaxis, :]
        self.n_T = np.array(self.keys_T).T[0][np.newaxis, :]

        self.N, self.M = np.max( np.hstack((np.array([c for c in self.tor_keys]).T, np.array([c for c in self.tor_keys]).T)), axis = 1)

        if self.tor_keys[0] == (1,0) or self.tor_keys[0] == (3,0):
            self.legendrelike_func = legendre
        elif self.tor_keys[0] == (2,0):
            if zero_lats is None:
                def legendrelike_func(*args, **kwargs):
                    return get_R_arrays(*args,**kwargs)
            elif hasattr(zero_lats,'__len__'):
                if len(zero_lats) == 2:
                    def legendrelike_func(*args, zero_lats=zero_lats, **kwargs):
                        return get_R_arrays(*args,**kwargs, zero_thetas=90.-zero_lats)
                else:
                    def legendrelike_func(*args, zero_lats=zero_lats, **kwargs):
                        return get_R_arrays__symm(*args,**kwargs,
                                                  zero_thetas=90.-np.array([zero_lats,-zero_lats]))
            else:
                def legendrelike_func(*args, zero_lats=zero_lats, **kwargs):
                    return get_R_arrays__symm(*args,**kwargs,
                                              zero_thetas=90.-np.array([zero_lats,-zero_lats]))
                

            self.legendrelike_func = legendrelike_func

        self.height = height

        self._eqgridkw = dict(dr = dr,
                              M0 = M0,
                              N = int(40/dr))

        assert len(self.tor_s) == len(self.tor_c)

        self.minlat = minlat
        self.maxlat = maxlat

        self.vectorgrid = self._get_vectorgrid()
        self.scalargrid = self._get_scalargrid(resolution = resolution)

        mlats = np.split(self.scalargrid[0], 2)[0].reshape((self.scalar_resolution, self.scalar_resolution))
        mlts  = np.split(self.scalargrid[1], 2)[0].reshape((self.scalar_resolution, self.scalar_resolution))
        mlatv = np.split(self.vectorgrid[0], 2)[0]
        mltv  = np.split(self.vectorgrid[1], 2)[0]

        self.plotgrid_scalar = (mlats, mlts)
        self.plotgrid_vector = (mlatv, mltv)

        self.calculate_matrices()

        if min_Efield__mVm is not None:
            min_Efield__mVm = np.float64(min_Efield__mVm)
        self.min_Efield__mVm = min_Efield__mVm

        if min_emwork is None:
            self.min_emwork = DEFAULT_MIN_EMWORK
        if min_hall is None:
            self.min_hall = DEFAULT_MIN_HALL
        if max_hall is None:
            self.max_hall = DEFAULT_MAX_HALL

        self.pax_plotopts = dict(minlat = self.minlat,
                                 linestyle = ':',
                                 linewidth = .7,
                                 color = 'grey')# ,
                                 # color = 'lightgrey')



    def update_model(self, v, By, Bz, tilt, f107, coeff_fn = None):
        """
        Update the model vectors without updating all the other matrices. This leads to better
        performance than just making a new SWIPE object.

        Parameters
        ----------
        v : float
            solar wind velocity in km/s
        By : float
            IMF GSM y component in nT
        Bz : float
            IMF GSM z component in nT
        tilt : float
            dipole tilt angle in degrees
        f107 : float
            F10.7 index in s.f.u.

        Examples
        --------
        If model currents shall be calculated on the same grid for a range of 
        external conditions, it is faster to do this:
        
        >>> m1 = SWIPE(solar_wind_velocity_in_km_per_s, IMF_By_in_nT, IMF_Bz_in_nT, dipole_tilt_in_deg, F107_index)
        >>> # ... current calculations ...
        >>> m1.update_model(new_v, new_By, new_Bz, new_tilt, new_f107)
        >>> # ... new current calcuations ...
        
        than to make a new object:
        
        >>> m2 = SWIPE(new_v, new_By, new_Bz, new_tilt, new_f107)
        >>> # ... new current calculations ...
        
        Also note that the inputs are scalars in both cases. It is possible to optimize the calculations significantly
        by allowing the inputs to be arrays. That is not yet implemented.

        """

        # if coeff_fn is None:
        #     self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = self.coeff_fn)
        # else:
        #     self.tor_c, self.tor_s, self.pol_c, self.pol_s, self.pol_keys, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = coeff_fn)
       
        if coeff_fn is None:
            self.tor_c, self.tor_s, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = self.coeff_fn)
        else:
            self.tor_c, self.tor_s, self.tor_keys = get_model_vectors(v, By, Bz, tilt, f107, coeff_fn = coeff_fn)

        # OLD
        # self.keys_T = [c for c in self.tor_keys]
        # self.m_T = np.array(self.keys_T).T[1][np.newaxis, :]
        # self.n_T = np.array(self.keys_T).T[0][np.newaxis, :]

        # n_m0 = np.sum([key[1] == 0 for key in self.tor_keys])
        # n_m1 = np.sum([key[1] == 1 for key in self.tor_keys])

        # # Do we need to apply A matrix to get remaining coefficients?
        # apply_A = False
        # if n_m1 < n_m0:
        #     apply_A = True

        #     A = get_A_matrix__Ephizero(self.N, self.M,
        #                                zero_thetas = 90.-np.array([47.,-47.]),
        #                                return_all = False)

        #     checkemout_coeffs = (self.n_T.ravel() < (self.N-1)) | (self.m_T.ravel() == 0)


        # elif (n_m1 == n_m0) and (self.n_T.min() == 3):
        #     apply_A = True

        #     # warnings.warn("Zero thetas flipped!")
        #     # A = get_A_matrix__potzero(self.N, self.M,
        #     #                           zero_thetas = 90.-np.array([-47.,47.]),
        #     #                           return_all = False)

        #     # warnings.warn("Zero thetas flipped!")
        #     A = get_A_matrix__potzero(self.N, self.M,
        #                               zero_thetas = 90.-np.array([47.,-47.]),
        #                               return_all = False)

        #     checkemout_coeffs = []
        #     for count,(n,m) in enumerate(zip(self.n_T.ravel(),self.m_T.ravel())):
        #         nprime = np.maximum(1,m)
        #         checkemout_coeffs.append( n >= (nprime + 2) )
        #     checkemout_coeffs = np.array(checkemout_coeffs)


        # # Apply A, if necessary
        # if apply_A:
        #     orig_tor_c = self.tor_c[checkemout_coeffs]
        #     orig_tor_s = self.tor_s[checkemout_coeffs]

        #     tmp_tor_c = orig_tor_c.copy()
        #     tmp_tor_s = orig_tor_s.copy()

        #     self.tor_c = A@tmp_tor_c
        #     self.tor_s = A@tmp_tor_s

        #     keys = SHkeys(self.N, self.M).setNmin(1).MleN().Mge(0)
        #     self.keys_T = [key for key in keys]
        #     self.m_T = keys.m
        #     self.n_T = keys.n

        # NEW
        self.keys_T = [c for c in self.tor_keys]
        self.m_T = np.array(self.keys_T).T[1][np.newaxis, :]
        self.n_T = np.array(self.keys_T).T[0][np.newaxis, :]

        self.N, self.M = np.max( np.hstack((np.array([c for c in self.tor_keys]).T, np.array([c for c in self.tor_keys]).T)), axis = 1)

        self.inputs['v'] = v
        self.inputs['By'] = By
        self.inputs['Bz'] = Bz
        self.inputs['tilt'] = tilt
        self.inputs['f107'] = f107

        if self.amps is not None:
            # self.amps.update_model(self, v, By, Bz, tilt, f107, coeff_fn = self.amps.coeff_fn)
            # breakpoint()
            self.amps.update_model(v, By, Bz, tilt, f107)


    def get_AMPS_current(self, mlat = None, mlt = None, grid = False):
        """
        Get AMPS horizontal currents in A/m.

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 

        2023/07/27
        """

        if self.amps is None:
            from pyamps import amps

            self.amps = amps.AMPS(self.inputs['v'],
                                  self.inputs['By'],
                                  self.inputs['Bz'],
                                  self.inputs['tilt'],
                                  self.inputs['f107'],
                                  minlat=self.inputs['minlat'],
                                  maxlat=self.inputs['maxlat'],
                                  height=self.inputs['height'],
                                  dr=self.inputs['dr'],
                                  M0=self.inputs['M0'],
                                  resolution=self.inputs['resolution'])

        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            sinI = self.sinI_vector.copy().ravel()
            mlat, mlt = mlat.ravel(), mlt.ravel()

            # mlat, mlt = self.scalargrid
            # sinI = self.sinI_scalar.copy().ravel()
            # mlat, mlt = mlat.ravel(), mlt.ravel()
            J_e, J_n = self.amps.get_total_current(grid=grid)
        else:
            # assert 2<0,"Here you need to calc sinI manually. Make sure this is OK."
            sinI = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*np.cos(mlat * d2r)**2)
            J_e, J_n = self.amps.get_total_current(mlat,mlt,grid=grid)

        # From mA/m to A/m
        J_e /= 1000.
        J_n /= 1000.

        return J_e, J_n


    def _get_vectorgrid(self, **kwargs):
        """ 
        Make grid for plotting vectors

        kwargs are passed to equal_area_grid(...)
        """

        # grid = equal_area_grid(dr = self._eqgridkw['dr'], M0 = self._eqgridkw['M0'], **kwargs)
        grid = equal_area_grid(**self._eqgridkw)
        # grid = equal_area_grid(dr = self._eqgridkw['dr'], M0 = self._eqgridkw['M0'], N=40, **kwargs)
        mlt  = grid[1] + grid[2]/2. # shift to the center points of the bins
        # mlat = grid[0] + (grid[0][1] - grid[0][0])/2  # shift to the center points of the bins
        mlat = grid[0] + self._eqgridkw['dr']/2  # shift to the center points of the bins

        mlt  = mlt[ (mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <=60 )]
        mlat = mlat[(mlat >= self.minlat) & (mlat <= self.maxlat)]# & (mlat <= 60)]

        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,  mlt)) # add southern hemisphere points


        return mlat[:, np.newaxis], mlt[:, np.newaxis] # reshape to column vectors and return


    def _get_scalargrid(self, resolution = 100):
        """ 
        Make grid for calculations of scalar fields 

        Parameters
        ----------
        resolution : int, optional
            resolution in both directions of the scalar field grids (default 100)
        """

        mlat, mlt = map(np.ravel, np.meshgrid(np.linspace(self.minlat , self.maxlat, resolution), np.linspace(-179.9, 179.9, resolution)))
        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,   mlt)) * 12/180 # add points for southern hemisphere and scale to mlt
        self.scalar_resolution = resolution

        return mlat[:, np.newaxis], mlt[:, np.newaxis] + 12 # reshape to column vectors and return


    def get_grid_binedges(self,gridtype='scalar'):
        """
        """

        if gridtype == 'scalar':
            mlats, mlts = self.scalargrid

            mlatsN,mlatsS = np.split(mlats,2)
            mlatsN,mlatsS = mlatsN.flatten(),mlatsS.flatten()
            mltsN,mltsS = np.split(mlts,2)
            mltsN,mltsS = mltsN.flatten(),mltsS.flatten()
            
            # Get MLAT diffs
            mlatdiffs = np.diff(mlatsN)
            shifts = np.where(mlatdiffs < 0)[0]
            mlatdiffs[mlatdiffs < 0] = np.median(mlatdiffs)  # Get rid of neg vals
            mlatdiffs = np.insert(mlatdiffs,-1,mlatdiffs[0])
            mlatmin = mlatsN-mlatdiffs/2.
            mlatmax = mlatsN+mlatdiffs/2.
            mlatmax[mlatmax > 90] = 90.
            
            # Get MLT diffs
            # Proof that the spacing is alltid the same
            tmpmltdiffs = np.diff(mltsN)[shifts]
            assert np.allclose(np.ones(shape=tmpmltdiffs.shape)*tmpmltdiffs[0],tmpmltdiffs)
            
            mltmin = mltsN-tmpmltdiffs[0]/2.
            mltmax = mltsN+tmpmltdiffs[0]/2.
    
        elif gridtype == 'vector':
            mlats,mlts,mltres = equal_area_grid(dr = self._eqgridkw['dr'], M0 = self._eqgridkw['M0'])

            mlatmin = mlats
            mlatmax = mlatmin+self._eqgridkw['dr']
            mlatmax[np.isclose(mlats.max(),mlats)] = 90 

            mltmin = mlts
            mltmax = mlts+mltres

        return mlatmin,mlatmax,mltmin,mltmax


    def calculate_matrices(self):
        """ 
        Calculate the matrices that are needed to calculate currents and potentials 

        Call this function if and only if the grid has been changed manually
        """

        mlt2r = np.pi/12

        # cos(m * phi) and sin(m * phi):
        self.tor_cosmphi_vector = np.cos(self.m_T * self.vectorgrid[1] * mlt2r)
        self.tor_cosmphi_scalar = np.cos(self.m_T * self.scalargrid[1] * mlt2r)
        self.tor_sinmphi_vector = np.sin(self.m_T * self.vectorgrid[1] * mlt2r)
        self.tor_sinmphi_scalar = np.sin(self.m_T * self.scalargrid[1] * mlt2r)

        self.coslambda_vector = np.cos(self.vectorgrid[0] * d2r)
        self.coslambda_scalar = np.cos(self.scalargrid[0] * d2r)

        self.sinI_vector = 2 * np.sin(self.vectorgrid[0] * d2r)/np.sqrt(4-3*self.coslambda_vector**2)
        self.sinI_scalar = 2 * np.sin(self.scalargrid[0] * d2r)/np.sqrt(4-3*self.coslambda_scalar**2)

        # P and dP ( shape  NEQ, NED):
        vector_P, vector_dP = self.legendrelike_func(self.N, self.M, 90 - self.vectorgrid[0])
        scalar_P, scalar_dP = self.legendrelike_func(self.N, self.M, 90 - self.scalargrid[0])

        self.tor_P_vector  =  np.array([vector_P[ key] for key in self.keys_T ]).squeeze().T
        self.tor_dP_vector = -np.array([vector_dP[key] for key in self.keys_T ]).squeeze().T
        self.tor_P_scalar  =  np.array([scalar_P[ key] for key in self.keys_T ]).squeeze().T
        self.tor_dP_scalar = -np.array([scalar_dP[key] for key in self.keys_T ]).squeeze().T


    def get_toroidal_scalar(self, mlat = None, mlt = None, grid = False):
        """ 
        Calculate the toroidal scalar values (unit is V(??)). 

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the toroidal scalar. Will be ignored if mlt is not 
            also specified. If not specified, the calculations will be done using the coords of the 
            `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the toroidal scalar. Will be ignored if mlat is not
            also specified. If not specified, the calculations will be done using the coords of the 
            `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        T : numpy.ndarray
            Toroidal scalar evaluated at self.scalargrid, or, if specified, mlat/mlt 
        """

        if mlat is None or mlt is None:
            T = (  np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c)
                 + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) 

        else: # calculate at custom coordinates
            if grid:
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = self.legendrelike_func(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                m_T = self.m_T[np.newaxis, ...] # (1, 1, 257)

                cosmphi = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)

                T = np.dot(P * cosmphi, self.tor_c) + \
                    np.dot(P * sinmphi, self.tor_s)

                T = T.squeeze()

            else:
                shape = mlat.shape

                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = self.legendrelike_func(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )

                T = np.dot(P * cosmphi, self.tor_c) + \
                    np.dot(P * sinmphi, self.tor_s) 

                T = T.reshape(shape)


        return T


    def get_potential(self, mlat = None, mlt = None, grid = False):
        """
        Calculate the potential (unit is kV, I hope). The 
        calculations refer to the height chosen upon initialization of the 
        SWIPE object (default 110 km).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `scalargrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Returns
        -------
        phi : numpy.ndarray
            Ionospheric potential evaulated at self.scalargrid, or, if specified, mlat/mlt
        """

        if mlat is None or mlt is None:
            # Ju = -1e-6/(MU0 * (REFRE + self.height) ) * (   np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
            #                                               + np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) )


            # phi = (np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
            #      + np.dot(self.n_T * (self.n_T + 1) * self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) )

            # phi = (np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
            #      + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) # div by 1000 to get kV, I hope

            # Since coeffs are in mV/m multiplying by REFRE in km gives phi units of V, and dividing by 1000 gives phi in kV
            phi = REFRE * (np.dot(self.tor_P_scalar * self.tor_cosmphi_scalar, self.tor_c) 
                           + np.dot(self.tor_P_scalar * self.tor_sinmphi_scalar, self.tor_s) ) / 1000

        else: # calculate at custom coordinates
            if grid:
                assert 2<0,"You haven't implemented grid calcs!"

                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = self.legendrelike_func(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                n_T, m_T = self.n_T[np.newaxis, ...], self.m_T[np.newaxis, ...] # (1, 1, 257)
                
                cosmphi = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                Ju = -1e-6/(MU0 * (REFRE + self.height) ) * ( np.dot(n_T * (n_T + 1) * P * cosmphi, self.tor_c) 
                                                          +   np.dot(n_T * (n_T + 1) * P * sinmphi, self.tor_s) )

                Ju = Ju.squeeze() # (nmlat, nmlt), transpose of original  
    
            else:    
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[:, np.newaxis]

                P, dP = self.legendrelike_func(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )
                phi = REFRE * (np.dot(P * cosmphi, self.tor_c) 
                               + np.dot(P * sinmphi, self.tor_s) ) / 1000

        return phi


    def get_efield_MA(self, mlat = None, mlt = None, grid = False,
                      return_emphi_emlambda=False,
                      return_magnitude=False):
        """ 
        Calculate the electric field, in mV/m, along Modified Apex basevectors d1 and d2.
        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.


        Return
        ------
        E_d1 : numpy.ndarray, float
            Component of the electric field along MA(110) base vector d1, which is "more-or-less in the magnetic eastward direction" (Just below Eq. 3.18 in Richmond, 1995)
        E_d2 : numpy.ndarray, float
            Component of the electric field along MA(110) base vector d2, which is "generally downward and/or equatorward" (Just below Eq. 3.18 in Richmond, 1995)

        See Also
        --------
        """

        assert not return_magnitude,"Need to calculate Apex basis vectors here if you want to get magnitude!"

        # Rratio = -1.e-6/MU0
        Rratio = REFRE/(REFRE+110)
        if mlat is None or mlt is None:
            Ed1 = - Rratio/self.coslambda_vector * \
                ( - np.dot(self.tor_P_vector * self.m_T * self.tor_sinmphi_vector, self.tor_c ) \
                  + np.dot(self.tor_P_vector * self.m_T * self.tor_cosmphi_vector, self.tor_s )) 
    
            Ed2 = Rratio/self.sinI_vector * \
                (   np.dot(self.tor_dP_vector * self.tor_cosmphi_vector, self.tor_c) \
                  + np.dot(self.tor_dP_vector * self.tor_sinmphi_vector, self.tor_s))

            if return_emphi_emlambda:
            
                sinI      = 2 * np.sin(self.vectorgrid[0].flatten() * d2r)/ \
                    np.sqrt(4-3*self.coslambda_vector.flatten()**2)

                sinI = sinI.reshape(Ed2.shape)

                Emphi = Ed1             # eastward component
                Emlambda = -Ed2 * sinI  # northward component, trur eg

                return Emphi.flatten(), Emlambda.flatten()
            
            else:
                if return_magnitude:
                    Emag = np.sqrt(np.sum((Ed1*d1)**2+(Ed2*d2)**2 + Ed1*Ed2*d1*d2,axis=0))
                    return Ed1.flatten(), Ed2.flatten(), Emag.flatten()
                else:
                    return Ed1.flatten(), Ed2.flatten()


        else: # calculate at custom mlat, mlt
            if grid:
                warnings.warn("Custom mlat/mlt is untested!")
                assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

                P, dP = self.legendrelike_func(self.N, self.M, 90 - mlat)
                P  =  np.transpose(np.array([ P[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                dP = -np.transpose(np.array([dP[ key] for key in self.keys_T]), (1,2,0)) # (nlat, 1, 257)
                mlt = mlt.reshape(1,-1,1)
                mlat = mlat.reshape(-1, 1, 1)
                n_T, m_T = self.n_T[np.newaxis, ...], self.m_T[np.newaxis, ...] # (1, 1, 257)

                coslambda = np.cos(      mlat * d2r) # (nmlat, 1   , 177)                
                cosmphi   = np.cos(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinmphi   = np.sin(m_T *  mlt * np.pi/12 ) # (1, nmlt, 257)
                sinI      = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*coslambda**2)

                Ed1 = - Rratio/coslambda * \
                    ( - np.dot(P * m_T * sinmphi,self.tor_c) \
                      + np.dot(P * m_T * cosmphi,self.tor_s))

                Ed2 = Rratio/sinI * \
                    (   np.dot(dP * cosmphi,self.tor_c) \
                      + np.dot(dP * sinmphi,self.tor_s))

                if return_emphi_emlambda:
                
                    
                    warnings.warn("This is untested!")
                    Emphi = Ed1             # eastward component
                    Emlambda = -Ed2 * sinI  # northward component, trur eg
                
                    return Emphi.squeeze(), Emlambda.squeeze()
                
                else:
                    if return_magnitude:
                        Emag = np.sqrt(np.sum((Ed1*d1)**2+(Ed2*d2)**2 + Ed1*Ed2*d1*d2,axis=0))
                        return Ed1.squeeze(), Ed2.squeeze(), Emag.squeeze()
                    else:
                        return Ed1.squeeze(), Ed2.squeeze()

            else:
                shape = mlat.shape
                mlat = mlat.flatten()[:, np.newaxis]
                mlt  = mlt.flatten()[ :, np.newaxis]
                n_T, m_T = self.n_T[np.newaxis, ...], self.m_T[np.newaxis, ...] # (1, 1, 257)
                P, dP = self.legendrelike_func(self.N, self.M, 90 - mlat)
                P  =  np.array([ P[ key] for key in self.keys_T]).T.squeeze()
                dP = -np.array([dP[ key] for key in self.keys_T]).T.squeeze()
                cosmphi   = np.cos(self.m_T *  mlt * np.pi/12 )
                sinmphi   = np.sin(self.m_T *  mlt * np.pi/12 )
                coslambda = np.cos(           mlat * d2r)
                sinI      = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*coslambda**2)

                Ed1 = - Rratio/coslambda * \
                    ( - np.dot(P * m_T * sinmphi,self.tor_c) \
                      + np.dot(P * m_T * cosmphi,self.tor_s))

                Ed2 = Rratio/sinI * \
                    (   np.dot(dP * cosmphi,self.tor_c) \
                      + np.dot(dP * sinmphi,self.tor_s))


                if return_emphi_emlambda:
                
                    sinI      = 2 * np.sin(self.vectorgrid[0].flatten() * d2r)/ \
                        np.sqrt(4-3*self.coslambda_vector**2)
                
                    Emphi = Ed1             # eastward component
                    Emlambda = -Ed2 * sinI  # northward component, trur eg
                
                    warnings.warn("This is untested!")
                    return Emphi.reshape(shape), Emlambda.reshape(shape)
                
                else:
                    if return_magnitude:
                        Emag = np.sqrt(np.sum((Ed1*d1)**2+(Ed2*d2)**2 + Ed1*Ed2*d1*d2,axis=0))
                        return Ed1.reshape(shape), Ed2.reshape(shape), Emag.reshape(shape)
                    else:
                        return Ed1.reshape(shape), Ed2.reshape(shape)


    def get_convection_vel_MA(self, mlat = None, mlt = None,
                              heights = None,
                              times = None,
                              apex_refdate=datetime(2020,1,1),
                              apex_refheight=110,
                              grid = False,
                              return_magnitude=False):
        """ 
        Calculate the convection velocity, in m/s, along Modified Apex basevectors e1 and e2.
        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.
        apex_refheight : scalar, default 110
            Reference altitude to use for calculating geodetic latitude and longitude from Apex latitude and magnetic local time
        apex_refdate : datetime, default datetime(2020,1,1)
            Reference date for initializing apexpy.Apex object

        Return
        ------
        ve1 : numpy.ndarray, float
            Component of the convection velocity along MA(110) base vector e1, which is "more-or-less in the magnetic eastward direction" (Just below Eq. 3.18 in Richmond, 1995)
        ve2 : numpy.ndarray, float
            Component of the convection velocity along MA(110) base vector e2, which is "generally downward and/or equatorward" (Just below Eq. 3.18 in Richmond, 1995)

        See Also
        --------
        """

        Rratio = REFRE/(REFRE+110)
        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid

        Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid)

        if times is None:
            times = np.array([datetime.now()]*len(mlat)).astype(np.datetime64)

        if heights is None:
            heights = np.array([apex_refheight]*len(mlat))

        a = apexpy.Apex(apex_refdate,apex_refheight)

        mlon = mlt_to_mlon(mlt, times, apex_refdate.year)

        glat, glon, error = a.apex2geo(mlat.flatten(),mlon,heights)
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(mlat.flatten(),mlon,heights,coords='apex')
        d10,d11, d12 = d1
        d20,d21, d22 = d2
        D = np.sqrt( (d11*d22-d12*d21)**2 + (d12*d20-d10*d22)**2 + (d10*d21-d11*d20)**2)

        # Get B0
        Be,Bn,Bu = ppigrf.igrf(glon,glat,heights,apex_refdate)
        B0 = np.sqrt(Be**2+Bn**2+Bu**2).ravel()
        Be3 = B0/D*1e-9     # in T

        ve1 =  Ed2.ravel()/Be3/1000   # in V/m
        ve2 = -Ed1.ravel()/Be3/1000   # in V/m

        if return_magnitude:
            vmag = np.sqrt(np.sum((ve1*e1)**2 + (ve2*e2)**2 + ve1*ve2*e1*e2,axis=0))
            return ve1.flatten(), ve2.flatten(), vmag.flatten()
        else:
            return ve1.flatten(), ve2.flatten()

        # else: # calculate at custom mlat, mlt
        #     if grid:
        #         warnings.warn("Custom mlat/mlt is untested!")
        #         assert len(mlat.shape) == len(mlt.shape) == 1 # enforce 1D input arrays

        #         return ve1.squeeze(), ve2.squeeze()

        #     else:
        #         warnings.warn("Custom mlat/mlt is untested!")
        #         shape = mlat.shape
        #         mlat = mlat.flatten()[:, np.newaxis]
        #         mlt  = mlt.flatten()[ :, np.newaxis]

        #         return ve1.reshape(shape), ve2.reshape(shape)


    # def get_joule_dissipation(self, mlat = None, mlt = None, grid = False):
    #     """
    #     """
    #     print("Aha! I don't think we can calculate Joule dissipation with AMPS and Swarm Hi-C. But we can calculate work. I'm redirecting your request to SWIPE.get_emwork()")
    #     return self.get_emwork(mlat = mlat, mlt = mlt, grid = grid)


    def get_emwork(self, mlat = None, mlt = None, grid = False):
        """ 
        Calculate E&M work, in mW/m^2, from the Swarm Hi-C and AMPS models.
        E&M work is the term J.E in Poynting's theorem, and depends on reference frame.

        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 
        Because of this, we have to make the same assumption here. To do this, I *believe* 
        we can use the definitions of Emphi and Emlambda given as Eqs 5.9 and 5.10 in
        Richmond (1995).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Return
        ------
        SigmaP : numpy.ndarray, float
            Pedersen conductance, calculated used Sigma_P = J_perp . E_perp / |E_perp|^2

        See Also
        --------
        """

        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            mlat, mlt = mlat.ravel(), mlt.ravel()

            sinI = self.sinI_vector.copy().ravel()

        else:
            sinI = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*np.cos(mlat * d2r)**2)

            # mlat, mlt = self.scalargrid
            # sinI = self.sinI_scalar.copy().ravel()
            # mlat, mlt = mlat.ravel(), mlt.ravel()

        J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)

        Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid)

        # from mV/m to V/m
        Ed1 /= 1000.
        Ed2 /= 1000.

        # Ed2[mlat.ravel() < 0] = Ed2[mlat.ravel() < 0]*(-1)
        # Ed2[mlat.ravel() > 0] = Ed2[mlat.ravel() > 0]*(-1)  # Do THIS LINE if not doing the Emphi/Emlambda greie below
        # return (J_e * Ed1 + J_n * Ed2)/np.sqrt(Ed1**2+Ed2**2)


        # Use Emphi and Emlambda as approximations when assuming apex coordinates are orthogonal spherical coordinates (see Eqs 5.9 and 5.10 in Richmond, 1995)
        Emphi = Ed1             # eastward component
        Emlambda = -Ed2 * sinI  # northward component, trur eg

        return self._emwork_func(J_e, J_n, Emphi, Emlambda)


    def get_conductances(self, mlat = None, mlt = None, grid = False):
        """
        Calculate Hall and Pedersen conductances, in mho, from the Swarm Hi-C and AMPS models.
        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 
        Because of this, we have to make the same assumption here. To do this, I *believe* 
        we can use the definitions of Emphi and Emlambda given as Eqs 5.9 and 5.10 in
        Richmond (1995).

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Return
        ------
        SigmaH : numpy.ndarray, float
            Hall conductance, calculated used Sigma_H = rhat . (J_perp x E_perp) / |E_perp|^2

        SigmaP : numpy.ndarray, float
            Pedersen conductance, calculated used Sigma_P = J_perp . E_perp / |E_perp|^2

        mask : numpy.ndarray, bool
            Mask indicating where the reconstructed picture of average electrodynamics is inconsistent with the assumed neutral wind pattern (corotation with Earth by default)

        """
        
        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            mlat, mlt = mlat.ravel(), mlt.ravel()

            sinI = self.sinI_vector.copy().ravel()

        else:
            sinI = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*np.cos(mlat * d2r)**2)

        J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)

        Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid)

        # from mV/m to V/m
        Ed1 /= 1000.
        Ed2 /= 1000.

        # Use Emphi and Emlambda as approximations when assuming apex coordinates are orthogonal spherical coordinates (see Eqs 5.9 and 5.10 in Richmond, 1995)
        Emphi = Ed1             # eastward component
        Emlambda = -Ed2 * sinI  # northward component, trur eg

        SigmaH = self._sigmahall_func(J_e, J_n, Emphi, Emlambda, mlat)

        SigmaP = self._sigmaped_func(J_e, J_n, Emphi, Emlambda)

        mask = self._inconsistency_mask(J_e,J_n,
                                        Emphi,Emlambda,
                                        mlat,
        )

        return SigmaH, SigmaP, mask


    def get_pedersen_conductance(self, mlat = None, mlt = None, grid = False):
        """ 
        Calculate the Pedersen conductance, in mho, from the Swarm Hi-C and AMPS models.
        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 
        Because of this, we have to make the same assumption here. To do this, I *believe* 
        we can use the definitions of Emphi and Emlambda given as Eqs 5.9 and 5.10 in
        Richmond (1995).


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Return
        ------
        SigmaP : numpy.ndarray, float
            Pedersen conductance, calculated used Sigma_P = J_perp . E_perp / |E_perp|^2

        See Also
        --------
        """

        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            mlat, mlt = mlat.ravel(), mlt.ravel()

            sinI = self.sinI_vector.copy().ravel()

        else:
            sinI = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*np.cos(mlat * d2r)**2)

        J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)

        Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid)

        # from mV/m to V/m
        Ed1 /= 1000.
        Ed2 /= 1000.

        # Ed2[mlat.ravel() < 0] = Ed2[mlat.ravel() < 0]*(-1)
        # Ed2[mlat.ravel() > 0] = Ed2[mlat.ravel() > 0]*(-1)  # Do THIS LINE if not doing the Emphi/Emlambda greie below
        # return (J_e * Ed1 + J_n * Ed2)/np.sqrt(Ed1**2+Ed2**2)


        # Use Emphi and Emlambda as approximations when assuming apex coordinates are orthogonal spherical coordinates (see Eqs 5.9 and 5.10 in Richmond, 1995)
        Emphi = Ed1             # eastward component
        Emlambda = -Ed2 * sinI  # northward component, trur eg

        SigmaP = self._sigmaped_func(J_e, J_n, Emphi, Emlambda)

        return SigmaP


    def get_hall_conductance(self, mlat = None, mlt = None, grid = False,
    ):
        """ 
        Calculate the Hall conductance, in mho, from the Swarm Hi-C and AMPS models.
        The calculations refer to the height chosen upon initialization of the AMPS 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 
        Because of this, we have to make the same assumption here. To do this, I *believe* 
        we can use the definitions of Emphi and Emlambda given as Eqs 5.9 and 5.10 in
        Richmond (1995).


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Return
        ------
        SigmaH : numpy.ndarray, float
            Hall conductance, calculated used Sigma_H = rhat . (J_perp x E_perp) / |E_perp|^2

        mask : numpy.ndarray, bool
            Mask indicating where the reconstructed picture of average electrodynamics is inconsistent with the assumed neutral wind pattern (corotation with Earth by default)
        

        See Also
        --------
        """

        # warnings.warn("Expression for Hall conductance should have sign flipped in SH! See Equation (8) and immediately following text in Amm (2001).")

        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            mlat, mlt = mlat.ravel(), mlt.ravel()

            sinI = self.sinI_vector.copy().ravel()

        else:
            sinI = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*np.cos(mlat * d2r)**2)

        J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)

        Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid)

        # from mV/m to V/m
        Ed1 /= 1000.
        Ed2 /= 1000.

        # Ed2[mlat.ravel() < 0] = Ed2[mlat.ravel() < 0]*(-1)
        # Ed2[mlat.ravel() > 0] = Ed2[mlat.ravel() > 0]*(-1)  # Do THIS LINE if not doing the Emphi/Emlambda greie below
        # return (J_e * Ed1 + J_n * Ed2)/np.sqrt(Ed1**2+Ed2**2)

        # Use Emphi and Emlambda as approximations when assuming apex coordinates are orthogonal spherical coordinates (see Eqs 5.9 and 5.10 in Richmond, 1995)
        Emphi = Ed1             # eastward component
        Emlambda = -Ed2 * sinI  # northward component, trur eg

        SigmaH = self._sigmahall_func(J_e, J_n, Emphi, Emlambda, mlat)
        return SigmaH


    def get_cowling_conductance(self, mlat = None, mlt = None, grid = False):
        """ 
        Calculate the Cowling conductance, in mho, from the Swarm Hi-C and AMPS models.
        This is done using the 'get_pedersen_conductance' and 'get_hall_conductance' SWIPE methods .
        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 
        Because of this, we have to make the same assumption here. To do this, I *believe* 
        we can use the definitions of Emphi and Emlambda given as Eqs 5.9 and 5.10 in
        Richmond (1995).


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Return
        ------
        SigmaC : numpy.ndarray, float
            Cowling conductance, calculated via SigmaC = SigmaP+SigmaH²/SigmaP

        See Also
        --------
        """

        SigmaP = self.get_pedersen_conductance(mlat = mlat, mlt = mlt, grid = grid)

        SigmaH = self.get_hall_conductance(mlat = mlat, mlt = mlt, grid = grid)
        
        return SigmaP+SigmaH**2/SigmaP


    def get_joule_dissipation(self, mlat = None, mlt = None, grid = False):
        """ 
        Estimate Joule dissipation in mW/m², from the Swarm Hi-C and AMPS models.

        NOTE! This calculation is supposed to give _true_ Joule dissipation, which does not
        depend on one's frame of reference. (See Mannucci et al, 2022, doi:10.1029/2021JA030009).
        It is the frictional heating due to collisions between plasma and neutrals (Strangeway, 
        2012, doi:10.1029/2011JA017302).

        Where this quantity is negative, it is because the estimate of the Pedersen conductance is 
        negative. This is a sign that either that the conductance-weighted neutral wind velocity 
        cannot be neglected, or that the model uncertainties are too great.

        When this quantity is similar to the output from get_emwork for the same input location, it 
        is a sign that the conductance-weighted neutral wind velocity is small in Earth's corotating 
        frame of reference (which is the frame of reference in which all Swarm Hi-C/AMPS calculations are 
        made). This means that the estimates of Pedersen and Hall conductance at this location are 
        good.

        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). Modeled after get_curl_free_current. (2021/09/03)

        IMPORTANT: The AMPS model assumes that apex coordinates are spherical and orthogonal 
        in calculating J_perp [see first paragraph of Appendix B in Laundal et al (2018)]. 
        Because of this, we have to make the same assumption here. To do this, I *believe* 
        we can use the definitions of Emphi and Emlambda given as Eqs 5.9 and 5.10 in
        Richmond (1995).


        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.

        Return
        ------
        JD : numpy.ndarray, float
            Joule dissipation, calculated using JD = |J|²/Sigma_C

        See Also
        --------
        """

        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            mlat, mlt = mlat.ravel(), mlt.ravel()

        J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)

        SigmaC = self.get_cowling_conductance(mlat = mlat, mlt = mlt, grid = grid)

        Jsq = J_e**2 + J_n**2

        return Jsq / SigmaC * 1000  # from W/m² to mW/m²


    def get_poynting_flux(self, mlat = None, mlt = None,
                          times = None,
                          heights = None,
                          apex_refdate=datetime(2020,1,1),
                          apex_refheight=110,
                          grid = False,
                          killpoloidalB=False):
        """ 
        Calculate the Poynting flux, in mW/m^2, along Modified Apex basevectors e1, e2, and e3 at one point in time.
        The calculations refer to the height chosen upon initialization of the SWIPE 
        object (default 110 km). 
        2021/11/24

        killpoloidalB added because the divergence of Poynting flux given by an E-field and a B-field that are represented by gradients of scalar potentials is zero. Thus the contribution to the divergence of Poynting flux from poloidal ΔB perturbations (at least when ΔB^pol = - grad(V) with V a scalar) is zero.

        Parameters
        ----------
        mlat : numpy.ndarray, optional
            array of mlats at which to calculate the current. Will be ignored if mlt is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        mlt : numpy.ndarray, optional
            array of mlts at which to calculate the current. Will be ignored if mlat is not also specified. If 
            not specified, the calculations will be done using the coords of the `vectorgrid` attribute.
        times : array-like of datetime-like objects for calculating
        apex_refheight : scalar, default 110
            Reference altitude to use for calculating geodetic latitude and longitude from Apex latitude and magnetic local time
        apex_refdate : datetime, default datetime(2020,1,1)
            Reference date for initializing apexpy.Apex object
        grid : bool, optional, default False
            if True, mlat and mlt are interpreted as coordinates in a regular grid. They must be 
            1-dimensional, and the output will have dimensions len(mlat) x len(mlt). If mlat and mlt 
            are not set, this keyword is ignored.


        Return
        ------
        pfluxe1 : numpy.ndarray, float
            Component of the Poynting flux along MA(110) base vector e1, which is "more-or-less in the magnetic eastward direction" (Just below Eq. 3.18 in Richmond, 1995)
        pfluxe2 : numpy.ndarray, float
            Component of the electric field along MA(110) base vector e2, which is "generally downward and/or equatorward" (Just below Eq. 3.18 in Richmond, 1995)
        pfluxpar : numpy.ndarray, float
            Component of the electric field along MA(110) base vector e3, which is (I believe?) along the main geomagnetic field

        See Also
        --------
        """

        # Get mlat, mlt for AMPS
        if mlat is None or mlt is None:
            mlat, mlt = self.vectorgrid
            sinI = self.sinI_vector.copy().ravel()
            mlat,mlt = mlat.ravel(),mlt.ravel()
            # mlat, mlt = self.scalargrid
            # sinI = self.sinI_scalar.copy().ravel()
            # J_e, J_n = amp.get_total_current(grid=grid)

        else:
            mlat,mlt = mlat.ravel(),mlt.ravel()
            # assert 2<0,"Here you need to calc sinI manually. Make sure this is OK."
            sinI = 2 * np.sin(mlat * d2r)/np.sqrt(4-3*np.cos(mlat * d2r)**2)
            # J_e, J_n = amp.get_total_current(mlat,mlt,grid=grid)

        # Convert mlat, mlt to glat, glon
        epoch = apex_refdate.year+apex_refdate.timetuple().tm_yday/365
        # epoch = time.year+time.timetuple().tm_yday/365
        # mlon = mlt_to_mlon(mlt, [time]*mlt.size, epoch)
        
        if times is None:
            times = np.array([datetime.now()]*len(mlat)).astype(np.datetime64)
        elif not hasattr(times,'__len__'):
            times = np.array([times]*mlat.size).astype(np.datetime64)

        if heights is None:
            map_Efield = False
            heights = np.array([apex_refheight]*len(mlat))
        else:
            map_Efield = True

        a = apexpy.Apex(apex_refdate,apex_refheight)
        mlon = mlt_to_mlon(mlt, times, epoch)
        glat, glon, error = a.apex2geo(mlat.flatten(),mlon,heights.flatten())
        # f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(mlat.flatten(),mlon,heights,coords='apex')

        _, _, _, _, _, _, d1gnd, d2gnd, d3gnd, e1gnd, e2gnd, e3gnd = a.basevectors_apex(
            mlat.flatten(),mlon,np.array([apex_refheight]*len(mlat)),
            coords='apex')

        ####################
        # Get E-field
        Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid)

        # To geodetic coords
        E_Egeod = Ed1*d1gnd[0,:]+Ed2*d2gnd[0,:]
        E_Ngeod = Ed1*d1gnd[1,:]+Ed2*d2gnd[1,:]
        E_Ugeod = Ed1*d1gnd[2,:]+Ed2*d2gnd[2,:]
        Egeod = np.vstack([E_Egeod,E_Ngeod,E_Ugeod])/1e3  # convert from mV/m  to V/m

        # Map E-field if heights are not apex_refheight
        if map_Efield:
            warnings.warn("Mapping of E-field (or something else?) is currently incorrect!!\nI think what's happening is that you use Apex.map_E_to_height to map the E-field to a higher altitude, when this is already done for you by virtue of the Apex basevectors. Consult with Kalle")
            # breakpoint()
            Egeodold = Egeod.copy()
            Egeod = a.map_E_to_height(mlat.flatten(), mlon.flatten(), apex_refheight,
                                      heights.flatten(),
                                      Egeodold)

            f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(
                mlat.flatten(),mlon.flatten(),heights.flatten(),coords='apex')

        else:
            
            d1, d2, d3, e1, e2, e3 = d1gnd, d2gnd, d3gnd, e1gnd, e2gnd, e3gnd


        ####################
        # Get ΔB perturbations

        # Initialize amps
        from pyamps import amps
        # amp = amps.AMPS(self.inputs['v'],
        #                 self.inputs['By'],
        #                 self.inputs['Bz'],
        #                 self.inputs['tilt'],
        #                 self.inputs['f107'],
        #                 minlat=self.inputs['minlat'],
        #                 maxlat=self.inputs['maxlat'],
        #                 height=self.inputs['height'],
        #                 dr=self.inputs['dr'],
        #                 M0=self.inputs['M0'],
        #                 resolution=self.inputs['resolution'])


        Be, Bn, Bu = amps.get_B_space(glat, glon, heights.flatten(), times,
                                      np.array([self.inputs['v'   ]]*len(glat)),
                                      np.array([self.inputs['By'  ]]*len(glat)),
                                      np.array([self.inputs['Bz'  ]]*len(glat)),
                                      np.array([self.inputs['tilt']]*len(glat)),
                                      np.array([self.inputs['f107']]*len(glat)),
                                      epoch=epoch,
                                      h_R=apex_refheight,
                                      killpoloidal=killpoloidalB)
        deltaBgeod = np.vstack([Be,Bn,Bu])/1e9            # convert from nT to T

        ####################
        # Calculate Poynting flux
        PFlux_geod = np.cross(Egeod.T,deltaBgeod.T).T/MU0*1e3  # convert from W/m^2 to mW/m^2
        pfluxe1 = e1[0,:]*PFlux_geod[0,:]+e1[1,:]*PFlux_geod[1,:]+e1[2,:]*PFlux_geod[2,:]
        pfluxe2 = e2[0,:]*PFlux_geod[0,:]+e2[1,:]*PFlux_geod[1,:]+e2[2,:]*PFlux_geod[2,:]
        pfluxpar = e3[0,:]*PFlux_geod[0,:]+e3[1,:]*PFlux_geod[1,:]+e3[2,:]*PFlux_geod[2,:]
        
        return pfluxe1,pfluxe2,pfluxpar


    def plot_potential(self,
                       convection=False,
                       vector_scale=None,
                       minlat_for_cpcp_calc = 65,
                       flip_panel_order=False,
                       vmin=None,
                       vmax=None):
        """ 
        Create a summary plot of the ionospheric potential and electric field

        Parameters
        ----------
        convection           : boolean (default False)
            Show convection velocity (in m/s) instead of convection electric field
        vector_scale         : optional
            Vector lengths are shown relative to a template. This parameter determines
            the magnitude of that template, in mV/m. Default is 20 mV/m
        minlat_for_cpcp_calc : float (default 65)
            Minimum latitude allowed for determination of min/max potential when 
            calculating the cross-polar cap potential

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_potential()

        """

        # get the grids:
        mlats, mlts = self.plotgrid_scalar
        mlatv, mltv = self.plotgrid_vector

        # set up figure and polar coordinate plots:
        fig = plt.figure(figsize = (15, 7))
        if flip_panel_order:
            pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
        else:
            pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
        pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
        
        # labels
        pax_n.writeLTlabels(lat = self.minlat, size = 14)
        pax_s.writeLTlabels(lat = self.minlat, size = 14)
        pax_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' , ha = 'left', va = 'top', size = 14)
        pax_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$', ha = 'left', va = 'top', size = 14)
        # pax_n.write(self.minlat-5, 12, r'North' , ha = 'center', va = 'center', size = 18)
        # pax_s.write(self.minlat-5, 12, r'South' , ha = 'center', va = 'center', size = 18)
        self._make_pax_n_label(pax_n)
        self._make_pax_s_label(pax_s)

        # OLD
        # # calculate and plot FAC
        # Jun, Jus = np.split(self.get_upward_current(), 2)
        # faclevels = np.r_[-.925:.926:.05]
        # pax_n.contourf(mlats, mlts, Jun, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')
        # pax_s.contourf(mlats, mlts, Jus, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')

        # Convection electric field
        if not convection:
            if vector_scale is None:
                vector_scale = 20  # mV/m
            Ed1, Ed2 = self.get_efield_MA()
            Ed1NH, Ed1SH = np.split(Ed1, 2)
            Ed2NH, Ed2SH = np.split(Ed2, 2)
            # nn, ns = np.split(j_n, 2)
            # en, es = np.split(j_e, 2)
            pax_n.plotpins(mlatv, mltv, -Ed2NH, Ed1NH, SCALE = vector_scale, markersize = 10, unit = None, linewidth = .5, color = 'gray', markercolor = 'grey')
            pax_s.plotpins(mlatv, mltv, -Ed2SH, Ed1SH, SCALE = vector_scale, markersize = 10, unit = 'mV/m', linewidth = .5, color = 'gray', markercolor = 'grey')
        else:
            if vector_scale is None:
                vector_scale = 300  # m/s
            ve1, ve2 = self.get_convection_vel_MA()
            ve1NH, ve1SH = np.split(ve1, 2)
            ve2NH, ve2SH = np.split(ve2, 2)
            # nn, ns = np.split(j_n, 2)
            # en, es = np.split(j_e, 2)
            pax_n.plotpins(mlatv, mltv, -ve2NH, ve1NH, SCALE = vector_scale, markersize = 10, unit = 'm/s', linewidth = .5, color = 'gray', markercolor = 'grey')
            pax_s.plotpins(mlatv, mltv, -ve2SH, ve1SH, SCALE = vector_scale, markersize = 10, unit = None  , linewidth = .5, color = 'gray', markercolor = 'grey')

        # NEW
        # calculate and plot potential
        phin, phis = np.split(self.get_potential(), 2)
        phin = phin - np.median(phin)
        phis = phis - np.median(phis)
        # potlevels = np.r_[-.4:.4:.025]
        # potlevels = np.r_[-8.5:8.6:1]

        # minpot,maxpot = -25.5,26
        # potlevelscoarse = np.r_[minpot:maxpot:5]
        # potlevels = np.r_[minpot:maxpot:.25]

        if vmin is None and vmax is None:
            minpot = np.min([np.quantile(phin,0.03),np.quantile(phis,0.03)])
            maxpot = np.max([np.quantile(phin,0.97),np.quantile(phis,0.97)])

            maxpot = np.max(np.abs([minpot,maxpot]))
            minpot = -maxpot

        else:
            if vmin is not None:
                minpot = vmin
            else:
                minpot = np.min([np.quantile(phin,0.03),np.quantile(phis,0.03)])
            
            if vmax is not None:            
                maxpot = vmax
            else:
                maxpot = np.max([np.quantile(phin,0.97),np.quantile(phis,0.97)])

        potlevelscoarse = np.r_[minpot:maxpot:5]
        potlevels = np.r_[minpot:maxpot:.25]

        pax_n.contourf(mlats, mlts, phin, levels = potlevels, cmap = plt.cm.bwr, extend = 'both')
        pax_s.contourf(mlats, mlts, phis, levels = potlevels, cmap = plt.cm.bwr, extend = 'both')

        opts__contour = dict(levels = potlevelscoarse, linestyles='solid', colors='black', linewidths=1)
        pax_n.contour(mlats, mlts, phin, **opts__contour)
        pax_s.contour(mlats, mlts, phis, **opts__contour)

        # OLD (no limit on minlat for cpcp)
        dPhiN = phin.max()-phin.min()
        dPhiS = phis.max()-phis.min()
        minn,maxn = np.argmin(phin),np.argmax(phin)
        mins,maxs = np.argmin(phis),np.argmax(phis)
        cpcpmlats,cpcpmlts = mlats,mlts

        # NEW (yes limit on minlat for cpcp)        
        OKinds = mlats.flatten() > minlat_for_cpcp_calc
        cpcpmlats,cpcpmlts = mlats.flatten()[OKinds],mlts.flatten()[OKinds]
        dPhiN = phin[OKinds].max()-phin[OKinds].min()
        dPhiS = phis[OKinds].max()-phis[OKinds].min()
        minn,maxn = np.argmin(phin[OKinds]),np.argmax(phin[OKinds])
        mins,maxs = np.argmin(phis[OKinds]),np.argmax(phis[OKinds])

        pax_n.write(self.minlat-6, 2, r'$\Delta \Phi = $'+f"{dPhiN:.1f} kV" ,
                    ha='center',va='center',size=18,ignore_plot_limits=True)
        pax_s.write(self.minlat-6, 2, r'$\Delta \Phi = $'+f"{dPhiS:.1f} kV" ,
                    ha='center',va='center',size=18,ignore_plot_limits=True)
        
        pax_n.write(cpcpmlats[minn],cpcpmlts[minn],r'x',
                    ha='center',va='center',size=18,ignore_plot_limits=True)
        pax_n.write(cpcpmlats[maxn],cpcpmlts[maxn],r'+',
                    ha='center',va='center',size=18,ignore_plot_limits=True)

        pax_s.write(cpcpmlats[mins],cpcpmlts[mins],r'x',
                    ha='center',va='center',size=18,ignore_plot_limits=True)
        pax_s.write(cpcpmlats[maxs],cpcpmlts[maxs],r'+',
                    ha='center',va='center',size=18,ignore_plot_limits=True)

        # colorbar
        pax_c.contourf(np.vstack((np.zeros_like(potlevels), np.ones_like(potlevels))), 
                       np.vstack((potlevels, potlevels)), 
                       np.vstack((potlevels, potlevels)), 
                       levels = potlevels, cmap = plt.cm.bwr)
        pax_c.set_xticks([])
        # pax_c.set_ylabel(r'downward    $\mu$A/m$^2$      upward', size = 18)
        pax_c.set_ylabel(r'neg    kV      pos', size = 18)
        pax_c.yaxis.set_label_position("right")
        pax_c.yaxis.tick_right()

        # print AL index values and integrated up/down currents
        # AL_n, AL_s, AU_n, AU_s = self.get_AE_indices()
        # ju_n, jd_n, ju_s, jd_s = self.get_integrated_upward_current()

        # pax_n.ax.text(pax_n.ax.get_xlim()[0], pax_n.ax.get_ylim()[0], 
        #               'AL: \t${AL_n:+}$ nT\nAU: \t${AU_n:+}$ nT\n $\int j_{uparrow:}$:\t ${jn_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${jn_down:+.1f}$ MA'.format(AL_n = int(np.round(AL_n)), AU_n = int(np.round(AU_n)), jn_up = ju_n, jn_down = jd_n, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)
        # pax_s.ax.text(pax_s.ax.get_xlim()[0], pax_s.ax.get_ylim()[0], 
        #               'AL: \t${AL_s:+}$ nT\nAU: \t${AU_s:+}$ nT\n $\int j_{uparrow:}$:\t ${js_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${js_down:+.1f}$ MA'.format(AL_s = int(np.round(AL_s)), AU_s = int(np.round(AU_s)), js_up = ju_s, js_down = jd_s, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)

        plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (pax_n, pax_s, pax_c)

    def plot_Pedersen(self,
                      flip_panel_order=False,
                      vmin=None,
                      vmax=None,
                      cmap=None):
        """ 
        Create a summary plot of the ionospheric pedersen conductance

        Parameters
        ----------

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_Pedersen()

        """

        # Want the vector grid? Uncomment here
        # mlatv, mltv = self.plotgrid_vector
        # mlat,mlt = mlatv,mltv
        # SigmaP = self.get_pedersen_conductance()

        # Want the scalar grid? Uncomment here
        mlat,mlt = self.scalargrid
        mlat,mlt = mlat.ravel(),mlt.ravel()
        SigmaP = self.get_pedersen_conductance(mlat,mlt)

        SigmaPN,SigmaPS = np.split(SigmaP,2)

        # set up figure and polar coordinate plots:
        fig = plt.figure(figsize = (15, 7))
        if flip_panel_order:
            pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
        else:
            pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
        pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
        
        # labels
        pax_n.writeLTlabels(lat = self.minlat, size = 14)
        pax_s.writeLTlabels(lat = self.minlat, size = 14)
        pax_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' ,
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        pax_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$',
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        self._make_pax_n_label(pax_n)
        self._make_pax_s_label(pax_s)

        # if vector_scale is None:
        #     vector_scale = 20  # mV/m
        # siglevels = np.linspace(SigmaP.min(),SigmaP.max(),31)
        siglevels = np.linspace(np.quantile(SigmaP,0.01),np.quantile(SigmaP,0.99),31)

        if vmin is None:
            sigmin = np.clip(np.quantile(SigmaP,0.01),-0.5,0)
        else:
            sigmin = vmin

        if vmax is None:
            sigmax = np.quantile(SigmaP,0.975)
        else:
            sigmax = vmax

        siglevels = np.linspace(sigmin,sigmax,31)

        if cmap is None:
            cmapper = plt.cm.magma
        else:
            cmapper = cmap
        # cmapper = plt.cm.bwr
        # pax_n.contourf(mlatv, mltv, np.abs(SigmaPN),levels=siglevels,cmap=cmapper,extend='both')
        # pax_s.contourf(mlatv, mltv, np.abs(SigmaPS),levels=siglevels,cmap=cmapper,extend='both')
        if np.isclose(mlat.size/SigmaPN.size,2):
            pax_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaPN,
                           levels=siglevels,cmap=cmapper,extend='both')
            pax_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaPS,
                           levels=siglevels,cmap=cmapper,extend='both')
        else:
            pax_n.contourf(mlat, mlt, SigmaPN,levels=siglevels,cmap=cmapper,extend='both')
            pax_s.contourf(mlat, mlt, SigmaPS,levels=siglevels,cmap=cmapper,extend='both')

        # colorbar
        pax_c.contourf(np.vstack((np.zeros_like(siglevels), np.ones_like(siglevels))), 
                       np.vstack((siglevels, siglevels)), 
                       np.vstack((siglevels, siglevels)), 
                       levels = siglevels, cmap = cmapper)
        pax_c.set_xticks([])
        # pax_c.set_ylabel(r'downward    $\mu$A/m$^2$      upward', size = 18)
        pax_c.set_ylabel(r'$\Sigma_P$ [mho]', size = 18)
        pax_c.yaxis.set_label_position("right")
        pax_c.yaxis.tick_right()

        # print AL index values and integrated up/down currents
        # AL_n, AL_s, AU_n, AU_s = self.get_AE_indices()
        # ju_n, jd_n, ju_s, jd_s = self.get_integrated_upward_current()

        # pax_n.ax.text(pax_n.ax.get_xlim()[0], pax_n.ax.get_ylim()[0], 
        #               'AL: \t${AL_n:+}$ nT\nAU: \t${AU_n:+}$ nT\n $\int j_{uparrow:}$:\t ${jn_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${jn_down:+.1f}$ MA'.format(AL_n = int(np.round(AL_n)), AU_n = int(np.round(AU_n)), jn_up = ju_n, jn_down = jd_n, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)
        # pax_s.ax.text(pax_s.ax.get_xlim()[0], pax_s.ax.get_ylim()[0], 
        #               'AL: \t${AL_s:+}$ nT\nAU: \t${AU_s:+}$ nT\n $\int j_{uparrow:}$:\t ${js_up:+.1f}$ MA\n $\int j_{downarrow:}$:\t ${js_down:+.1f}$ MA'.format(AL_s = int(np.round(AL_s)), AU_s = int(np.round(AU_s)), js_up = ju_s, js_down = jd_s, uparrow = r'\uparrow',downarrow = r'\downarrow'), ha = 'left', va = 'bottom', size = 12)

        plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (pax_n, pax_s, pax_c)


    def plot_Hall(self,
                  flip_panel_order=False,
                  vmin=None,
                  vmax=None,
                  cmap=None):
        """ 
        Create a summary plot of the ionospheric Hall conductance

        Parameters
        ----------

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_Hall()

        """

        # get the grids:

        # Want the vector grid? Uncomment here
        # mlatv, mltv = self.plotgrid_vector
        # mlat,mlt = mlatv,mltv
        # SigmaH = self.get_hall_conductance()

        # Want the scalar grid? Uncomment here
        mlat,mlt = self.scalargrid
        mlat,mlt = mlat.ravel(),mlt.ravel()
        SigmaH = self.get_hall_conductance(mlat,mlt)

        SigmaHN,SigmaHS = np.split(SigmaH,2)
        SigmaHS *= -1

        # set up figure and polar coordinate plots:
        fig = plt.figure(figsize = (15, 7))
        if flip_panel_order:
            pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
        else:
            pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
        pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
        
        # labels
        pax_n.writeLTlabels(lat = self.minlat, size = 14)
        pax_s.writeLTlabels(lat = self.minlat, size = 14)
        pax_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' ,
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        pax_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$',
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        self._make_pax_n_label(pax_n)
        self._make_pax_s_label(pax_s)

        # siglevels = np.linspace(SigmaH.min(),SigmaH.max(),31)
        # siglevels = np.linspace(np.quantile(SigmaH,0.01),np.quantile(SigmaH,0.99),31)
        # siglevels = np.linspace(np.clip(np.quantile(SigmaH,0.01),-0.5,0),np.quantile(SigmaH,0.975),31)

        if vmin is None:
            sigmin = np.clip(np.quantile(SigmaH,0.01),-0.5,0)
        else:
            sigmin = vmin

        if vmax is None:
            sigmax = np.quantile(SigmaH,0.975)
        else:
            sigmax = vmax

        siglevels = np.linspace(sigmin,sigmax,31)

        if cmap is None:
            cmapper = plt.cm.magma
        else:
            cmapper = cmap
        # cmapper = plt.cm.bwr
        # pax_n.contourf(mlatv, mltv, np.abs(SigmaHN),levels=siglevels,cmap=cmapper,extend='both')
        # pax_s.contourf(mlatv, mltv, np.abs(SigmaHS),levels=siglevels,cmap=cmapper,extend='both')
        if np.isclose(mlat.size/SigmaHN.size,2):
            pax_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaHN,
                           levels=siglevels,cmap=cmapper,extend='both')
            pax_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaHS,
                           levels=siglevels,cmap=cmapper,extend='both')
        else:
            pax_n.contourf(mlat, mlt, SigmaHN,levels=siglevels,cmap=cmapper,extend='both')
            pax_s.contourf(mlat, mlt, SigmaHS,levels=siglevels,cmap=cmapper,extend='both')

        # colorbar
        pax_c.contourf(np.vstack((np.zeros_like(siglevels), np.ones_like(siglevels))), 
                       np.vstack((siglevels, siglevels)), 
                       np.vstack((siglevels, siglevels)), 
                       levels = siglevels, cmap = cmapper)
        pax_c.set_xticks([])
        pax_c.set_ylabel(r'$\Sigma_H$ [mho]', size = 18)
        pax_c.yaxis.set_label_position("right")
        pax_c.yaxis.tick_right()

        plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (pax_n, pax_s, pax_c)


    def plot_conductance(self,
                         flip_panel_order=False,
                         vmin=None,
                         vmax=None,
                         cmap=None):
        """ 
        Create a summary plot of the ionospheric Hall and Pedersen conductances

        Parameters
        ----------

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_conductance()

        """

        # get the grids:

        # Want the vector grid? Uncomment here
        mlatv, mltv = self.vectorgrid
        mlat,mlt = mlatv,mltv
        mlat,mlt = mlat.ravel(),mlt.ravel()
        # SigmaH = self.get_hall_conductance()
        # SigmaP = self.get_pedersen_conductance()

        # Want the scalar grid? Uncomment here
        # mlat,mlt = self.scalargrid
        # mlat,mlt = mlat.ravel(),mlt.ravel()
        # SigmaH = self.get_hall_conductance(mlat,mlt)
        # SigmaP = self.get_pedersen_conductance(mlat,mlt)

        SigmaH, SigmaP, mask = self.get_conductances(mlat, mlt)

        SigmaH[mask] = 0.
        SigmaP[mask] = 0.

        # SigmaH[mask] = np.nan
        # SigmaP[mask] = np.nan

        SigmaHm = np.ma.masked_array(SigmaH,mask=mask)
        SigmaPm = np.ma.masked_array(SigmaP,mask=mask)

        SigmaHN,SigmaHS = np.split(SigmaHm,2)
        # SigmaHS *= -1
        SigmaPN,SigmaPS = np.split(SigmaPm,2)

        # breakpoint()

        # set up figure and polar coordinate plots:
        fig = plt.figure(figsize = (15, 15))
        if flip_panel_order:
            paxh_s = Polarplot(plt.subplot2grid((2, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            paxh_n = Polarplot(plt.subplot2grid((2, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            paxp_s = Polarplot(plt.subplot2grid((2, 15), (1,  0), colspan = 7), **self.pax_plotopts)
            paxp_n = Polarplot(plt.subplot2grid((2, 15), (1,  7), colspan = 7), **self.pax_plotopts)
        else:
            paxh_n = Polarplot(plt.subplot2grid((2, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            paxh_s = Polarplot(plt.subplot2grid((2, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            paxp_n = Polarplot(plt.subplot2grid((2, 15), (1,  0), colspan = 7), **self.pax_plotopts)
            paxp_s = Polarplot(plt.subplot2grid((2, 15), (1,  7), colspan = 7), **self.pax_plotopts)
        paxh_c = plt.subplot2grid((2, 150), (0, 149), colspan = 1)
        paxp_c = plt.subplot2grid((2, 150), (1, 149), colspan = 1)
        
        # labels
        paxh_n.writeLTlabels(lat = self.minlat, size = 14)
        paxh_s.writeLTlabels(lat = self.minlat, size = 14)
        paxp_n.writeLTlabels(lat = self.minlat, size = 14)
        paxp_s.writeLTlabels(lat = self.minlat, size = 14)
        paxh_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' ,
                     ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        paxh_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$',
                     ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        # paxh_n.write(self.minlat-5, 12, r'North' , ha = 'center', va = 'center', size = 18)
        # paxh_s.write(self.minlat-5, 12, r'South' , ha = 'center', va = 'center', size = 18)
        self._make_pax_n_label(paxh_n)
        self._make_pax_s_label(paxh_s)

        def _get_siglevels(Sigma,vmin=None,vmax=None):
            # return np.linspace(np.quantile(Sigma,0.01),
            #                    np.quantile(Sigma,0.99),31)

            if vmin is None:
                sigmin = np.clip(np.quantile(Sigma,0.01),-0.5,0)
            else:
                sigmin = vmin
    
            if vmax is None:
                sigmax = np.quantile(Sigma,0.975)
            else:
                sigmax = vmax
    
            siglevels = np.linspace(sigmin,sigmax,31)
    
            return siglevels

        # siglevels = _get_siglevels(np.vstack([SigmaH,SigmaP]).flatten())
        # sighlevels, sigplevels = siglevels,siglevels

        sighlevels = _get_siglevels(SigmaH,
                                    vmin=vmin,
                                    vmax=vmax)
        # sigplevels = _get_siglevels(SigmaP,
        #                             vmin=vmin,
        #                             vmax=vmax)


        good = np.isfinite(SigmaH) & np.isfinite(SigmaP) & (np.abs(SigmaP) > 0)
        goodN = np.isfinite(SigmaHN) & np.isfinite(SigmaPN) & (np.abs(SigmaPN) > 0)
        goodS = np.isfinite(SigmaHS) & np.isfinite(SigmaPS) & (np.abs(SigmaPS) > 0)
        sigplevels = _get_siglevels(SigmaH[good]/SigmaP[good],
                                    vmin=np.quantile(SigmaH[good]/SigmaP[good],0.03),
                                    vmax=np.quantile(SigmaH[good]/SigmaP[good],0.97))

        if cmap is None:
            cmapper = plt.cm.magma
        else:
            cmapper = cmap
        # cmapper = plt.cm.bwr

        if np.isclose(mlat.size/SigmaHN.size,2):
            paxh_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaHN,
                           levels=sighlevels,cmap=cmapper,extend='both')
            paxh_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaHS,
                           levels=sighlevels,cmap=cmapper,extend='both')

            # paxp_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaPN,
            #                levels=sigplevels,cmap=cmapper,extend='both')
            # paxp_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaPS,
            #                levels=sigplevels,cmap=cmapper,extend='both')

            paxp_n.contourf(np.split(mlat,2)[0][goodN],np.split(mlt,2)[0][goodN],SigmaHN[goodN]/SigmaPN[goodN],
                           levels=sigplevels,cmap=cmapper,extend='both')
            paxp_s.contourf(np.split(mlat,2)[0][goodS],np.split(mlt,2)[0][goodS],SigmaHN[goodS]/SigmaPS[goodS],
                           levels=sigplevels,cmap=cmapper,extend='both')

        else:
            paxh_n.contourf(mlat, mlt, SigmaHN,levels=sighlevels,cmap=cmapper,extend='both')
            paxh_s.contourf(mlat, mlt, SigmaHS,levels=sighlevels,cmap=cmapper,extend='both')

            # paxp_n.contourf(mlat, mlt, SigmaPN,levels=sigplevels,cmap=cmapper,extend='both')
            # paxp_s.contourf(mlat, mlt, SigmaPS,levels=sigplevels,cmap=cmapper,extend='both')

            paxp_n.contourf(mlat[goodN], mlt[goodN], SigmaHN[goodN]/SigmaPN[goodN],levels=sigplevels,cmap=cmapper,extend='both')
            paxp_s.contourf(mlat[goodS], mlt[goodS], SigmaHS[goodS]/SigmaPS[goodS],levels=sigplevels,cmap=cmapper,extend='both')

        # colorbar
        paxh_c.contourf(np.vstack((np.zeros_like(sighlevels), np.ones_like(sighlevels))), 
                       np.vstack((sighlevels, sighlevels)), 
                       np.vstack((sighlevels, sighlevels)), 
                       levels = sighlevels, cmap = cmapper)

        paxp_c.contourf(np.vstack((np.zeros_like(sigplevels), np.ones_like(sigplevels))), 
                       np.vstack((sigplevels, sigplevels)), 
                       np.vstack((sigplevels, sigplevels)), 
                       levels = sigplevels, cmap = cmapper)

        # for pax_c,lab in zip([paxh_c,paxp_c],[r'$\Sigma_H$ [mho]',r'$\Sigma_P$ [mho]']):
        for pax_c,lab in zip([paxh_c,paxp_c],[r'$\Sigma_H$ [mho]',r'$\Sigma_H/\Sigma_P$ [mho]']):
            pax_c.set_xticks([])
            pax_c.set_ylabel(lab, size = 18)
            pax_c.yaxis.set_label_position("right")
            pax_c.yaxis.tick_right()

        plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (paxh_n, paxh_s, paxh_c, paxp_n, paxp_s, paxp_c)


    def plot_conductance2(self,
                          flip_panel_order=False,
                          vmin=None,
                          vmax=None,
                          cmap=None):
        """ 
        Create a summary plot of the ionospheric Hall and Pedersen conductances

        Parameters
        ----------

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_conductance2()

        """

        # get the grids:

        dr = 1
        N = 40
        M0 = 4
        
        maxlat = self.maxlat
        minlat = self.minlat
        
        grid = equal_area_grid(dr = dr, M0 = M0, N = N)
        
        mlat, mlt = grid[0], grid[1]
        
        mltc  = grid[1] + grid[2]/2. # shift to the center points of the bins
        mlatc = grid[0] + (grid[0][1] - grid[0][0])/2  # shift to the center points of the bins
        
        mltc  = mltc[ (mlatc >= minlat) & (mlatc <= maxlat)]# & (mlat <=60 )]
        mlatc = mlatc[(mlatc >= minlat) & (mlatc <= maxlat)]# & (mlat <= 60)]
        
        mlatc = np.hstack((mlatc, -mlatc)) # add southern hemisphere points
        mltc  = np.hstack((mltc ,  mltc)) # add southern hemisphere points
        
        mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
        mlt  = np.hstack((mlt ,  mlt)) # add southern hemisphere points
        
        mltres = grid[2]
        mltres = np.hstack((mltres, mltres))
        
        # Want the vector grid? Uncomment here
        # mlatv, mltv = self.vectorgrid
        # mlat,mlt = mlatv,mltv
        # mlat,mlt = mlat.ravel(),mlt.ravel()
        # SigmaH = self.get_hall_conductance()
        # SigmaP = self.get_pedersen_conductance()

        # Want the scalar grid? Uncomment here
        # mlat,mlt = self.scalargrid
        # mlat,mlt = mlat.ravel(),mlt.ravel()
        # SigmaH = self.get_hall_conductance(mlat,mlt)
        # SigmaP = self.get_pedersen_conductance(mlat,mlt)

        SigmaH, SigmaP, mask = self.get_conductances(mlat, mlt)

        SigmaH[mask] = 0.
        SigmaP[mask] = 0.

        # SigmaH[mask] = np.nan
        # SigmaP[mask] = np.nan

        SigmaHm = np.ma.masked_array(SigmaH,mask=mask)
        SigmaPm = np.ma.masked_array(SigmaP,mask=mask)

        SigmaHN,SigmaHS = np.split(SigmaHm,2)
        # SigmaHS *= -1
        SigmaPN,SigmaPS = np.split(SigmaPm,2)

        # breakpoint()

        # set up figure and polar coordinate plots:
        fig = plt.figure(figsize = (15, 15))
        if flip_panel_order:
            paxh_s = Polarplot(plt.subplot2grid((2, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            paxh_n = Polarplot(plt.subplot2grid((2, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            paxp_s = Polarplot(plt.subplot2grid((2, 15), (1,  0), colspan = 7), **self.pax_plotopts)
            paxp_n = Polarplot(plt.subplot2grid((2, 15), (1,  7), colspan = 7), **self.pax_plotopts)
        else:
            paxh_n = Polarplot(plt.subplot2grid((2, 15), (0,  0), colspan = 7), **self.pax_plotopts)
            paxh_s = Polarplot(plt.subplot2grid((2, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            paxp_n = Polarplot(plt.subplot2grid((2, 15), (1,  0), colspan = 7), **self.pax_plotopts)
            paxp_s = Polarplot(plt.subplot2grid((2, 15), (1,  7), colspan = 7), **self.pax_plotopts)
        paxh_c = plt.subplot2grid((2, 150), (0, 149), colspan = 1)
        paxp_c = plt.subplot2grid((2, 150), (1, 149), colspan = 1)
        
        # labels
        paxh_n.writeLTlabels(lat = self.minlat, size = 14)
        paxh_s.writeLTlabels(lat = self.minlat, size = 14)
        paxp_n.writeLTlabels(lat = self.minlat, size = 14)
        paxp_s.writeLTlabels(lat = self.minlat, size = 14)
        paxh_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' ,
                     ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        paxh_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$',
                     ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        # paxh_n.write(self.minlat-5, 12, r'North' , ha = 'center', va = 'center', size = 18)
        # paxh_s.write(self.minlat-5, 12, r'South' , ha = 'center', va = 'center', size = 18)
        self._make_pax_n_label(paxh_n)
        self._make_pax_s_label(paxh_s)

        def _get_siglevels(Sigma,vmin=None,vmax=None):
            # return np.linspace(np.quantile(Sigma,0.01),
            #                    np.quantile(Sigma,0.99),31)

            if vmin is None:
                sigmin = np.clip(np.quantile(Sigma,0.01),-0.5,0)
            else:
                sigmin = vmin
    
            if vmax is None:
                sigmax = np.quantile(Sigma,0.975)
            else:
                sigmax = vmax
    
            siglevels = np.linspace(sigmin,sigmax,31)
    
            return siglevels

        # siglevels = _get_siglevels(np.vstack([SigmaH,SigmaP]).flatten())
        # sighlevels, sigplevels = siglevels,siglevels

        sighlevels = _get_siglevels(SigmaH,
                                    vmin=vmin,
                                    vmax=vmax)
        # sigplevels = _get_siglevels(SigmaP,
        #                             vmin=vmin,
        #                             vmax=vmax)

        good = np.isfinite(SigmaH) & np.isfinite(SigmaP) & (np.abs(SigmaP) > 0)
        goodN = np.isfinite(SigmaHN) & np.isfinite(SigmaPN) & (np.abs(SigmaPN) > 0)
        goodS = np.isfinite(SigmaHS) & np.isfinite(SigmaPS) & (np.abs(SigmaPS) > 0)
        # sigplevels = _get_siglevels(SigmaH[good]/SigmaP[good],
        #                             vmin=np.quantile(SigmaH[good]/SigmaP[good],0.03),
        #                             vmax=np.quantile(SigmaH[good]/SigmaP[good],0.97))
        sigplevels = _get_siglevels(SigmaP[good],
                                    vmin=np.quantile(SigmaP[good],0.03),
                                    vmax=np.quantile(SigmaP[good],0.97))

        if cmap is None:
            cmapper = plt.cm.magma
        else:
            cmapper = cmap
        # cmapper = plt.cm.bwr

        if np.isclose(mlat.size/SigmaHN.size,2):

            kwargs = dict(cmap=cmapper)
            paxh_n.filled_cells(np.split(mlat,2)[0], np.split(mlt,2)[0], dr, np.split(mltres,2)[0], SigmaHN,
                                resolution = 10, crange = None, levels = sighlevels, bgcolor = 'lightgray',
                                verbose = False, **kwargs)

            paxh_s.filled_cells(np.split(mlat,2)[0], np.split(mlt,2)[0], dr, np.split(mltres,2)[0], SigmaHS,
                                resolution = 10, crange = None, levels = sighlevels, bgcolor = 'lightgray',
                                verbose = False, **kwargs)

            paxp_n.filled_cells(np.split(mlat,2)[0], np.split(mlt,2)[0], dr, np.split(mltres,2)[0], SigmaPN,
                                resolution = 10, crange = None, levels = sigplevels, bgcolor = 'lightgray',
                                verbose = False, **kwargs)

            paxp_s.filled_cells(np.split(mlat,2)[0], np.split(mlt,2)[0], dr, np.split(mltres,2)[0], SigmaPS,
                                resolution = 10, crange = None, levels = sigplevels, bgcolor = 'lightgray',
                                verbose = False, **kwargs)

            # paxh_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaHN,
            #                levels=sighlevels,cmap=cmapper,extend='both')
            # paxh_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaHS,
            #                levels=sighlevels,cmap=cmapper,extend='both')

            # paxp_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaPN,
            #                levels=sigplevels,cmap=cmapper,extend='both')
            # paxp_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],SigmaPS,
            #                levels=sigplevels,cmap=cmapper,extend='both')

            # paxp_n.contourf(np.split(mlat,2)[0][goodN],np.split(mlt,2)[0][goodN],SigmaHN[goodN]/SigmaPN[goodN],
            #                levels=sigplevels,cmap=cmapper,extend='both')
            # paxp_s.contourf(np.split(mlat,2)[0][goodS],np.split(mlt,2)[0][goodS],SigmaHN[goodS]/SigmaPS[goodS],
            #                levels=sigplevels,cmap=cmapper,extend='both')

        else:
            paxh_n.contourf(mlat, mlt, SigmaHN,levels=sighlevels,cmap=cmapper,extend='both')
            paxh_s.contourf(mlat, mlt, SigmaHS,levels=sighlevels,cmap=cmapper,extend='both')

            # paxp_n.contourf(mlat, mlt, SigmaPN,levels=sigplevels,cmap=cmapper,extend='both')
            # paxp_s.contourf(mlat, mlt, SigmaPS,levels=sigplevels,cmap=cmapper,extend='both')

            paxp_n.contourf(mlat[goodN], mlt[goodN], SigmaHN[goodN]/SigmaPN[goodN],levels=sigplevels,cmap=cmapper,extend='both')
            paxp_s.contourf(mlat[goodS], mlt[goodS], SigmaHS[goodS]/SigmaPS[goodS],levels=sigplevels,cmap=cmapper,extend='both')

        # colorbar
        paxh_c.contourf(np.vstack((np.zeros_like(sighlevels), np.ones_like(sighlevels))), 
                       np.vstack((sighlevels, sighlevels)), 
                       np.vstack((sighlevels, sighlevels)), 
                       levels = sighlevels, cmap = cmapper)

        paxp_c.contourf(np.vstack((np.zeros_like(sigplevels), np.ones_like(sigplevels))), 
                       np.vstack((sigplevels, sigplevels)), 
                       np.vstack((sigplevels, sigplevels)), 
                       levels = sigplevels, cmap = cmapper)

        # for pax_c,lab in zip([paxh_c,paxp_c],[r'$\Sigma_H$ [mho]',r'$\Sigma_H/\Sigma_P$ [mho]']):
        for pax_c,lab in zip([paxh_c,paxp_c],[r'$\Sigma_H$ [mho]',r'$\Sigma_P$ [mho]']):
            pax_c.set_xticks([])
            pax_c.set_ylabel(lab, size = 18)
            pax_c.yaxis.set_label_position("right")
            pax_c.yaxis.tick_right()

        plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (paxh_n, paxh_s, paxh_c, paxp_n, paxp_s, paxp_c)


    def plot_joule_dissip(self,
                          flip_panel_order=False,
                          vmin=None,
                          vmax=None,
                          cmap=None,
                          axN=None,
                          axS=None,
                          cax=None,
                          cax_opts=dict(),
                          suppress_labels=False):
        """ 
        Create a summary plot of the ionospheric Joule dissipation

        Parameters
        ----------

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                     -4, # IMF By in nT
                     -3, # IMF Bz in nT
                     20, # dipole tilt angle in degrees
                     150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_joule_dissip()

        """

        # Want the vector grid? Uncomment here
        # mlatv, mltv = self.plotgrid_vector
        # mlat,mlt = mlatv,mltv
        # SigmaP = self.get_pedersen_conductance()

        # Want the scalar grid? Uncomment here
        mlat,mlt = self.scalargrid
        mlat,mlt = mlat.ravel(),mlt.ravel()
        JH = self.get_joule_dissipation(mlat,mlt)

        JHN,JHS = np.split(JH,2)

        # set up figure and polar coordinate plots:
        fig = None
        if (axN is None) or (axS is None) or (cax is None):
            
            fig = plt.figure(figsize = (15, 7))
            if flip_panel_order:
                pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
                pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            else:
                pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
                pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
        else:
            pax_n = Polarplot(axN, **self.pax_plotopts)
            pax_s = Polarplot(axS, **self.pax_plotopts)
            pax_c = cax
            
        
        # labels
        pax_n.writeLTlabels(lat = self.minlat, size = 14)
        pax_s.writeLTlabels(lat = self.minlat, size = 14)
        pax_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' ,
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        pax_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$',
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        if not suppress_labels:
            self._make_pax_n_label(pax_n)
            self._make_pax_s_label(pax_s)

        # if vector_scale is None:
        #     vector_scale = 20  # mV/m
        # siglevels = np.linspace(JH.min(),JH.max(),31)
        # siglevels = np.linspace(np.quantile(JH,0.01),np.quantile(JH,0.99),31)
        # siglevels = np.linspace(np.clip(np.quantile(JH,0.01),-0.5,0),np.quantile(JH,0.975),31)

        if vmin is None:
            sigmin = np.clip(np.quantile(JH,0.01),-0.5,0)
        else:
            sigmin = vmin

        if vmax is None:
            sigmax = np.quantile(JH,0.975)
        else:
            sigmax = vmax

        siglevels = np.linspace(sigmin,sigmax,31)

        if cmap is None:
            cmapper = plt.cm.magma
        else:
            cmapper = cmap
        # cmapper = plt.cm.bwr
        # pax_n.contourf(mlatv, mltv, np.abs(JHN),levels=siglevels,cmap=cmapper,extend='both')
        # pax_s.contourf(mlatv, mltv, np.abs(JHS),levels=siglevels,cmap=cmapper,extend='both')
        if np.isclose(mlat.size/JHN.size,2):
            pax_n.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],JHN,
                           levels=siglevels,cmap=cmapper,extend='both')
            pax_s.contourf(np.split(mlat,2)[0],np.split(mlt,2)[0],JHS,
                           levels=siglevels,cmap=cmapper,extend='both')
        else:
            pax_n.contourf(mlat, mlt, JHN,levels=siglevels,cmap=cmapper,extend='both')
            pax_s.contourf(mlat, mlt, JHS,levels=siglevels,cmap=cmapper,extend='both')

        # colorbar
        pax_c.contourf(np.vstack((np.zeros_like(siglevels), np.ones_like(siglevels))), 
                       np.vstack((siglevels, siglevels)), 
                       np.vstack((siglevels, siglevels)), 
                       levels = siglevels, cmap = cmapper)
        pax_c.set_xticks([])
        # pax_c.set_ylabel(r'downward    $\mu$A/m$^2$      upward', size = 18)
        pax_c.set_ylabel(r'$W_J$ [mW/m$^2$]', size = 18)
        pax_c.yaxis.set_label_position("right")
        pax_c.yaxis.tick_right()

        # Add integrated power
        addintegpower = True
        if addintegpower:
            mlatmin,mlatmax,mltmin,mltmax = self.get_grid_binedges(gridtype='scalar')
            
            binareaopts = dict(haversine=True,
                               spherical_rectangle=True,
                               do_extra_width_calc=True,
                               altitude=110)
            
            binareas = get_h2d_bin_areas(mlatmin, mlatmax, mltmin*15, mltmax*15,
                                         **binareaopts)

            # Integrated power in GW
            integN = np.sum((JHN*1e-3)*(binareas*1e6))/1e9  # convert mW/m^2 -> W/m^2 and km^2 -> m^2
            integS = np.sum((JHS*1e-3)*(binareas*1e6))/1e9
            pax_n.write(self.minlat, 9, f"{integN:4.1f}",
                        ha = 'left', va = 'bottom', size = 16, ignore_plot_limits=True)
            pax_s.write(self.minlat, 9, f"{integS:4.1f}",
                        ha = 'left', va = 'bottom', size = 16, ignore_plot_limits=True)


        if (axN is None) or (axS is None) or (cax is None):
            plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (pax_n, pax_s, pax_c)


    def plot_pflux(self,
                   height=None,
                   showhoriz=False,
                   showabs=False,
                   onlyshowdownward=True,
                   vector_scale=None,
                   flip_panel_order=False,
                   vmin=None,
                   vmax=None,
                   cmap=None,
                   axN=None,
                   axS=None,
                   cax=None,
                   cax_opts=dict(),
                   suppress_labels=False):
        """ 
        Create a summary plot of the Poyntingflux

        Parameters
        ----------
        height           : float (default self.height, which should be 110 km)
        showhoriz        : boolean (default False)
            Show horizontal components of Poynting flux (in mW/m^2) 
        showabs          : boolean (default False)
            Show absolute value of field-aligned component of Poynting flux (in mW/m^2) 
        onlyshowdownward : boolean (default True)
            Show absolute value of field-aligned component of Poynting flux (in mW/m^2) 
        
        vector_scale : optional
            Vector lengths will be shown relative to a template. This parameter determines
            the magnitude of that template, in mW/m^2. Default is 1 mW/m^2

        Examples
        --------
        >>> # initialize by supplying a set of external conditions:
        >>> m = SWIPE(300, # solar wind velocity in km/s 
                      -4, # IMF By in nT
                      -3, # IMF Bz in nT
                      20, # dipole tilt angle in degrees
                      150) # F10.7 index in s.f.u.
        >>> # make summary plot:
        >>> m.plot_pflux()

        """

        assert 2<0,"SWIPE.plot_pflux() is not in working order, sorry!"

        # get the grids:
        mlats, mlts = self.plotgrid_scalar
        mlatv, mltv = self.plotgrid_vector

        # set up figure and polar coordinate plots:
        fig = None
        if (axN is None) or (axS is None) or (cax is None):
            
            fig = plt.figure(figsize = (15, 7))
            if flip_panel_order:
                pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
                pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            else:
                pax_n = Polarplot(plt.subplot2grid((1, 15), (0,  0), colspan = 7), **self.pax_plotopts)
                pax_s = Polarplot(plt.subplot2grid((1, 15), (0,  7), colspan = 7), **self.pax_plotopts)
            pax_c = plt.subplot2grid((1, 150), (0, 149), colspan = 1)
        else:
            pax_n = Polarplot(axN, **self.pax_plotopts)
            pax_s = Polarplot(axS, **self.pax_plotopts)
            pax_c = cax

        # labels
        pax_n.writeLTlabels(lat = self.minlat, size = 14)
        pax_s.writeLTlabels(lat = self.minlat, size = 14)
        pax_n.write(self.minlat, 3,    str(self.minlat) + r'$^\circ$' ,
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        pax_s.write(self.minlat, 3,    r'$-$' + str(self.minlat) + '$^\circ$',
                    ha = 'left', va = 'top', size = 14, ignore_plot_limits=True)
        if not suppress_labels:
            self._make_pax_n_label(pax_n)
            self._make_pax_s_label(pax_s)

        if height is not None:
            heights_scalar = np.ones(shape=np.vstack([mlats,-mlats]).shape)*height
            heights_vector = np.array([height]*len(self.vectorgrid[0]))
        else:
            heights_scalar = np.ones(shape=np.vstack([mlats,-mlats]).shape)*self.height
            heights_vector = np.array([self.height]*len(self.vectorgrid[0]))
    

        ####################
        # Get Poynting flux components
        # # OLD
        # pfluxe1,pfluxe2,pfluxpar = self.get_Poynting_flux__grid()

        # NEW, let pfluxpar be scalargrid
        pfluxe1,pfluxe2,_ = self.get_Poynting_flux__grid(heights=heights_vector)
        # _,_,pfluxpar = self.get_Poynting_flux__grid(mlats,mlts)
        _,_,pfluxpar = self.get_Poynting_flux__grid(np.vstack([mlats,-mlats]),np.vstack([mlts,mlts]),
                                                    heights=heights_scalar)

        # Horizontal Poynting flux components
        if showhoriz:
            if vector_scale is None:
                vector_scale = 1  # mW/m^2
            pfluxe1NH, pfluxe1SH = np.split(pfluxe1, 2)
            pfluxe2NH, pfluxe2SH = np.split(pfluxe2, 2)

            pax_n.plotpins(mlatv, mltv, -pfluxe2NH, pfluxe1NH, SCALE = vector_scale, markersize = 10, unit = 'mW/m^2', linewidth = .5, color = 'gray', markercolor = 'grey')
            pax_s.plotpins(mlatv, mltv, -pfluxe2SH, pfluxe1SH, SCALE = vector_scale, markersize = 10, unit = None  , linewidth = .5, color = 'gray', markercolor = 'grey')

        # Parallel Poynting flux
        pfluxparn, pfluxpars = np.split(pfluxpar, 2)

        if showabs:
            pfluxparn = np.abs(pfluxparn)
            pfluxpars = np.abs(pfluxpars)
            cmapper = plt.cm.magma
        elif onlyshowdownward:
            pfluxpars = (-1)*pfluxpars
            pfluxparn[pfluxparn < 0] = 0
            pfluxpars[pfluxpars < 0] = 0
            cmapper = plt.cm.magma
        else:
            cmapper = plt.cm.bwr             

        if cmap is not None:
            cmapper = cmap


        if vmin is None:
            fluxmin = np.min([np.quantile(pfluxparn,0.01),np.quantile(pfluxpars,0.01)])
        else:
            fluxmin = vmin

        if vmax is None:
            fluxmax = np.max([np.quantile(pfluxparn,0.975),np.quantile(pfluxpars,0.975)])
        else:
            fluxmax = vmax

        siglevels = np.linspace(fluxmin,fluxmax,31)

        # fluxlevelscoarse = np.r_[fluxmin:fluxmax:1] #We don't show coarse anymo'
        # fluxlevels = np.r_[fluxmin:fluxmax:.05]
        fluxlevels = np.linspace(fluxmin,fluxmax,31)

        pax_n.contourf(mlats, mlts, pfluxparn, levels = fluxlevels, cmap = cmapper, extend = 'both')
        pax_s.contourf(mlats, mlts, pfluxpars, levels = fluxlevels, cmap = cmapper, extend = 'both')

        # opts__contour = dict(levels = fluxlevelscoarse, linestyles='solid', colors='black', linewidths=1)
        # pax_n.contour(mlats, mlts, pfluxparn, **opts__contour)
        # pax_s.contour(mlats, mlts, pfluxpars, **opts__contour)

        # #We don't show coarse anymo'
        # opts__contour = dict(levels = fluxlevelscoarse, linestyles='solid', colors='black', linewidths=1)
        # pax_n.contour(mlats, mlts, pfluxparn, **opts__contour)
        # pax_s.contour(mlats, mlts, pfluxpars, **opts__contour)

        # dPfluxparN = pfluxparn.max()-pfluxparn.min()
        # dPfluxparS = pfluxpars.max()-pfluxpars.min()
        # pax_n.write(self.minlat-6, 2, r'$\Delta \Pfluxpar = $'+f"{dPfluxparN:.1f} kV" ,ha='center',va='center',size=18)
        # pax_s.write(self.minlat-6, 2, r'$\Delta \Pfluxpar = $'+f"{dPfluxparS:.1f} kV" ,ha='center',va='center',size=18)

        minn,maxn = np.argmin(pfluxparn),np.argmax(pfluxparn)
        mins,maxs = np.argmin(pfluxpars),np.argmax(pfluxpars)
        
        # pax_n.write(mlats.flatten()[minn],mlts.flatten()[minn],r'x',ha='center',va='center',size=18)
        # pax_n.write(mlats.flatten()[maxn],mlts.flatten()[maxn],r'+',ha='center',va='center',size=18)

        # pax_s.write(mlats.flatten()[mins],mlts.flatten()[mins],r'x',ha='center',va='center',size=18)
        # pax_s.write(mlats.flatten()[maxs],mlts.flatten()[maxs],r'+',ha='center',va='center',size=18)

        # colorbar
        pax_c.contourf(np.vstack((np.zeros_like(fluxlevels), np.ones_like(fluxlevels))), 
                       np.vstack((fluxlevels, fluxlevels)), 
                       np.vstack((fluxlevels, fluxlevels)), 
                       levels = fluxlevels, cmap = cmapper)
        pax_c.set_xticks([])
        pax_c.set_ylabel(r'Poynting flux [mW/m$^2$]', size = 18)
        pax_c.yaxis.set_label_position("right")
        pax_c.yaxis.tick_right()

        # Add integrated power
        addintegpower = True
        if addintegpower:
            mlatmin,mlatmax,mltmin,mltmax = self.get_grid_binedges(gridtype='scalar')
            
            binareaopts = dict(haversine=True,
                               spherical_rectangle=True,
                               do_extra_width_calc=True,
                               altitude=110)
            
            binareas = get_h2d_bin_areas(mlatmin, mlatmax, mltmin*15, mltmax*15,
                                         **binareaopts)

            # Integrated power in GW
            integN = np.sum((pfluxparn*1e-3)*(binareas*1e6))/1e9  # convert mW/m^2 -> W/m^2 and km^2 -> m^2
            integS = np.sum((pfluxpars*1e-3)*(binareas*1e6))/1e9
            pax_n.write(self.minlat, 9, f"{integN:4.1f}",
                        ha = 'left', va = 'bottom', size = 16, ignore_plot_limits=True)
            pax_s.write(self.minlat, 9, f"{integS:4.1f}",
                        ha = 'left', va = 'bottom', size = 16, ignore_plot_limits=True)

        if (axN is None) or (axS is None) or (cax is None):
            plt.subplots_adjust(hspace = 0, wspace = 0.4, left = .05, right = .935, bottom = .05, top = .945)

        _ = self._make_figtitle(fig)

        plt.show()

        return fig, (pax_n, pax_s, pax_c)


    def _make_pax_n_label(self,pax_n):
        pax_n.write(self.minlat-7, 12, r'North' ,
                    ha = 'center', va = 'center', size = 18, ignore_plot_limits=True)


    def _make_pax_s_label(self,pax_s):
        pax_s.write(self.minlat-7, 12, r'South' ,
                    ha = 'center', va = 'center', size = 18, ignore_plot_limits=True)

    
    def _make_figtitle(self,fig,x=0.5,y=0.95,ha='center',size=16):
        v = self.inputs['v']
        By = self.inputs['By']
        Bz = self.inputs['Bz']
        tilt = self.inputs['tilt']
        f107 = self.inputs['f107']

        # strr = f'$v$={v} km/s, $(B_y, B_z)$=({By}, {Bz}) nT, $\psi$={tilt}°, F10.7={f107} sfu'

        # title = fig.text(x,y, strr, ha=ha, size=size)

        # strr = f'$v$={v} km/s\n$(B_y, B_z)$=({By}, {Bz}) nT\n$\psi$={tilt}°\nF10.7={f107} sfu'
        # strr = f'$v$={v} km/s\n$B_y$={By} nT\n$B_z$={Bz} nT\n$\psi$={tilt}°\nF10.7={f107} sfu'
        strr = f'v     = {v:5.0f} km/s\nBy    = {By:5.2f} nT\nBz    = {Bz:5.2f} nT\ntilt  = {tilt:5.2f}°\nF10.7 = {f107:5.0f} sfu'

        x = 0.02
        y = 0.07
        size = 12
        ha = 'left'
        title = fig.text(x,y, strr, ha=ha, size=size,fontdict={'family':'monospace'})


        return title

    def _inconsistency_mask(self,J_e,J_n,
                            Emphi,Emlambda,
                            mlat,
                            verbose=False):
        """
        In this function it is assumed that 
        • J_e and J_n (in A/m) come from calling J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)
        • Emphi and Emlambda (in mW/m²) come from calling Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid), and using Emph = Ed1, Emlambda = -Ed2 * sinI

        OUTPUT
        ======

        mask : numpy.ndarray, bool
            Mask indicating where the reconstructed picture of average electrodynamics is inconsistent with the assumed neutral wind pattern (corotation with Earth by default)
        """

        mask = np.zeros_like(J_e,dtype=bool)

        if self.min_Efield__mVm is not None:
            mask[np.sqrt(Emphi*Emphi+Emlambda*Emlambda)*1000 < self.min_Efield__mVm] = 1
            if verbose:
                print(f"Inconsistency mask set to zero where E-field magnitude < {self.min_Efield__mVm:.2f} mV/m")

        if self.min_emwork is not None:
            emwork = self._emwork_func(J_e, J_n, Emphi, Emlambda)
            mask[emwork < self.min_emwork] = 1
            if verbose:
                print(f"Inconsistency mask set to zero where EM work < {self.min_emwork:.2f} mW/m²")

        if (self.min_hall is not None) or (self.max_hall is not None):
            hall = self._sigmahall_func(J_e, J_n, Emphi, Emlambda, mlat)

        if self.min_hall is not None:
            mask[hall < self.min_hall] = 1
            if verbose:
                print(f"Inconsistency mask set to zero where Hall conductance < {self.min_hall:.2f} mho")

        if self.max_hall is not None:
            mask[hall > self.max_hall] = 1
            if verbose:
                print(f"Inconsistency mask set to zero where Hall conductance > {self.max_hall:.2f} mho")

        return mask


    def _emwork_func(self, J_e, J_n, Emphi, Emlambda):
        """
        In this function it is assumed that 
        • J_e and J_n (in A/m) come from calling J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)
        • Emphi and Emlambda (in mW/m²) come from calling Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid), and using Emph = Ed1, Emlambda = -Ed2 * sinI

        The output of this function is in mW/m²
        """

        return (J_e * Emphi + J_n * Emlambda)*1000


    def _sigmahall_func(self, J_e, J_n, Emphi, Emlambda, mlat):
        """
        In this function it is assumed that 
        • J_e and J_n (in A/m) come from calling J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)
        • Emphi and Emlambda (in mW/m²) come from calling Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid), and using Emph = Ed1, Emlambda = -Ed2 * sinI

        The output of this function is in mho
        """

        SigmaH = (J_e * Emlambda - J_n * Emphi)/(Emphi**2+Emlambda**2)
        SigmaH[mlat < 0] *= -1

        return SigmaH

    def _sigmaped_func(self, J_e, J_n, Emphi, Emlambda):
        """
        In this function it is assumed that 
        • J_e and J_n (in A/m) come from calling J_e, J_n = self.get_AMPS_current(mlat = mlat, mlt = mlt, grid = grid)
        • Emphi and Emlambda (in mW/m²) come from calling Ed1, Ed2 = self.get_efield_MA(mlat,mlt,grid), and using Emph = Ed1, Emlambda = -Ed2 * sinI

        The output of this function is in mho
        """

        SigmaP = (J_e * Emphi + J_n * Emlambda)/(Emphi**2+Emlambda**2)

        return SigmaP
