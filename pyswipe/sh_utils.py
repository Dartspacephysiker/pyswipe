""" tools that are useful for spherical harmonic analysis

    SHkeys       -- class to contain n and m - the indices of the spherical harmonic terms
    nterms       -- function which calculates the number of terms in a 
                    real expansion of a poloidal (internal + external) and toroidal expansion 
    legendre -- calculate associated legendre functions - with option for Schmidt semi-normalization



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
import apexpy
from .mlt_utils import mlon_to_mlt
from builtins import range

import ppigrf

d2r = np.pi/180


DEFAULT = object()
refre = 6371.2 # reference radius

class SHkeys(object):
    """ container for n and m in spherical harmonics

        keys = SHkeys(Nmax, Mmax)

        keys will behave as a tuple of tuples, more or less
        keys['n'] will return a list of the n's
        keys['m'] will return a list of the m's
        keys[3] will return the fourth n,m tuple

        keys is also iterable

    """

    def __init__(self, Nmax, Mmax):
        keys = []
        for n in range(Nmax + 1):
            for m in range(Mmax + 1):
                keys.append((n, m))

        self.keys = tuple(keys)
        self.make_arrays()

    def __getitem__(self, index):
        if index == 'n':
            return [key[0] for key in self.keys]
        if index == 'm':
            return [key[1] for key in self.keys]

        return self.keys[index]

    def __iter__(self):
        for key in self.keys:
            yield key

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def __str__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def setNmin(self, nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= nmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def Mge(self, limit):
        """ set m >= limit  """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= limit])
        self.make_arrays()
        return self

    def NminusModd(self):
        """ remove keys if n - m is even """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """ remove keys if n - m is odd """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """ add negative m to the keys """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))
        
        self.keys = tuple(keys)
        self.make_arrays()
        
        return self


    def Shaveoff_last_k_nterms_for_m_gt(self,k,mmin):
        """For a given value of m (with m > mmin), remove the last k n terms.
        
        For example, if nmax = 65 and mmax = 3, 
        SHkeys(65, 3).setNmin(Nmin).MleN().Mge(0).Shaveoff_last_k_nterms_for_m_gt(k=2,mmin=0)
        will remove the n=64 and and n=65 keys for m > mmin = 0.
"""
        nmax = self.n.max()
        keepkey = lambda key: (key[0] <= (nmax - k) ) or (key[1] <= mmin)

        self.keys = tuple([key for key in self.keys if keepkey(key)])
        self.make_arrays()
        return self


    def Shaveoff_first_k_nterms_for_m_gt(self,k,mmin=-1):
        """ similar to Shaveoff_last_k_nterms_for_m_gt, but removes first k n-terms instead of last k n-terms """
        nmax = self.n.max()
        get_nprime = lambda key: np.maximum(key[1],1)
        keepkey = lambda key: (key[0] >= (get_nprime(key) + k) ) or (key[1] <= mmin)

        self.keys = tuple([key for key in self.keys if keepkey(key)])
        self.make_arrays()
        return self


    def make_arrays(self):
        """ prepare arrays with shape ( 1, len(keys) )
            these are used when making G matrices
        """

        if len(self) > 0:
            self.m = np.array(self)[:, 1][np.newaxis, :]
            self.n = np.array(self)[:, 0][np.newaxis, :]
        else:
            self.m = np.array([])[np.newaxis, :]
            self.n = np.array([])[np.newaxis, :]



def nterms(NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
    """ return number of coefficients in an expansion in real spherical harmonics of
        toroidal magnetic potential truncated at NT, MT
        poloidal magnetic potential truncated at NVi, MVi for internal sources
        poloidal magnetic potential truncated at NVe, MVe for external sources
    """

    return len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(1))



def legendre(nmax, mmax, theta, schmidtnormalize = True, keys = None):
    """ Calculate associated Legendre function P and its derivative

        Algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz


        Parameters
        ----------
        nmax : int
            highest spherical harmonic degree
        mmax : int
            hightest spherical harmonic order
        theta : array, float
            colatitude in degrees (shape is not preserved)
        schmidtnormalize : bool, optional
            True if Schmidth seminormalization is wanted, False otherwise. Default True
        keys : SHkeys, optional
            If this parameter is set, an array will be returned instead of a dict. 
            The array will be (N, 2M), where N is the number of elements in `theta`, and 
            M is the number of keys. The first M columns represents a matrix of P values, 
            and the last M columns represent values of dP/dtheta

        Returns
        -------
        P : dict
            dictionary of Legendre function evalulated at theta. Dictionary keys are spherical harmonic
            wave number tuples (n, m), and values will have shape (N, 1), where N is number of 
            elements in `theta`. 
        dP : dict
            dictionary of Legendre function derivatives evaluated at theta. Dictionary keys are spherical
            harmonic wave number tuples (n, m), and values will have shape (N, 1), where N is number of 
            elements in theta. 
        PdP : array (only if keys != None)
            if keys != None, PdP is returned instaed of P and dP. PdP is an (N, 2M) array, where N is 
            the number of elements in `theta`, and M is the number of keys. The first M columns represents 
            a matrix of P values, and the last M columns represent values of dP/dtheta

    """

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre functions and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]


    if keys is None:
        return P, dP
    else:
        Pmat  = np.hstack(tuple(P[key] for key in keys))
        dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    
        return np.hstack((Pmat, dPmat))


def get_legendre_arrays(nmax, mmax, theta, keys,
                        schmidtnormalize = True,
                        negative_m = False,
                        minlat = 0,
                        return_full_P_and_dP=False):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
        ***NOTE: The algorithm calculates P^m_n (μ) = P^m_n(cosθ) and dP^m_n/dθ, but we wish
                 to instead calculate dP^m_n/dλ = -dP^m_n/dθ. Hence the application of a 
                 negative sign to dP^m_n here.

        must be tested for large n - this could be unstable
        sum over m should be 1 for all thetas

        Same as get_legendre, but returns a N by 2M array, where N is the size of theta,
        and M is the number of keys. The first half the columns correspond to P[n,m], with
        n and m determined from keys - an shkeys.SHkeys object - and the second half is dP[n,m]

        theta must be a column vector (N, 1)
    """


    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    P[0, 0][np.abs(90 - theta) < minlat] = 0
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]

    if negative_m:
        for n  in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, -m]  = -1.**(-m) * factorial(n-m)/factorial(n+m) *  P[n, m]
                dP[n, -m] = -1.**(-m) * factorial(n-m)/factorial(n+m) * dP[n, m]

    if return_full_P_and_dP:
        return P, dP

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 

    return np.hstack((Pmat, dPmat))


def get_legendre_arrays__Amatrix(nmax, mmax, theta, keys, A,
                                 schmidtnormalize = True,
                                 negative_m = False,
                                 minlat = 0,
                                 zero_thetas = 90.-np.array([47.,-47.]),
                                 return_full_P_and_dP=False,
                                 multiply_dP_by_neg1=False):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
        ***NOTE: The algorithm calculates P^m_n (μ) = P^m_n(cosθ) and dP^m_n/dθ, but we wish
                 to instead calculate dP^m_n/dλ = -dP^m_n/dθ. Hence the application of a 
                 negative sign to dP^m_n here.

        must be tested for large n - this could be unstable
        sum over m should be 1 for all thetas

        Same as get_legendre, but returns a N by 2M array, where N is the size of theta,
        and M is the number of keys. The first half the columns correspond to P[n,m], with
        n and m determined from keys - an shkeys.SHkeys object - and the second half is dP[n,m]

        theta must be a column vector (N, 1)
    """

    P = {}
    dP = {}

    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    P[0, 0][np.abs(90 - theta) < minlat] = 0
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]

    if negative_m:
        for n  in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, -m]  = -1.**(-m) * factorial(n-m)/factorial(n+m) *  P[n, m]
                dP[n, -m] = -1.**(-m) * factorial(n-m)/factorial(n+m) * dP[n, m]

    # Make fmn
    # f = {}
    # df = {}
    # for m in range(0,mmax + 1):
    #     for n in range (1, nmax + 1):
    
    #         f[n, m] = Tcoeff[n, m] * P[nmax, m] - Qtilde[n, m] * P[nmax-1, m] + P[n, m]
    #         df[n, m] = Tcoeff[n, m] * dP[nmax, m] - Qtilde[n, m] * dP[nmax-1, m] + dP[n, m]

    if multiply_dP_by_neg1:
        dP = {key:(-1)*dP[key] for key in dP.keys()}

    if return_full_P_and_dP:
        return P, dP

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys))
    # return Pmat, dPmat, A
    return np.hstack((Pmat@A, dPmat@A))


def get_R_arrays(nmax, mmax, theta, keys = None,
                 zero_thetas = 90.-np.array([47.,-47.]),
                 schmidtnormalize = True,
                 return_Q = False):#,
                 # minlat = 0,
                 # return_full_P_and_dP=False):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
        ***NOTE: The algorithm calculates P^m_n (μ) = P^m_n(cosθ) and dP^m_n/dθ, NOT dP^m_n/dλ

        must be tested for large n - this could be unstable
        sum over m should be 1 for all thetas

        Same as get_legendre, but returns a N by 2M array, where N is the size of theta,
        and M is the number of keys. The first half the columns correspond to P[n,m], with
        n and m determined from keys - an shkeys.SHkeys object - and the second half is dP[n,m]

        theta must be a column vector (N, 1)
    """

    assert len(zero_thetas) == 2
    # assert np.all([key in zero_keys for key in keys]),"'zero_keys' must include all keys in 'keys'!"

    zero_keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0)

    zero_thetas = zero_thetas.reshape((2,1))
    zero_T = get_legendre_arrays(nmax, mmax, zero_thetas, zero_keys, return_full_P_and_dP=True)
    zero_T_P = {key:zero_T[0][key] for key in zero_keys}
    zero_T_dP = {key:zero_T[1][key] for key in zero_keys}

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}

    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    # P[0, 0][np.abs(90 - theta) < minlat] = 0
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]

    iplus = 0
    iminus = 1

    # Make Q terms
    Pratio_mu_plus = {key:zero_T_P[key][iplus][0]/zero_T_P[1,0][iplus][0] for key in zero_keys}

    Q = {key:P[key]-Pratio_mu_plus[key]*P[1,0] for key in zero_keys}
    dQ = {key:dP[key]-Pratio_mu_plus[key]*dP[1,0] for key in zero_keys}

    # Make R terms
    Q11_mu_minus = zero_T_P[1,1][iminus][0]-Pratio_mu_plus[1,1]*zero_T_P[1,0][iminus][0]

    R = {key:Q[key]*(1-Q[1,1]/Q11_mu_minus) for key in zero_keys}
    dR = {key:dQ[key]*(1-Q[1,1]/Q11_mu_minus)-Q[key]*dQ[1,1]/Q11_mu_minus for key in zero_keys}

    if keys is None:
        if return_Q:
            return R, dR, Q, dQ
        else:
            return R, dR
    else:

        # Pmat  = np.hstack(tuple(P[key] for key in keys))
        # dPmat = np.hstack(tuple(dP[key] for key in keys)) 
        # Qmat  = np.hstack(tuple(Q[key] for key in keys))
        # dQmat = np.hstack(tuple(dQ[key] for key in keys)) 
        Rmat  = np.hstack(tuple(R[key] for key in keys))
        dRmat = np.hstack(tuple(dR[key] for key in keys)) 
    
        if return_Q:
            Qmat  = np.hstack(tuple(Q[key] for key in keys))
            dQmat = np.hstack(tuple(dQ[key] for key in keys)) 
            
            return np.hstack((Rmat, dRmat)), np.hstack((Qmat, dQmat))

        else:
            return np.hstack((Rmat, dRmat))


def get_R_arrays__symm(nmax, mmax, theta, keys = None,
                 zero_thetas = 90.-np.array([47.,-47.]),
                 schmidtnormalize = True,
                 return_Q = False):#,
                 # minlat = 0,
                 # return_full_P_and_dP=False):
    """ get R arrays, but here we make sure that they are either symmetric or antisymmetric!
    """

    assert not return_Q,"Not implemented"

    zeros_0 = zero_thetas
    zeros_1 = 90.+(90.-zero_thetas)

    out0 = get_R_arrays(nmax, mmax, theta, keys = keys,
                        zero_thetas = zeros_0,
                        schmidtnormalize = schmidtnormalize,
                        return_Q = return_Q)

    out1 = get_R_arrays(nmax, mmax, theta, keys = keys,
                        zero_thetas = zeros_1,
                        schmidtnormalize = schmidtnormalize,
                        return_Q = return_Q)
    
    if keys is None:
        R0, dR0 = out0
        R1, dR1 = out1
        R = {key:(R0[key]+R1[key])/2 for key in R1.keys()}
        dR = {key:(dR0[key]+dR1[key])/2 for key in R1.keys()}
        
        return R, dR

    else:
        return (out0+out1)/2


def get_A_matrix__Ephizero(nmax, mmax,
                           zero_thetas = 90.-np.array([47.,-47.]),
                           return_all = False,
):
    
    assert len(zero_thetas) == 2

    zero_keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0)

    zero_thetas = zero_thetas.reshape((2,1))
    zero_T = get_legendre_arrays(nmax, mmax, zero_thetas, zero_keys, return_full_P_and_dP=True)
    zero_T_P = {key:zero_T[0][key] for key in zero_keys}
    zero_T_dP = {key:zero_T[1][key] for key in zero_keys}

    iplus = 0
    iminus = 1

    #Make Ptilde coeffs
    Ptilde = {}
    for m in range(0,mmax + 1):

        for n in range (1, nmax + 1):

            if (m == 0) or (n < m):
                Ptilde[n, m] = 0.
            else:
                Ptilde[n, m] = zero_T_P[n, m][iplus] / zero_T_P[nmax, m][iplus]


    #Make zero-Q
    zero_T_Q = {}
    for m in range(0,mmax + 1):
        for n in range (1, nmax + 1):
            if (m == 0) or (n < m):
                zero_T_Q[n, m] = 0.
            else:
                zero_T_Q[n, m] = zero_T_P[n, m][iminus] - Ptilde[n, m] * zero_T_P[nmax, m][iminus]
    

    #Make Qtilde coeffs
    Qtilde = {}
    for m in range(0,mmax + 1):
        for n in range (1, nmax + 1):

            if (m == 0) or (n < m):
                Qtilde[n, m] = 0.
            else:
                Qtilde[n, m] = zero_T_Q[n, m] / zero_T_Q[nmax-1, m]
            
    
    #Make T coeffs
    Tcoeff = {}
    for m in range(0,mmax + 1):
        for n in range (1, nmax + 1):

            if (m == 0) or (n < m):
                Tcoeff[n, m] = 0.
            else:
                Tcoeff[n, m] = Ptilde[nmax-1, m] * Qtilde[n, m] - Ptilde[n, m]

    # Need-'ems  for A matrix
    keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0)
    Ncoeffs = len(keys)
    narr = keys.n[0]
    marr = keys.m[0]
    
    # Now make A matrix
    A = np.zeros((Ncoeffs,Ncoeffs-2*mmax))
    zerorow = np.zeros(Ncoeffs-2*mmax)
    count = 0
    fixcount = 0
    for n in range (1, nmax + 1):
        for m in range(0,np.minimum(mmax + 1,n+1)):
    
            if (n >= (nmax-1)) and (m > 0):
                fixcount += 1
    
                tmprow = zerorow.copy()
    
                # get indices of (n', m) coefficients, n' ≤ nmax - 2, that we 
                # write the (nmax, m) or (nmax-1, m) coefficient in terms of
                fix_columns = np.where((narr < (nmax-1)) & (marr == m))[0]  
    
    
                # get values of n and m
                tmpn = narr[fix_columns]
                tmpm = marr[fix_columns]
    
                if n == nmax-1:
                    # do the thing for g^m_N-1
                    # tmprow[fix_columns] = -np.array([coeffs['Qtilde'][n,m] for n,m in zip(tmpn,tmpm)]).flatten()
                    tmprow[fix_columns] = -np.array([Qtilde[n,m] for n,m in zip(tmpn,tmpm)]).flatten()
    
                elif n == nmax:
                    # do the thing for g^m_N
                    # tmprow[fix_columns] = np.array([coeffs['Tcoeff'][n,m] for n,m in zip(tmpn,tmpm)]).flatten()
                    tmprow[fix_columns] = np.array([Tcoeff[n,m] for n,m in zip(tmpn,tmpm)]).flatten()
    
            else:
                tmprow = zerorow.copy()
                tmprow[count-fixcount] = 1
    
            A[count,:] = tmprow
    
            count +=1   
    
    if return_all:
        return A, dict(Qtilde=Qtilde,
                       Ptilde=Ptilde,
                       zero_T_Q=zero_T_Q,
                       Tcoeff=Tcoeff)
    else:
        return A


def get_A_matrix__potzero(nmax, mmax,
                          zero_thetas = 90.-np.array([47.,-47.]),
                          return_all = False,
):
    
    assert len(zero_thetas) == 2

    zero_keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0)

    zero_thetas = zero_thetas.reshape((2,1))
    zero_T = get_legendre_arrays(nmax, mmax, zero_thetas, zero_keys, return_full_P_and_dP=True)
    zero_T_P = {key:zero_T[0][key] for key in zero_keys}
    zero_T_dP = {key:zero_T[1][key] for key in zero_keys}

    iplus = 0
    iminus = 1

    #Make Ptilde coeffs
    Ptilde = {}
    for m in range(0,mmax + 1):

        nprime = np.maximum(m,1)

        for n in range(1, nmax + 1):

            if n < (nprime+1):
                Ptilde[n, m] = 0.
            else:
                Ptilde[n, m] = zero_T_P[n, m][iplus] / zero_T_P[nprime, m][iplus]


    #Make zero-Q
    zero_T_Q = {}
    for m in range(0,mmax + 1):

        nprime = np.maximum(m,1)

        for n in range (1, nmax + 1):
            if n < (nprime+1):
                zero_T_Q[n, m] = 0.
            else:
                zero_T_Q[n, m] = zero_T_P[n, m][iminus] - Ptilde[n, m] * zero_T_P[nprime, m][iminus]
    

    #Make Qtilde coeffs
    Qtilde = {}
    for m in range(0,mmax + 1):

        nprime = np.maximum(m,1)

        for n in range (1, nmax + 1):
            if n < (nprime + 2):
                Qtilde[n, m] = 0.
            else:
                Qtilde[n, m] = zero_T_Q[n, m] / zero_T_Q[nprime+1, m]
            
    #Make T coeffs
    Tcoeff = {}
    for m in range(0,mmax + 1):

        nprime = np.maximum(m,1)

        for n in range (1, nmax + 1):

            if n < (nprime + 2):
                Tcoeff[n, m] = 0.
            else:
                Tcoeff[n, m] = Ptilde[nprime+1, m] * Qtilde[n, m] - Ptilde[n, m]

    # Need-'ems  for A matrix
    keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0).Shaveoff_first_k_nterms_for_m_gt(2)
    Ntotcoeffs = len(zero_keys)
    Ncoeffs = len(keys)
    nallarr = zero_keys.n[0]
    mallarr = zero_keys.m[0]
    narr = keys.n[0]
    marr = keys.m[0]
    
    # Now make A matrix
    # A = np.zeros((Ntotcoeffs,Ntotcoeffs-2*(mmax+1)))
    # zerorow = np.zeros(Ntotcoeffs-2*(mmax+1))
    A = np.zeros((Ntotcoeffs,Ncoeffs))
    zerorow = np.zeros(Ncoeffs)
    count = 0
    fixcount = 0
    for n in range (1, nmax + 1):
        for m in range(0,np.minimum(mmax,n)+1):
    
            nprime = np.maximum(1,m)

            # if (n >= (nmax-1)) and (m > 0):
            if (n <= (nprime+1)):

                fixcount += 1
    
                tmprow = zerorow.copy()
    
                # get indices of (n, m) coefficients, n ≥ n' + 2, that we 
                # write the (n', m) or (n'+1, m) coefficient in terms of
                # where n' = max(m,1)
                # fix_columns = np.where((nallarr > (nprime+1)) & (mallarr == m))[0]  
                fix_columns = np.where((narr > (nprime+1)) & (marr == m))[0]  
    
                # get values of n and m
                tmpn = narr[fix_columns]
                tmpm = marr[fix_columns]
    
                if n == nprime:
                    # do the thing for g^m_n'
                    # tmprow[fix_columns] = -np.array([coeffs['Qtilde'][n,m] for n,m in zip(tmpn,tmpm)]).flatten()
                    # tmprow[fix_columns] = -np.array([Qtilde[n,m] for n,m in zip(tmpn,tmpm)]).flatten()
                    tmprow[fix_columns] = np.array([Tcoeff[n,m] for n,m in zip(tmpn,tmpm)]).flatten()
    
                elif n == nprime+1:
                    # do the thing for g^m_n'+1
                    # tmprow[fix_columns] = np.array([coeffs['Tcoeff'][n,m] for n,m in zip(tmpn,tmpm)]).flatten()
                    tmprow[fix_columns] = -np.array([Qtilde[n,m] for n,m in zip(tmpn,tmpm)]).flatten()
    
            else:
                tmprow = zerorow.copy()
                tmprow[count-fixcount] = 1
    
            A[count,:] = tmprow
    
            count +=1   
    
    if return_all:
        return A, dict(Qtilde=Qtilde,
                       Ptilde=Ptilde,
                       zero_T_Q=zero_T_Q,
                       Tcoeff=Tcoeff)
    else:
        return A


def getG_vel(glat, glon, height, time, epoch = 2015., h_R = 110.,
             NT = 65, MT = 3, 
             RR=6371.2,
             makenoise=False,
             toroidal_minlat=0,
             zero_lats=np.array([47.,-47.]),
             coords='geo'):
    """ all input arrays should be dask arrays with shape (N, 1), and with the same chunksize. """

    glat   = np.asarray(glat).flatten()
    glon   = np.asarray(glon).flatten()
    height = np.asarray(height).flatten()

    # convert to magnetic coords and get base vectors
    a = apexpy.Apex(epoch, refh = h_R)
    qlat, qlon = a.geo2qd(  glat.flatten(), glon.flatten(), height.flatten())
    alat, alon = a.geo2apex(glat.flatten(), glon.flatten(), height.flatten())
    f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(alat, alon, height, coords  = 'apex')
    d1e = d1[0].reshape(-1, 1)
    d1n = d1[1].reshape(-1, 1)
    d1u = d1[2].reshape(-1, 1)
    d2e = d2[0].reshape(-1, 1)
    d2n = d2[1].reshape(-1, 1)
    d2u = d2[2].reshape(-1, 1)
    e1e = e1[0].reshape(-1, 1)
    e1n = e1[1].reshape(-1, 1)
    e1u = e1[2].reshape(-1, 1)
    e2e = e2[0].reshape(-1, 1)
    e2n = e2[1].reshape(-1, 1)
    e2u = e2[2].reshape(-1, 1)

    from datetime import datetime
    Be, Bn, Bu = ppigrf.igrf(
        glon,
        glat,
        height,
        datetime(int(epoch),
                 np.clip(int((epoch-int(epoch))*12),1,12),
                 1)
    )
    B0IGRF = np.sqrt(Be**2+Bn**2+Bu**2)

    D = np.sqrt( (d1n*d2u-d1u*d2n)**2 + \
                 (d1u*d2e-d1e*d2u)**2 + \
                 (d1e*d2n-d1n*d2e)**2)

    Be3_in_Tesla = (B0IGRF/D.flatten()/1e9).T

    # calculate magnetic local time
    phi = mlon_to_mlt(qlon, time, a.year)[:, np.newaxis]*15 # multiply by 15 to get degrees

    # turn the coordinate arrays into column vectors:
    alat, h = map(lambda x: x.flatten()[:, np.newaxis], [alat, height])

    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0).Shaveoff_first_k_nterms_for_m_gt(2)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1).Shaveoff_first_k_nterms_for_m_gt(2)

    fullkeys = {} # dictionary of spherical harmonic keys
    fullkeys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    fullkeys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)

    m_cos_T = keys['cos_T'].m
    m_sin_T = keys['sin_T'].m

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    if makenoise: print( 'Calculating Legendre functions. alat shape and chunks:', alat.shape, alat.chunks)
    assert len(zero_lats) == 2
    A = get_A_matrix__potzero(NT, MT,
                              zero_thetas = 90.-zero_lats)

    P_T = get_legendre_arrays__Amatrix(NT, MT, 90 - alat, fullkeys['cos_T'], A,
                                       minlat = toroidal_minlat,
                                       zero_thetas = 90.-zero_lats,
                                       multiply_dP_by_neg1=True)


    P_cos_T  =  P_T[:, :len(keys['cos_T']) ] # split
    dP_cos_T = P_T[:,  len(keys['cos_T']):]

    if makenoise: print( 'P, dP cos_T size and chunks', P_cos_T.shape, dP_cos_T.shape)#, P_cos_T.chunks, dP_cos_T.chunks
    P_sin_T  =  P_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
    dP_sin_T =  dP_cos_T[:, keys['cos_T'].m.flatten() != 0]

    if makenoise: print( 'P, dP sin_T size and chunks', P_sin_T.shape, dP_sin_T.shape, P_sin_T.chunks[0], dP_sin_T.chunks[1])

    # trig matrices:
    cos_T  =  np.cos(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    sin_T  =  np.sin(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))

    dcos_T = -np.sin(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    dsin_T =  np.cos(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))

    if makenoise: print( cos_T.shape, sin_T.shape)

    cos_alat   = np.cos(alat * d2r)

    sinI  = 2 * np.sin( alat * d2r )/np.sqrt(4 - 3*cos_alat**2)

    R = (RR + h_R)                   # DON'T convert from km to m; this way potential is in kV

    # matrices with partial derivatives in MA coordinates:
    dT_dalon  = np.hstack(( P_cos_T * dcos_T * m_cos_T,  P_sin_T * dsin_T * m_sin_T))
    dT_dalat  = np.hstack((dP_cos_T *  cos_T          , dP_sin_T *  sin_T          ))

    # Stuff for ve1 and ve2, cf Equation 19 in Hatch et al (2023)
    # Remember that because Ed1 = -1/[ (R_E+h_R)*cos_alat ] dPhi/dphi [cf Eq 16], the negative sign in Eq 19 disappears
    # Hence no negative signs in expression for G matrix below
    V_E1 = RR/(R * Be3_in_Tesla * 1000) * (1 / sinI     * dT_dalat)
    V_E2 = RR/(R * Be3_in_Tesla * 1000) * (1 / cos_alat * dT_dalon)

    if coords == 'apex':
        G = np.vstack((V_E1, V_E2))

        return G

    elif coords == 'geo':

        v_e = V_E1 * e1e + V_E2 * e2e
        v_n = V_E1 * e1n + V_E2 * e2n
        v_u = V_E1 * e1u + V_E2 * e2u
    
        # G matrix for geographic components of E-field, cf Equation 15 in Hatch et al (2023)
        G     = np.vstack((v_e, v_n, v_u))
    
        return G


def getG_E(glat, glon, height, time, epoch = 2015., h_R = 110.,
           NT = 65, MT = 3, 
           RR=6371.2,
           makenoise=False,
           toroidal_minlat=0,
           zero_lats=np.array([47.,-47.]),
           coords='geo'):
    """ all input arrays should be dask arrays with shape (N, 1), and with the same chunksize. """

    assert coords in ['geo', 'apex']

    glat   = np.asarray(glat).flatten()
    glon   = np.asarray(glon).flatten()
    height = np.asarray(height).flatten()

    # convert to magnetic coords and get base vectors
    a = apexpy.Apex(epoch, refh = h_R)
    qlat, qlon = a.geo2qd(  glat.flatten(), glon.flatten(), height.flatten())
    alat, alon = a.geo2apex(glat.flatten(), glon.flatten(), height.flatten())
    f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(alat, alon, height, coords  = 'apex')
    d1e = d1[0].reshape(-1, 1)
    d1n = d1[1].reshape(-1, 1)
    d1u = d1[2].reshape(-1, 1)
    d2e = d2[0].reshape(-1, 1)
    d2n = d2[1].reshape(-1, 1)
    d2u = d2[2].reshape(-1, 1)

    # calculate magnetic local time
    phi = mlon_to_mlt(qlon, time, a.year)[:, np.newaxis]*15 # multiply by 15 to get degrees

    # turn the coordinate arrays into column vectors:
    alat, h = map(lambda x: x.flatten()[:, np.newaxis], [alat, height])

    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0).Shaveoff_first_k_nterms_for_m_gt(2)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1).Shaveoff_first_k_nterms_for_m_gt(2)

    fullkeys = {} # dictionary of spherical harmonic keys
    fullkeys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    fullkeys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)

    m_cos_T = keys['cos_T'].m
    m_sin_T = keys['sin_T'].m

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    if makenoise: print( 'Calculating Legendre functions. alat shape and chunks:', alat.shape, alat.chunks)
    assert len(zero_lats) == 2
    A = get_A_matrix__potzero(NT, MT,
                              zero_thetas = 90.-zero_lats)

    P_T = get_legendre_arrays__Amatrix(NT, MT, 90 - alat, fullkeys['cos_T'], A,
                                       minlat = toroidal_minlat,
                                       zero_thetas = 90.-zero_lats,
                                       multiply_dP_by_neg1=True)


    P_cos_T  =  P_T[:, :len(keys['cos_T']) ] # split
    dP_cos_T = P_T[:,  len(keys['cos_T']):]

    if makenoise: print( 'P, dP cos_T size and chunks', P_cos_T.shape, dP_cos_T.shape)#, P_cos_T.chunks, dP_cos_T.chunks
    P_sin_T  =  P_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
    dP_sin_T =  dP_cos_T[:, keys['cos_T'].m.flatten() != 0]

    if makenoise: print( 'P, dP sin_T size and chunks', P_sin_T.shape, dP_sin_T.shape, P_sin_T.chunks[0], dP_sin_T.chunks[1])

    # trig matrices:
    cos_T  =  np.cos(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    sin_T  =  np.sin(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))

    dcos_T = -np.sin(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    dsin_T =  np.cos(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))

    if makenoise: print( cos_T.shape, sin_T.shape)

    cos_alat   = np.cos(alat * d2r)

    sinI  = 2 * np.sin( alat * d2r )/np.sqrt(4 - 3*cos_alat**2)

    R = (RR + h_R)                   # DON'T convert from km to m; this way potential is in kV

    # matrices with partial derivatives in MA coordinates:
    dT_dalon  = np.hstack(( P_cos_T * dcos_T * m_cos_T,  P_sin_T * dsin_T * m_sin_T))
    dT_dalat  = np.hstack((dP_cos_T *  cos_T          , dP_sin_T *  sin_T          ))

    E_D1 = - RR/R * (1 / cos_alat * dT_dalon)
    E_D2 =   RR/R * (1 / sinI     * dT_dalat)

    if coords == 'apex':

        G = np.vstack((E_D1, E_D2))

        return G

    elif coords == 'geo':

        E_e = E_D1 * d1e + E_D2 * d2e
        E_n = E_D1 * d1n + E_D2 * d2n
        E_u = E_D1 * d1u + E_D2 * d2u
    
        # G matrix for geographic components of E-field, cf Equation 15 in Hatch et al (2023)
        G     = np.vstack((E_e, E_n, E_u))
    
        return G
    
