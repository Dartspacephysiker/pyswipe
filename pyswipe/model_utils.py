import numpy as np
import pandas as pd
import os
from functools import reduce
from .sh_utils import get_A_matrix__Ephizero, get_A_matrix__potzero, SHkeys
from .constants import REFRE,d2r

basepath = os.path.dirname(__file__)
# basepath = '/SPENCEdata/Research/database/SHEIC/matrices/10k_points/'

default_coeff_fn = os.path.abspath(os.path.join(basepath,'coefficients','SW_OPER_EIO_SHA_2E_00000000T000000_99999999T999999_0101.txt'))

# read coefficient file and store in pandas DataFrame - with column names from last row of header:
colnames = ([x for x in open(default_coeff_fn).readlines() if x.startswith('#')][-1][1:]).strip().split(' ') 

get_coeffs = lambda fn: pd.read_table(fn, skipinitialspace = True, comment = '#', sep = ' ', names = colnames, index_col = [0, 1])


# organize the coefficients in arrays that are used to calculate magnetic field values
names = ['const', 'sinca', 'cosca',
         'epsilon', 'epsilon_sinca', 'epsilon_cosca', 
         'tilt', 'tilt_sinca', 'tilt_cosca',
         'tilt_epsilon', 'tilt_epsilon_sinca', 'tilt_epsilon_cosca',
         'tau', 'tau_sinca', 'tau_cosca',
         'tilt_tau', 'tilt_tau_sinca', 'tilt_tau_cosca',
         'f107']


def get_truncation_levels(coeff_fn = default_coeff_fn):
    """ read model truncation levels from coefficient file 
        returns NT, MT (spherical harmonic degree (N) and order (M)
        for toroidal (T) field(?))
    """

    # read relevant line and split in words:
    words = [l for l in open(coeff_fn).readlines() if 'Spherical harmonic degree' in l][0].split(' ')

    # remove commas from each word
    words = [w.replace(',', '') for w in words]

    # pick out the truncation levels and convert to ints
    NT, MT = [int(num) for num in words if num.isdigit()]

    return NT, MT


def get_m_matrix(coeff_fn = default_coeff_fn):
    """ make matrix of model coefficients - used in get_E for fast calculations
        of model field time series along trajectory with changing input
    """
    coeffs = get_coeffs(coeff_fn)

    # m_matrix = np.array([np.hstack((coeffs.loc[:, 'tor_c_' + ss].dropna().values,
    #                                 coeffs.loc[:, 'tor_s_' + ss].dropna().values,
    #                                 coeffs.loc[:, 'pol_c_' + ss].dropna().values,
    #                                 coeffs.loc[:, 'pol_s_' + ss].dropna().values)) for ss in names]).T
    m_matrix = np.array([np.hstack((coeffs.loc[:, 'tor_c_' + ss].dropna().values,
                                    coeffs.loc[:, 'tor_s_' + ss].dropna().values)) for ss in names]).T
    return m_matrix


def get_model_vectors(v, By, Bz, tilt, f107, epsilon_multiplier = 1., coeff_fn = default_coeff_fn):
    """ tor_c, tor_s = get_model_vectors(v, By, Bz, tilt, F107, epsilon_multiplier = 1., coeffs = coeffs)

        returns column vectors ((K,1)-shaped) corresponding to the spherical harmonic coefficients of the toroidal(?)
        part, with _c and _s denoting cos and sin terms, respectively.

        This function is used by sheic.SHEIC class
    """

    coeffs = get_coeffs(coeff_fn)
    coeffs = coeffs.fillna(0)

    ca = np.arctan2(By, Bz)
    epsilon = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 * epsilon_multiplier # Newell coupling           
    tau     = np.abs(v)**(4/3.) * np.sqrt(By**2 + Bz**2)**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # Newell coupling - inverse 

    # make a dict of the 19 external parameters, where the keys are postfixes in the column names of coeffs:
    external_params = {'const'             : 1                          ,                            
                       'sinca'             : 1              * np.sin(ca),
                       'cosca'             : 1              * np.cos(ca),
                       'epsilon'           : epsilon                    ,
                       'epsilon_sinca'     : epsilon        * np.sin(ca),
                       'epsilon_cosca'     : epsilon        * np.cos(ca),
                       'tilt'              : tilt                       ,
                       'tilt_sinca'        : tilt           * np.sin(ca),
                       'tilt_cosca'        : tilt           * np.cos(ca),
                       'tilt_epsilon'      : tilt * epsilon             ,
                       'tilt_epsilon_sinca': tilt * epsilon * np.sin(ca),
                       'tilt_epsilon_cosca': tilt * epsilon * np.cos(ca),
                       'tau'               : tau                        ,
                       'tau_sinca'         : tau            * np.sin(ca),
                       'tau_cosca'         : tau            * np.cos(ca),
                       'tilt_tau'          : tilt * tau                 ,
                       'tilt_tau_sinca'    : tilt * tau     * np.sin(ca),
                       'tilt_tau_cosca'    : tilt * tau     * np.cos(ca),
                       'f107'              : f107                        }

    # The SH coefficients are the sums in the expansion in terms of external parameters, scaled by the ext. params.:
    tor_c = reduce(lambda x, y: x+y, [coeffs.loc[:, 'tor_c_' + param] * external_params[param] for param in external_params.keys()]).fillna(0)
    tor_s = reduce(lambda x, y: x+y, [coeffs.loc[:, 'tor_s_' + param] * external_params[param] for param in external_params.keys()]).fillna(0)
    # pol_c = reduce(lambda x, y: x+y, [coeffs.loc[:, 'pol_c_' + param] * external_params[param] for param in external_params.keys()]).dropna()
    # pol_s = reduce(lambda x, y: x+y, [coeffs.loc[:, 'pol_s_' + param] * external_params[param] for param in external_params.keys()]).fillna(0)
    # pol_s = pol_s.loc[pol_c.index] # equal number of sin and cos terms, but sin coeffs will be 0 where m = 0
    tor_s = tor_s.loc[tor_c.index] # 

    # ORIG
    # return tor_c.values[:, np.newaxis], tor_s.values[:, np.newaxis], tor_c.index.values

    # NEW
    tor_c, tor_s, tor_keys = tor_c.values[:, np.newaxis], tor_s.values[:, np.newaxis], tor_c.index.values

    keys_T = [c for c in tor_keys]
    m_T = np.array(keys_T).T[1][np.newaxis, :]
    n_T = np.array(keys_T).T[0][np.newaxis, :]

    N, M = np.max( np.hstack((np.array([c for c in tor_keys]).T, np.array([c for c in tor_keys]).T)), axis = 1)

    # nmax = np.max([key[0] for key in tor_keys])
    n_m0 = np.sum([key[1] == 0 for key in tor_keys])
    n_m1 = np.sum([key[1] == 1 for key in tor_keys])
    # n_m2 = np.sum([key[1] == 2 for key in tor_keys])
    # mmax = np.max([key[1] for key in tor_keys])

    # Do we need to apply A matrix to get remaining coefficients?
    apply_A = False
    if n_m1 < n_m0:
        apply_A = True

        A = get_A_matrix__Ephizero(N, M,
                                   zero_thetas = 90.-np.array([47.,-47.]),
                                   return_all = False)

        checkemout_coeffs = (n_T.ravel() < (N-1)) | (m_T.ravel() == 0)


    elif (n_m1 == n_m0) and (n_T.min() == 3):
        apply_A = True

        # warnings.warn("Zero thetas flipped!")
        # A = get_A_matrix__potzero(N, M,
        #                           zero_thetas = 90.-np.array([-47.,47.]),
        #                           return_all = False)

        A = get_A_matrix__potzero(N, M,
                                  zero_thetas = 90.-np.array([47.,-47.]),
                                  return_all = False)

        checkemout_coeffs = []
        for count,(n,m) in enumerate(zip(n_T.ravel(),m_T.ravel())):
            nprime = np.maximum(1,m)
            checkemout_coeffs.append( n >= (nprime + 2) )
        checkemout_coeffs = np.array(checkemout_coeffs)


    # Apply A, if necessary
    if apply_A:
        orig_tor_c = tor_c[checkemout_coeffs]
        orig_tor_s = tor_s[checkemout_coeffs]

        tmp_tor_c = orig_tor_c.copy()
        tmp_tor_s = orig_tor_s.copy()

        tor_c = A@tmp_tor_c
        tor_s = A@tmp_tor_s

        tor_keys = SHkeys(N, M).setNmin(1).MleN().Mge(0)

    return tor_c, tor_s, tor_keys


def get_apex_base_vectors(alat, height, h_R=110.):


    """ Calculate apex coordinate base vectors d_i and e_i (i = 1, 2, 3)

    The base vectors are defined in Richmond (1995). They can be calculated analytically
    for a dipole magnetic field and spherical Earth. 

    The output vectors will have shape (3, N) where N is the combined size of the input,
    after numpy broadcasting. The three rows correspond to east, north and radial components
    for a spherical Earth

    Note
    ----
    This function only calculates Modified Apex base vectors. QD base vectors f_i and g_i are 
    just eastward, northward, and radial unit vectors for a dipole field (and f_i = g_i)
        
    Parameters
    ----------
    height : array-like
        altitude of the points where the base vectors shall be calculated, in same unit as h_R
    alat : array-like
        Modified Apex(h_R) magnetic latitude, for reference height h_R  (degrees)
    h_R : float, optional
        Reference altitude used in modified apex coordinates (assuming Dipole magnetic field!). Default is 110 km

    Returns
    -------
    d1 : array-like
        modified apex base vector d1 for a dipole magnetic field, shape (3, N)
    d2 : array-like
        modified apex base vector d2 for a dipole magnetic field, shape (3, N)
    d3 : array-like
        modified apex base vector d3 for a dipole magnetic field, shape (3, N)
    e1 : array-like
        modified apex base vector e1 for a dipole magnetic field, shape (3, N)
    e2 : array-like
        modified apex base vector e2 for a dipole magnetic field, shape (3, N)
    e3 : array-like
        modified apex base vector e3 for a dipole magnetic field, shape (3, N)
    """
    try:
        h, ala = map(np.ravel, np.broadcast_arrays(height, alat))
    except:
        raise ValueError('get_apex_base_vectors: Input not broadcastable')

    R   = REFRE + h_R
    r   = REFRE + h

    ala = np.deg2rad(ala)

    N = r.size

    def alat2qdlat_dip(alatrad,r,R):
        sign = np.ones_like(alatrad)
        sign[alatrad >= 0] = 1.
        sign[alatrad <  0] = -1.

        qlatrad = np.arccos(np.sqrt((r/R)*np.cos(alatrad)**2))

        return sign * qlatrad

    la = alat2qdlat_dip(ala, r, R)

    if np.any(r / np.cos(la)**2 < R):
        raise ValueError('get_apex_base_vectors: Some points have apex height < R. Apex height is r/cos(lat)**2.')

    R2r = R / r 
    C   = np.sqrt(4 - 3 * R2r * np.cos(la)**2)
    
    d1 =  R2r ** (3./2) * np.vstack((np.ones(N), np.zeros(N), np.zeros(N)))
    d2 = -R2r ** (3./2) / C * np.vstack((np.zeros(N), 2 * np.sin(la), np.cos(la)))
    d3 =  R2r ** (-3) * C / (4 - 3 * np.cos(la)**2) * np.vstack((np.zeros(N), np.cos(la), -2*np.sin(la)))

    e1 = np.cross(d2.T, d3.T).T
    e2 = np.cross(d3.T, d1.T).T
    e3 = np.cross(d1.T, d2.T).T

    return (d1, d2, d3, e1, e2, e3)

