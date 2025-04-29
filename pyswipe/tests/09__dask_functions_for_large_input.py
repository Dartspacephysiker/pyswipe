"""
Some examples showing how to use the dask-based Swipe functions that are good at computing time series of model outputs along, say, satellite tracks.
"""

from pyswipe import SWIPE, get_E, get_v, get_pflux, get_emwork, get_conductances
from datetime import datetime

import numpy as np
from apexpy import Apex


vsw = 300 # solar wind velocity in km/s 
By, Bz = -4, -3 # IMF By and Bz in nT
tilt, f107 = 20, 150 # dipole tilt angle in degrees, F10.7 index in s.f.u

h_R = 110.
refdt = datetime(2015,6,1,0,0)

m = SWIPE(vsw, 
          By, 
          Bz, 
          tilt, 
          f107,
          height=h_R)

alat, mlt = m.vectorgrid
alat, mlt = map(lambda x: x.flatten(), [alat, mlt])

a = Apex(date=refdt,refh=h_R)

alon = a.mlt2mlon(mlt, refdt)
height = np.ones_like(alon)*110.

glat, glon, error = a.apex2geo(alat, alon, height)
time = [refdt]*height.size

# Convert inputs to arrays, which is what dask functions get_E, get_v, and get_pflux are expecting
vsw, By, Bz, tilt, f107 = map(lambda x: np.ones_like(glat)*x, [vsw, By, Bz, tilt, f107])

epoch = a.year

chunksize = 15000               # chunksize to be used by dask

f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(alat, alon, height, coords  = 'apex')

## Compare E-field
E_e, E_n, E_u = get_E(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                      coords='geo',
                      epoch = epoch, h_R = h_R, chunksize = chunksize)

E_d1alt, E_d2alt = get_E(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                   coords='apex',
                   epoch = epoch, h_R = h_R, chunksize = chunksize)


E_d1, E_d2 = m.get_efield_MA(alat,mlt)

E_ealt = E_d1*d1[0] + E_d2*d2[0]
E_nalt = E_d1*d1[1] + E_d2*d2[1]
E_ualt = E_d1*d1[2] + E_d2*d2[2]

print("")
print("Comparing E-field from get_E and SWIPE.get_efield_MA in geo coords")
print("E_e median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((E_ealt-E_e)/E_e*100,3)))))
print("E_n median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((E_nalt-E_n)/E_n*100,3)))))
print("E_u median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((E_ualt-E_u)/E_u*100,3)))))
print("")
print("Comparing E-field from get_E and SWIPE.get_efield_MA in apex coords")
print("E_d1 median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((E_d1alt-E_d1)/E_d1*100,3)))))
print("E_d2 median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((E_d2alt-E_d2)/E_d2*100,3)))))
print("")

## Compare v
v_e, v_n, v_u = get_v(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                           epoch = epoch, h_R = h_R,
                           coords = 'geo',
                           chunksize = chunksize)

v_e1alt, v_e2alt = get_v(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                           epoch = epoch, h_R = h_R,
                           coords = 'apex',
                           chunksize = chunksize)


v_e1, v_e2 = m.get_convection_vel_MA(alat,mlt)

v_ealt = v_e1*e1[0] + v_e2*e2[0]
v_nalt = v_e1*e1[1] + v_e2*e2[1]
v_ualt = v_e1*e1[2] + v_e2*e2[2]


print("")
print("Comparing convection from get_v and SWIPE.get_convection_vel_MA in geo coords")
print("v_e median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_ealt-v_e)/v_e*100,3)))))
print("v_n median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_nalt-v_n)/v_n*100,3)))))
print("v_u median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_ualt-v_u)/v_u*100,3)))))
print("")
print("Comparing convection from get_v and SWIPE.get_convection_vel_MA in apex coords")
print("v_e1 median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_e1alt-v_e1)/v_e1*100,3)))))
print("v_e2 median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_e2alt-v_e2)/v_e2*100,3)))))
print("")


## Compare pflux
pfluxe1, pfluxe2, pfluxpar = m.get_poynting_flux(mlat = alat, mlt = mlt,
                                               times = time,
                                               heights = height,
                                               apex_refdate=refdt,
                                               apex_refheight=h_R,
                                               grid = False,
                                               killpoloidalB=True)

pflux = get_pflux(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                  epoch = epoch, h_R = h_R,
                  coords = 'apex',
                  chunksize = chunksize)

print("")
print("Comparing field-aligned poynting flux from get_pflux and SWIPE.get_poynting_flux in apex coords")
# print("v_e median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_ealt-v_e)/v_e*100,3)))))
# print("v_n median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((v_nalt-v_n)/v_n*100,3)))))
print("pflux_parallel median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((pflux[2]-pfluxpar)/pfluxpar*100,3)))))
print("")


## Compare em work
emwork = m.get_emwork(mlat = alat, mlt = mlt)

emwork2 = get_emwork(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                     epoch = epoch, h_R = h_R,
                     chunksize = chunksize)



print("")
print("Comparing EM work from get_emwork and SWIPE.get_emwork")
print("emwork median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((emwork-emwork2)/emwork*100,3)))))
print("emwork median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((emwork-emwork2)/emwork2*100,3)))))
print("")


## Compare conductances
sigmah, sigmap, mask = m.get_conductances(mlat = alat, mlt = mlt)

sigmah2, sigmap2, bonus = get_conductances(glat, glon, height, time, vsw, By, Bz, tilt, f107,
                                    epoch = epoch, h_R = h_R,
                                    chunksize = chunksize)
# bonus is (mlat, mlon, sinI, Emphi, Emlambda)


d2r = np.pi/180
sinI = 2 * np.sin(alat * d2r)/np.sqrt(4-3*np.cos(alat * d2r)**2)

# The problem is not sinI
# In [9]: np.abs((sinI-bonus[2])/sinI*100).max()
# Out[9]: 2.3655031869288063e-06

# The problem is not calculation of mlat
# In [11]: np.max(np.abs((alat-bonus[0])))
# Out[11]: 7.62939453125e-06

# The problem is not calculation of mlon
# In [23]: np.max(np.abs(alon -(bonus[1] % 360)))
# Out[23]: 0.0001678466796875



print("")
print("Comparing conductances from get_conductances and SWIPE.get_conductances")
print("sigmah median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((sigmah[mask]-sigmah2[mask])/sigmah[mask]*100,3)))))
print("sigmap median abs % diff: {:.2f}%".format(np.median(np.abs(np.round((sigmap[mask]-sigmap2[mask])/sigmap[mask]*100,3)))))
print("")

