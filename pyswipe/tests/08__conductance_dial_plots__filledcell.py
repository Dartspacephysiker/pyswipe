import numpy as np

from pyswipe.swipe import SWIPE

plotdir = '/plotdir/'

import matplotlib as mpl
# MPL opts
mpl.rcParams.update({'text.color': 'k'})
mpl.rcParams.update({'axes.labelcolor': 'k'})
mpl.rcParams.update({'xtick.color': 'k'})
mpl.rcParams.update({'ytick.color': 'k'})
mpl.rcParams.update({'text.usetex': False})

mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
mpl.rcParams.update({'savefig.directory': plotdir})

import matplotlib.pyplot as plt
plt.ion()

from polplot.polplot import Polarplot

d2r = np.pi/180

dontbescaredofnegs = True

do_jdote_screen = True
min_jdote = 0.5

do_hall_screen = True
min_hall = 0.05
max_hall = 100.

##############################

conductance = 'h'

assert conductance in ['p','h','c']

if conductance == 'p':
    titstr = 'Pedersen'
elif conductance == 'h':
    titstr = 'Hall'
elif conductance == 'c':
    titstr = 'Cowling'

B = 5                          # nT

ca = {(0, 1): 0, (0, 0):-45, (1, 0):-90,(2,0):-135,(2,1):180,(0,2):45,(1,2):90,(2,2):135}

tilts = [-25, 0, 25]

v = 450
f107val = 120

minlat = 45

potlevels = np.arange(0, 8.1, 2)
cmap = plt.cm.Blues
extend = 'both'
if dontbescaredofnegs:
    # potlevels = np.arange(-8, 8, 2)+1
    # cmap = plt.cm.bwr
    potlevels = np.arange(-18, 18, 2)+1
    cmap = plt.cm.bwr
    extend = None

doCalcBhattacharyya = False      # Code for this taken from "pct_diff_ns_currents_for_varying_vswB_and_clock_angle.py"


zero_latNU = None

##############################
# Initialize models
swipe_n = SWIPE(v, B, B, 0, f107val,
             minlat=minlat)#,
print(1/0)
swipe_s = SWIPE(v, B, B, 0, f107val,
             minlat=minlat)#,

##############################
# Get grid for calculating conductances
from pyswipe.plot_utils import equal_area_grid, Polarplot, get_h2d_bin_areas

# defaults
# dr = 2
# N = 20
# M0 = 4

# dad
dr = 1
N = 40
M0 = 4

maxlat = swipe_n.maxlat

grid = equal_area_grid(dr = dr, M0 = M0, N = N)

mlat, mlt = grid[0], grid[1]

mltc  = grid[1] + grid[2]/2. # shift to the center points of the bins
mlatc = grid[0] + dr/2  # shift to the center points of the bins

mltc  = mltc[ (mlatc >= minlat) & (mlatc <= maxlat)]# & (mlat <=60 )]
mlatc = mlatc[(mlatc >= minlat) & (mlatc <= maxlat)]# & (mlat <= 60)]

mlatc = np.hstack((mlatc, -mlatc)) # add southern hemisphere points
mltc  = np.hstack((mltc ,  mltc)) # add southern hemisphere points

mlat = np.hstack((mlat, -mlat)) # add southern hemisphere points
mlt  = np.hstack((mlt ,  mlt)) # add southern hemisphere points

mltres = grid[2]
mltres = np.hstack((mltres, mltres))

mlat_n, mlt_n = map(lambda x:np.split(x, 2)[0], (mlat, mlt))
mlat_s, mlt_s = map(lambda x:np.split(x, 2)[1], (mlat, mlt))

zero_latNU = None

# if doCalcBhattacharyya:

# Ripped from pyamps.amps.get_integrated_upward_current
REFRE = 6371.2 # Reference radius used in geomagnetic modeling

# get surface area element in each cell:
mlat, mlt = swipe_n.scalargrid
mlt_sorted = np.sort(np.unique(mlt))
mltresr = (mlt_sorted[1] - mlt_sorted[0]) * np.pi / 12
mlat_sorted = np.sort(np.unique(mlat))
mlatres = (mlat_sorted[1] - mlat_sorted[0]) * np.pi / 180
R = (REFRE + swipe_n.height) * 1e3  # radius in meters
dS = R**2 * np.cos(mlat * np.pi/180) * mlatres * mltresr

dS_n,dS_s = np.split(dS, 2)
assert np.array_equal(dS_n,dS_s),"Uh oh, calculations later assume dS_n == dS_s!"
dS = dS_n.ravel()

def sep_into_pos_and_neg_dist_and_norm(vals,dS):
    valspos, valsneg = np.copy(vals),-vals
    valspos[valspos < 0] = 0
    valsneg[valsneg < 0] = 0
    
    vals_sum = np.sum(valspos*dS+valsneg*dS)
    
    valspos = valspos/vals_sum
    valsneg = valsneg/vals_sum
    
    return valspos,valsneg


##############################
# Loop
# lists for holding CPCP for each bz value
cpcpn, cpcps = [], []
locns, locss = [], []
fig = plt.figure(1,figsize = (9, 9))

for tilt in tilts:#tilt = tilts[0]

    plt.clf()

    axes = [[Polarplot(plt.subplot2grid((91, 3), (j*30, i), rowspan = 30),
                          linestyle = ':', color = 'grey', minlat = minlat, linewidth = .9)
             for i in range(3) if (i, j) != (1, 1)]
             for j in range(3)]
    axdial = plt.subplot2grid((91, 3), (30, 1), rowspan = 30)
    axdial.set_aspect('equal')
    axdial.set_axis_off()
    # if withFAC:
    axcbar = plt.subplot2grid((91, 3), (90, 1))
    axinfo = plt.subplot2grid((91, 3), (90, 0))
    axinfo.set_axis_off()

    tmpcpcpn = []
    tmpcpcps = []
    tmplocn, tmplocs = [],[]
    for i in range(3):
        for j in range(3):
            if i == j == 1:
                continue # skip
            _ca = ca[(i,j)]
            By, Bz = np.sin(_ca * d2r) * B, np.cos(_ca * d2r) * B
            if _ca in [0, -90, 180, 90]:
                axdial.text(np.sin(_ca * d2r), np.cos(_ca * d2r),
                            str(_ca) + r'$^\circ$',
                            # rotation = -_ca,
                            ha = 'center', va = 'center')
                axdial.plot([.75*np.sin(_ca * d2r), .85*np.sin(_ca * d2r)],
                            [.75*np.cos(_ca * d2r), .85*np.cos(_ca * d2r)],
                            color = 'lightgrey')

            if (i, j) == (1, 2):
                j = 1
            swipe_n.update_model(v,  By, Bz,  tilt, f107val)
            swipe_s.update_model(v, -By, Bz, -tilt, f107val)


            SigmaHN, SigmaPN, maskN = self.get_conductances(mlat_n, mlt_n)
            SigmaHS, SigmaPS, maskS = self.get_conductances(mlat_s, mlt_s)

            if conductance == 'p':
                phin = SigmaPN
                phis = SigmaPS
            elif conductance == 'h':
                phin = SigmaHN
                phis = SigmaHS
            elif conductance == 'c':
                phin = SigmaPN
                phis = SigmaPS

            phin = phin.ravel()
            phis = phis.ravel()

            phinm = np.ma.masked_array(phin,mask=maskN)
            phism = np.ma.masked_array(phis,mask=maskS)

            filledn = axes[i][j].filled_cells(mlat_n, mlt_n, dr, np.split(mltres,2)[0], SigmaHN,
                                resolution = 10, crange = None, levels = sighlevels, bgcolor = 'lightgray',
                                verbose = False, **kwargs)

            contn = axes[i][j].contourf(mlat_n, mlt_n, phin, levels = potlevels, cmap = cmap, extend = extend)
            conts = axes[i][j].contour (mlat_s, mlt_s, phis, levels = potlevels, colors = 'k' , extend = extend, linewidths = .4)

            if doCalcBhattacharyya:
                # Separate into pos and neg current distributions
                psipos_n, psineg_n = np.copy(phin),-phin
                psipos_n[psipos_n < 0] = 0
                psineg_n[psineg_n < 0] = 0
            
                psipos_s, psineg_s = np.copy(phis),-phis
                psipos_s[psipos_s < 0] = 0
                psineg_s[psineg_s < 0] = 0
                
                phin_sum = np.sum(psipos_n*dS+psineg_n*dS)
                phis_sum = np.sum(psipos_s*dS+psineg_s*dS)
            
                # Normalize pos and neg current distributions
                psipos_n, psineg_n = sep_into_pos_and_neg_dist_and_norm(phin,dS)
                psipos_s, psineg_s = sep_into_pos_and_neg_dist_and_norm(phis,dS)
            
                # Calculate integrands for Bhattacharya coefficient
                facposintegrand = np.sqrt(psipos_n*psipos_s)
                facnegintegrand = np.sqrt(psineg_n*psineg_s)
                
                facBhatcoeff = np.sum(facposintegrand*dS+facnegintegrand*dS)

                showstring += "\n"+r"$BC = $"+f"{facBhatcoeff:.2f}"

    axdial.text(0, 0, titstr+"\nconductance", ha = 'center', va = 'center', size = 14)

    plt.subplots_adjust(hspace = .01, wspace = .01, left = .01, right = .99, bottom = .05, top = .99)

    axdial.set_xlim(-1.2, 1.2)
    axdial.set_ylim(-1.2, 1.2)

    cb = plt.colorbar(contn,cax=axcbar,orientation='horizontal')

    axcbar.set_yticks([])
    axcbar.set_xlabel('mho')

    axcbar.set_xticks([-15,-10,-5,0,5,10,15])

    axinfo.text(axinfo.get_xlim()[0], axinfo.get_ylim()[0], '$B_T$ = %s nT, $v$ = %s km/s,\n F$_{10.7}$ = %s, TILT $= %s^\circ$' % (B, v, f107val, tilt), ha = 'left', va = 'top', size = 12)

    plt.savefig(plotdir+'swipe_sigma'+conductance+'_ns_comparison_' + str(tilt) + '_deg_tilt.png', dpi = 300)

print("Save data ...")
plt.show()
