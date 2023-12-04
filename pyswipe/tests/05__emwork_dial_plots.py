import numpy as np

from pyswipe.swipe import SWIPE

plotdir = '/plotdir/'

import matplotlib as mpl
mplBkgrnd = 'QtAgg'
mpl.use(mplBkgrnd)

mpl.rcParams.update({'text.color': 'k'})
mpl.rcParams.update({'axes.labelcolor': 'k'})
mpl.rcParams.update({'xtick.color': 'k'})
mpl.rcParams.update({'ytick.color': 'k'})
mpl.rcParams.update({'font.size': 10})
# mpl.rcParams.update({'font.family': 'sans-serif'})
# mpl.rcParams.update({'font.sans-serif': 'Arial'})
mpl.rcParams.update({'text.usetex': False})

mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
mpl.rcParams.update({'savefig.directory': plotdir})

import matplotlib.pyplot as plt
plt.ion()

from polplot.polplot import Polarplot

d2r = np.pi/180

dontbescaredofnegs = True

no_SH = False

B = 5                          # nT

ca = {(0, 1): 0, (0, 0):-45, (1, 0):-90,(2,0):-135,(2,1):180,(0,2):45,(1,2):90,(2,2):135}

tilts = [-25, 0, 25]
# tilts = [0]

v = 450
f107val = 120

minlat = 60
minlat_for_cpcp_calc = 60

# potlevels = np.arange(-55, 55, 5)+2.5
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

##############################
# Initialize models
swipe_n = SWIPE(v, B, B, 0, f107val,
             minlat=minlat)
swipe_s = SWIPE(v, B, B, 0, f107val,
             minlat=minlat)#,

mlat_n, mlt_n = map(lambda x:np.split(x, 2)[0], swipe_n.scalargrid)
mlat_s, mlt_s = map(lambda x:np.split(x, 2)[1], swipe_n.scalargrid)

# Ripped from pyamps.amps.get_integrated_upward_current
REFRE = 6371.2 # Reference radius used in geomagnetic modeling

# get surface area element in each cell:
mlat, mlt = swipe_n.scalargrid
mlt_sorted = np.sort(np.unique(mlt))
mltres = (mlt_sorted[1] - mlt_sorted[0]) * np.pi / 12
mlat_sorted = np.sort(np.unique(mlat))
mlatres = (mlat_sorted[1] - mlat_sorted[0]) * np.pi / 180
R = (REFRE + swipe_n.height) * 1e3  # radius in meters
dS = R**2 * np.cos(mlat * np.pi/180) * mlatres * mltres

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
            # swipe_n.update_model(-v, by, bz, dptilt, f107val)

            phin      = swipe_n.get_emwork(mlat_n, mlt_n).ravel()
            phis      = swipe_s.get_emwork(mlat_s, mlt_s).ravel()

            # add a constant to both potentials to make it easier to compare
            # half  = (phin.max() - phin.min()) / 2
            # phis = phis - phis.min() - half
            # phin = phin - phin.min() - half

            # reverse sign of phis to facilitate comparison
            # phis = -phis

            dPhiN = np.sum(phin*dS)/1e3/1e9  # 1e3 to junk 'milli' prefix in mW/m², 1e12 to go to GW
            dPhiS = np.sum(phis*dS)/1e3/1e9

    
            tmpcpcpn.append(dPhiN)
            tmpcpcps.append(dPhiS)

            contn = axes[i][j].contourf(mlat_n, mlt_n, phin, levels = potlevels, cmap = cmap, extend = extend)

            if not no_SH:
                conts = axes[i][j].contour (mlat_s, mlt_s, phis, levels = potlevels, colors = 'k' , extend = extend, linewidths = .4)

                showstring = f'$W_N=$ {dPhiN:.0f}' + f' GW\n$W_S=$ {dPhiS:.0f}' + f' GW\n$W_N/W_S=$ {dPhiN/dPhiS:.2f}'

            else:
                
                showstring = f'$W_N=$ {dPhiN:.0f} GW'


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

            axes[i][j].write(60, 10.2, showstring,
                             ha = 'left', va = 'top', size = 9,
                             multialignment='left',
                             ignore_plot_limits=True)

    axdial.text(0, 0, r"$\mathbf{J}\cdot\mathbf{E}$ work"+u'\n(Earth-fixed frame)', ha = 'center', va = 'center', size = 14)

    plt.subplots_adjust(hspace = .01, wspace = .01, left = .01, right = .99, bottom = .05, top = .99)

    axdial.set_xlim(-1.2, 1.2)
    axdial.set_ylim(-1.2, 1.2)

    cb = plt.colorbar(contn,cax=axcbar,orientation='horizontal')

    axcbar.set_yticks([])
    axcbar.set_xlabel('mW/m$^2$')

    axcbar.set_xticks([-15,-10,-5,0,5,10,15])

    # Save cpcp info
    cpcpn.append(np.array(tmpcpcpn))
    cpcps.append(np.array(tmpcpcps))
    locns.append(np.array(tmplocn))
    locss.append(np.array(tmplocn))

    if np.isclose(tilt,0):
        tiltsign = ''
    elif tilt > 0:
        tiltsign = "±"
    else:
        tiltsign = "∓"
    axinfo.text(axinfo.get_xlim()[0], axinfo.get_ylim()[0], '$B_T$ = %s nT, $v$ = %s km/s,\n F$_{10.7}$ = %s, TILT $= %s%s^\circ$' % (B, v, f107val, tiltsign, np.abs(tilt)), ha = 'left', va = 'top', size = 12)

    if tilt >= 0:
        tiltstr = f"p{tilt}"
    else:
        tiltstr = f"n{np.abs(tilt)}"
    if no_SH:
        figname = plotdir+'swipe_emwork_ns_comparison_' + tiltstr + '_deg_tilt_noSH.png'
    else:
        figname = plotdir+'swipe_emwork_ns_comparison_' + tiltstr + '_deg_tilt.png'
    plt.savefig(figname, dpi = 300)

print("Save data ...")
plt.show()
