# import swipe
import numpy as np
import os
from pyswipe.swipe import SWIPE

plotdir = '/plotdir/'

import matplotlib as mpl
mplBkgrnd = 'QtAgg'
mpl.use(mplBkgrnd)

# MPL opts

mpl.rcParams.update({'text.color': 'k'})
mpl.rcParams.update({'axes.labelcolor': 'k'})
mpl.rcParams.update({'xtick.color': 'k'})
mpl.rcParams.update({'ytick.color': 'k'})
mpl.rcParams.update({'font.size': 15})
# mpl.rcParams.update({'font.family': 'sans-serif'})
# mpl.rcParams.update({'font.sans-serif': 'Arial'})
mpl.rcParams.update({'text.usetex': False})

mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
mpl.rcParams.update({'savefig.directory': plotdir})

import matplotlib.pyplot as plt
plt.ion()

##############################
# Data from other figures

# clock angles
thetac = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
thetacrad = np.array([0, 1/4, 1/2, 3/4, 1, 5/4, 3/2, 7/4, 2]) * np.pi

#Förster and Haaland (2015) Figures 2 (NH) and 3 (SH)
# They simply group all available data by IMF clock angle sector, and ensure that the bias vector length (see Section 3 of Haaland et al, 2007, for description) is less than 0.96
# They could not screen for seasonal dependence, since Cluster coverage varies with season
fhn = np.array([16.7, 28.2, 45.1, 61.0, 70.6, 63.7, 42.4, 22.9, 16.7])
fhs = np.array([16.1, 24.9, 42.8, 60.5, 75.0, 67.0, 42.8, 28.0, 16.1])


# Cousins and Shepherd (2010) Figure 3
# 2.2 < E_{sw} < 2.9 mV/m$, where $E_{sw} = |V_x B_T|$ is the "solar wind electric field magnitude."
# (Can get E_sw = 2.5 mV/m via for example V = 500 km/s and B_T = 5 nT, 
# NH for tilt > 10deg and in the SH for tilt < -10deg (SH second number)
cousinsn = np.array([20, 30, 40, 53, 67, 65, 50, 38, 20])
cousinss = np.array([11, 20, 37, 54, 71, 53, 37, 23, 11])


# Weimer (2005) Figure 2
# B_T = 5 nT
# V = 450 km/s
#N_sw = 4.0/cc
#tilt = 0 deg
weimer = np.array([24, 40, 69, 93, 102, 90, 58, 32, 24])


# Papitashvili and Rich (2002) Figure 2
#B_T = 5 nT
#n = 5 cm^{-3},
#V = 400 km/s.
#Dipole tilt is approximately zero, since they use all data centered around each equinox in a two-month window. NH and SH are given as first and second numbers respectively:
papitashn = np.array([11, 22, 40, 69, 79, 65, 33, 19, 11])
papitashs = np.array([14, 18, 33, 69, 88, 78, 45, 28, 14])

#For our comparison, we use $V_x = 450$~m/s and $B_T = 5$~nT, which corresponds to $E_{sw} = 2.25$~mV/m.

bt = 5                          # nT
#thetas = thetacrad
thetacraddeets = np.arange(0,2.01,0.025) * np.pi

bzs = bt * np.cos(thetacraddeets)
bys = bt * np.sin(thetacraddeets)
dptilt = 0.
# dptilts = np.arange(-33,33.1,11.)
# dptilts = np.arange(-22,22.1,11.)
# BY, DP = np.meshgrid(bys,dptilts,indexing='ij')

swvel = 450
#swden  = 5.590                  # NOTE: SWIPE does not use swdensity
# dptilt = -1.0
f107val = 120

savefig = True

##############################

zero_latNU = None

minlat = 47
minlat_for_cpcp_calc = 60

dPhidn = dict()
dPhids = dict()
locdn = dict()
locds = dict()
for i in range(1):

    # lists for holding CPCP for each bz value
    dPhiNss, dPhiSss = [], []
    locnss, locsss = [], []
    for j,(by,bz) in enumerate(zip(bys,bzs)):
    
        print(f"i={j}, by={by:.2f}, bz={bz:.2f}")
    
        # lists for holding CPCP in each hemisphere and the locations of extrema
        dPhiNs, dPhiSs = [], []
        locns, locss = [], []
    
        # paramstr = f"$B_y$={by}, $B_z$={bz},"+" $v_{SW}$="+f"{swvel}"
        # tottitl = paramstr + titls[i]
    
        # paramstr = f"$B_y$={by}, $B_z$={bz}, $\Psi$={tilt}$^\circ$,F10.7={f107val}\n"
        # print(f"v = {v} km/s, By = {by} nT, Bz = {bz} nT, Psi = {tilt} deg, F10.7 = {f107}")
    
        if j == 0:
            this = SWIPE(-swvel,by,bz,dptilt,f107val,
                         minlat=minlat,
                         zero_lats=zero_latNU)
    
            mlats, mlts = this.scalargrid
            mlatv, mltv = this.vectorgrid
            mlatvn, mlatvs = np.split(mlatv, 2)
            mltvn, mltvs = np.split(mltv, 2)
            mlatsn, mlatss = np.split(mlats, 2)
            mltsn, mltss = np.split(mlts, 2)
    
        else:
            this.update_model(-swvel, by, bz, dptilt, f107val)
    
        phin, phis = np.split(this.get_potential(), 2)
        phin = phin - np.median(phin)
        phis = phis - np.median(phis)
    
        OKindsn, OKindss = np.split(np.abs(mlats).flatten() >= minlat_for_cpcp_calc,2)
        # mlatsn, mlatss = np.split(mlats,2)
        # mltsn, mltss = np.split(mlts,2)
        # cpcpmlatsn,cpcpmltsn = mlatsn.flatten()[OKindsn],mltsn.flatten()[OKindsn]
        # cpcpmlatss,cpcpmltss = mlatss.flatten()[OKindss],mltss.flatten()[OKindss]
        dPhiN = phin[OKindsn].max()-phin[OKindsn].min()
        dPhiS = phis[OKindss].max()-phis[OKindss].min()
        minn,maxn = np.argmin(phin[OKindsn]),np.argmax(phin[OKindsn])
        mins,maxs = np.argmin(phis[OKindss]),np.argmax(phis[OKindss])
        
        dPhiNs.append(dPhiN)
        dPhiSs.append(dPhiS)
        locns.append([minn,maxn])
        locss.append([mins,maxs])
    
        dPhiNs = np.array(dPhiNs)
        dPhiSs = np.array(dPhiSs)
    
        # dPhiNs = dPhiNs.reshape(BY.shape)
        # dPhiSs = dPhiSs.reshape(BY.shape)
    
        locns, locss = np.array(locns), np.array(locss)
    
        dPhiNss.append(dPhiNs)
        dPhiSss.append(dPhiSs)
        locnss.append(locns)
        locsss.append(locns)
    
    dPhidn[i] = dPhiNss
    dPhids[i] = dPhiSss
    
    locdn[i] = locnss
    locds[i] = locsss
    
cpcpn = np.array(dPhidn[0]).ravel()
cpcps = np.array(dPhids[0]).ravel()


# REAL THING
cpcp = [cpcpn, cpcps]
fh = [fhn, fhs]
papitash = [papitashn,papitashs]
cousins = [cousinsn,cousinss]
weimer = weimer
titles = ['North','South']

weimerdeets = np.interp(thetacraddeets, thetacrad, weimer)
fhdeets = [np.interp(thetacraddeets, thetacrad, fhh) for fhh in fh]
papitashdeets = [np.interp(thetacraddeets, thetacrad, papitashv) for papitashv in papitash]
papitashdeets = [np.interp(thetacraddeets, thetacrad, papitashv) for papitashv in papitash]
cousinsdeets = [np.interp(thetacraddeets, thetacrad, cousinz) for cousinz in cousins]

showdeets = True
showdots = True and showdeets
showdotsize = 20
if showdeets:
    guestthetarad = thetacraddeets
    weimershow = weimerdeets
    fhshow = fhdeets
    papitashow = papitashdeets
    cousinshow = cousinsdeets
else:
    guestthetarad = thetacrad
    weimershow = weimer
    fhshow = fh
    papitashow = papitash
    cousinshow = cousins

fig = plt.figure(3,figsize=(14,7))
plt.clf()
ax0 = plt.subplot(121,projection='polar')
ax1 = plt.subplot(122,projection='polar')
rticks = [30, 60, 90]
rticks = [50,100]

weimercolor = 'black'
wkey = dict(color=weimercolor, linestyle='-',linewidth=1.5)
fhkey = dict(linestyle='-.',linewidth=2.2)
prkey = dict(linestyle='--', linewidth=1.5)
cskey = dict(linestyle=':', linewidth=2)

    
for i,ax in enumerate([ax0,ax1]):
    ax.set_theta_zero_location("N")  # theta=0 at the top
    ax.set_theta_direction(-1)  # theta increasing clockwise
    ax.set_rlabel_position(67.5)  # Move radial labels away from legend
    ax.set_rlabel_position(45)  # Move radial labels away from legend
    ax.spines['polar'].set_visible(False)

    # ax.set_rticks(rticks,labels=[f"{num} kV" for num in rticks])  # Less radial ticks
    ax.set_rticks(np.array(rticks),labels=[f"{num} kV" for num in rticks])  # Less radial ticks

    ax.set_xticks(np.array([-1/2, 0, 1/2, 1])*np.pi, labels=["270$^\circ$", r"$\theta_c = 0^\circ$", f"90$^\circ$", "180$^\circ$"])
    ax.set_thetalim(-np.pi, np.pi)

    ax.set_title(titles[i])
    
    nalpha = 0.1 if i == 1 else 1.0
    salpha = 0.1 if i == 0 else 1.0

    ncolor = 'blue'
    scolor = 'red'

    ncolor = 'blue' if i == 0 else 'black'
    scolor = 'red' if i == 1 else 'black'

    # ORIG LEGEND STUFF
    hicnlabel = 'Swarm Hi-C'
    pr02nlabel = 'PR02'
    cs10nlabel = 'CS10'
    fh15nlabel = 'FH15'
    wlabel = 'W05'

    hicslabel = None
    pr02slabel = None
    cs10slabel = None
    fh15slabel = None

    # NEW LEGEND STUFF
    if i == 0:
        # hicnlabel = 'Hi-C'
        # pr02nlabel = 'PR02'
        # cs10nlabel = 'CS10'
        # fh15nlabel = 'FH15'
        # wlabel = 'W05'
    
        hicnlabel = 'Hi-C (this study)'
        pr02nlabel = 'P&R (2002)'
        cs10nlabel = 'C&S (2010)'
        fh15nlabel = 'F&H (2015)'
        wlabel = 'Weimer (2005)'

        hicnlabel = 'Hi-C (this study)'
        pr02nlabel = 'PR02'
        cs10nlabel = 'CS10'
        fh15nlabel = 'FH15'
        wlabel = 'W05'

        hicslabel = None
        pr02slabel = None
        cs10slabel = None
    elif i == 1:
        hicnlabel = None
        pr02nlabel = None
        fh15nlabel = None
        cs10nlabel = None
        wlabel = 'W05'
    
        hicslabel = 'Hi-C'
        pr02slabel = 'PR02'
        fh15slabel = 'FH15'
        cs10slabel = 'CS10'
    

    # Cousins and Shepherd (2010)
    ax.plot(guestthetarad, cousinshow[0],label=cs10nlabel, **cskey, color=ncolor, alpha=nalpha)
    ax.plot(guestthetarad, cousinshow[1],label=cs10slabel, **cskey, color=scolor , alpha=salpha)
    if showdots:
        ax.scatter(thetacrad, cousins[0], s=showdotsize, color=ncolor, alpha=nalpha)
        ax.scatter(thetacrad, cousins[1], s=showdotsize, color=scolor , alpha=salpha)

    # Förster and Haaland (2015)
    ax.plot(guestthetarad, fhshow[0],label=fh15nlabel, **fhkey, color=ncolor, alpha=nalpha)
    ax.plot(guestthetarad, fhshow[1],label=fh15slabel, **fhkey, color=scolor , alpha=salpha)
    if showdots:
        ax.scatter(thetacrad, fh[0], s=showdotsize, color=ncolor, alpha=nalpha)
        ax.scatter(thetacrad, fh[1], s=showdotsize, color=scolor , alpha=salpha)

    # Papitashvili and Rich (2002)
    ax.plot(guestthetarad, papitashow[0],label=pr02nlabel, **prkey, color=ncolor, alpha=nalpha)
    ax.plot(guestthetarad, papitashow[1],label=pr02slabel, **prkey, color=scolor , alpha=salpha)
    if showdots:
        ax.scatter(thetacrad, papitash[0], s=showdotsize, color=ncolor, alpha=nalpha)
        ax.scatter(thetacrad, papitash[1], s=showdotsize, color=scolor , alpha=salpha)
    
    # Swarm Hi-C (this study, Hatch et al)
    ax.plot(thetacraddeets, cpcp[0], color=ncolor,label=hicnlabel,linewidth=4,
            alpha = nalpha)
    ax.plot(thetacraddeets, cpcp[1], color=scolor,label=hicslabel,linewidth=4,
            alpha = salpha)
    
    # Weimer (2005)
    ax.plot(guestthetarad, weimershow, label='W05', **wkey)
    if showdots:
        ax.scatter(thetacrad, weimer, s=showdotsize, color=weimercolor)
    
    
    # if i == 0:
    #     ax.legend(loc='upper center')
    
    if i == 0:
        ax.legend(loc='upper center')

    # ax.set_rmax(2)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    
plt.tight_layout()


plt.savefig(plotdir+'CPCP_comparison2.png',dpi=300)
