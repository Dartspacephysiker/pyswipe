# IPython log file

# import swipe
import numpy as np
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
get_var_str = 'convection'
# get_var_str = 'efield'

bys = np.arange(-5,5.1,5.)
bz = -5

swvel = 360
f107val = 120

savefig = False

##############################

thiss = []
potss = []
zero_latNU = None

minlat = 45
minlat_for_cpcp_calc = 70

# bzs = [-3.,0.,3.]
# bzs = [-3., 0., 3.]
dptilts = np.arange(-30,30.1,15.)
dptilts = np.arange(-20,20.1,20.)

Edn = dict()                    # E-field dictionary, north
Eds = dict()
# for i in range(len(coefffiles)):
i = 0

# lists for holding E-field for each bz value
ENss, ESss = [], []
for k,dptilt in enumerate(dptilts):

    # lists for holding CPCP in each hemisphere and the locations of extrema
    ENs, ESs = [], []
    for j,by in enumerate(bys):

        print(i,k,j,by,bz,dptilt)

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

            sizes = int(np.sqrt(np.split(mlats,2)[0].size))  # size of scalar grid

            # reshaper for scalar grid
            def reshapes(q):
                qn, qs = np.split(q,2)
                qn, qs = qn.reshape(sizes,sizes), qs.reshape(sizes,sizes)
                return qn, qs

            # lat-averaged stats 
            def latstats(qh):
                """'qh' means 'quantity in a particular hemisphere', 
                so q after running it through 'reshapes'
                """
                return np.mean(qh,axis=0), np.std(qh,axis=0)

            mlatsnu, mlatssu = mlatsn.reshape(100,100)[0,:], mlatss.reshape(100,100)[0,:]

        else:
            this.update_model(-swvel, by, bz, dptilt, f107val)

        # phin, phis = np.split(this.get_potential(), 2)
        # E1, E2: E_d1 and E_d2 field-perp components in Modified Apex coordinates
        if get_var_str == 'efield':
            E1, E2 = this.get_efield_MA(mlat=mlats, mlt=mlts,return_magnitude=False)
        elif get_var_str == 'convection':
            E1, E2, Emag = this.get_convection_vel_MA(mlat=mlats, mlt=mlts,return_magnitude=True)

        E1n, E1s = reshapes(E1)
        E2n, E2s = reshapes(E2)

        E1nmean, E1nstd = latstats(E1n)
        E2nmean, E2nstd = latstats(E2n)

        E1smean, E1sstd = latstats(E1s)
        E2smean, E2sstd = latstats(E2s)

        if get_var_str == 'convection':
            Emagn, Emags = reshapes(Emag)
            Emagnmean, Emagnstd = latstats(Emagn)
            Emagsmean, Emagsstd = latstats(Emags)

            ENs.append({1:[E1nmean,E1nstd],2:[E2nmean,E2nstd],'mag':[Emagnmean,Emagnstd]})
            ESs.append({1:[E1smean,E1sstd],2:[E2smean,E2sstd],'mag':[Emagsmean,Emagsstd]})
        else:
            ENs.append({1:[E1nmean,E1nstd],2:[E2nmean,E2nstd]})
            ESs.append({1:[E1smean,E1sstd],2:[E2smean,E2sstd]})

    ENss.append(ENs)
    ESss.append(ESs)

Edn[i] = ENss
Eds[i] = ESss

########################################

cmap = 'magma'
cmap2 = 'bwr'
labs = ['NH','SH',r"NH$-$SH"]

if get_var_str == 'efield':
    unitstr = 'mV/m'
    l1 = r"$E_{d1}$ (E-W)"
    l2 = r"$E_{d2}$ (N-S)"
elif get_var_str == 'convection':
    unitstr = 'm/s'
    l1 = r"$v_{e1}$ (E-W)"
    l2 = r"$v_{e2}$ (N-S)"
    l3 = r"$v_{\mathrm{mag}}$"

tilti = 0
byi = 2

tmptilt, tmpby = dptilts[tilti], bys[byi]

# E1nmean, E1nstd = Edn[0][tilti][byi][1]
# E2nmean, E2nstd = Edn[0][tilti][byi][2]

# E1smean, E1sstd = Eds[0][tilti][byi][1]
# E2smean, E2sstd = Eds[0][tilti][byi][2]

fig = plt.figure(9,figsize = (10, 10) if get_var_str == 'efield' else (15,10))
fig.clf()
tottitl = f"$B_y$={tmpby:.0f}, $B_z$={bz:.0f}"
fig.suptitle(tottitl)
if get_var_str == 'convection':
    ax1 = plt.subplot(331)
    ax2 = plt.subplot(332)
    ax3 = plt.subplot(333)
    ax4 = plt.subplot(334)
    ax5 = plt.subplot(335)
    ax6 = plt.subplot(336)
    ax7 = plt.subplot(337)
    ax8 = plt.subplot(338)
    ax9 = plt.subplot(339)

    axrows = [[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]]

elif get_var_str == 'efield':
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    ax3 = plt.subplot(323)
    ax4 = plt.subplot(324)
    ax5 = plt.subplot(325)
    ax6 = plt.subplot(326)
    
    axrows = [[ax1,ax2],[ax3,ax4],[ax5,ax6]]

stri1 = slice(None,None,4)
stri2 = slice(2,None,4)

tilt_i = [(2,0),(1,1),(0,2)]    # Assumes dptilts = [-15,0,15]. Local summer first, then equinox, then local winter

for i in range(len(dptilts)):

    ni,si = tilt_i[i]
    E1nmean, E1nstd = Edn[0][ni][byi][1]
    E2nmean, E2nstd = Edn[0][si][byi][2]

    E1smean, E1sstd = Eds[0][ni][byi][1]
    E2smean, E2sstd = Eds[0][si][byi][2]

    axrow = axrows[i]
    
    tmptilt = dptilts[i]

    plt.sca(axrow[0])

    plt.plot(mlatsnu        ,E1nmean,color='blue',label='NH')
    plt.plot(np.abs(mlatssu),E1smean,color='red' ,label='SH')
    plt.axvline(47,color='gray',linestyle=':')
    plt.fill_between(mlatsnu        ,E1nmean-E1nstd,E1nmean+E1nstd,color='blue',alpha=0.2,hatch='\\')
    plt.fill_between(np.abs(mlatssu),E1smean-E1sstd,E1smean+E1sstd,color='red' ,alpha=0.2,hatch='/')
    if i == 0:
        plt.title(l1)
        plt.legend(loc='upper left')
    
    plt.ylabel("["+unitstr+"]")
    if i == 2:
        plt.xlabel("MA Latitude [deg]")

    plt.sca(axrow[1])
    plt.plot(mlatsnu        ,E2nmean,color='blue',label='NH')
    plt.plot(np.abs(mlatssu),E2smean,color='red' ,label='SH')
    plt.axvline(47,color='gray',linestyle=':')
    plt.fill_between(mlatsnu        ,E2nmean-E2nstd,E2nmean+E2nstd,color='blue',alpha=0.2,hatch='\\')
    plt.fill_between(np.abs(mlatssu),E2smean-E2sstd,E2smean+E2sstd,color='red' ,alpha=0.2,hatch='/')
    if i == 0:
        plt.title(l2)

    if i == 2:
        plt.xlabel("MA Latitude [deg]")

    if get_var_str == 'convection':
        
        Emagnmean, Emagnstd = Edn[0][ni][byi]['mag']
        Emagsmean, Emagsstd = Eds[0][ni][byi]['mag']

        plt.sca(axrow[2])
        plt.plot(mlatsnu        ,Emagnmean,color='blue',label='NH')
        plt.plot(np.abs(mlatssu),Emagsmean,color='red' ,label='SH')
        plt.axvline(47,color='gray',linestyle=':')
        plt.fill_between(mlatsnu        ,Emagnmean-Emagnstd,Emagnmean+Emagnstd,color='blue',alpha=0.2,hatch='\\')
        plt.fill_between(np.abs(mlatssu),Emagsmean-Emagsstd,Emagsmean+Emagsstd,color='red' ,alpha=0.2,hatch='/')
        if i == 0:
            plt.title(l3)

tmp3 = fig.text(0.03,0.765,"Summer", size = 18, rotation='vertical', va='center')
tmp2 = fig.text(0.03,0.49,"Equinox", size = 18, rotation='vertical', va='center')
tmp1 = fig.text(0.03,0.22,"Winter" , size = 18, rotation='vertical', va='center')
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=None)

if savefig:
    fname = "16__lat_averages/"+get_var_str+'_by_season_'+tottitl.replace('$','').replace('_','').replace(' ','').replace('$^\circ$','deg')+'.png'
    fname = fname.replace("\\Psi","ψ").replace("^\\circ","deg").replace("\n","_").replace("{SW}","sw").replace("[","_").replace("]","").replace('CPCP_','')
    print("Saving fig to "+fname)
    plt.savefig(plotdir+fname,dpi=300)
        
else:
    print("NOT saving plot")
