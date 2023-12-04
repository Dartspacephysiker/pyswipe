# IPython log file

# import swipe
import numpy as np
import pyswipe.swipe
from pyswipe.swipe import SWIPE
from pyswipe.sh_utils import SHkeys

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
mpl.rcParams.update({'text.usetex': False})

mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
mpl.rcParams.update({'savefig.directory': plotdir})

import matplotlib.pyplot as plt
plt.ion()

d2r = np.pi/180

dptilts = [-25,0,25]

# BTs = [3,5]                          # nT
# BTs = [1,7]                          # nT
cas = np.arange(-180,180,15)
BTs = np.arange(0.,10.1,0.2)

cas = np.arange(-180,180,15)
BTs = np.arange(1.,10.1,0.5)


CA, BT = np.meshgrid(cas,BTs,indexing='ij')

print(f"N iters: {CA.size}")

swvel = 450
f107val = 120

savefig = True

nomirror = True

mirrorstr = ''
if nomirror:
    mirrorstr = '_nomirror'
##############################
# Select model coeffs, open 'er up

minlat = 45

# dPhidn = dict()
# dPhids = dict()
# locdn = dict()
# locds = dict()

dr = 1                          # deltalat for vector grid, which is equal-area

# lists for holding CPCP for each bz value
dPhiNss, dPhiSss = [], []
locnss, locsss = [], []
for dptilt in dptilts:

    # lists for holding CPCP in each hemisphere and the locations of extrema
    dPhiNs, dPhiSs = [], []
    locns, locss = [], []
    for j,(_ca,B) in enumerate(zip(CA.ravel(),BT.ravel())):

        by, bz = np.sin(_ca * d2r) * B, np.cos(_ca * d2r) * B

        if j == 0:
            swipen = SWIPE(-swvel,by,bz,dptilt,f107val,
                           minlat=minlat,
                           dr=dr)

            if nomirror:
                swipes = swipen
            else:
                swipes = SWIPE(-swvel,-by,bz,-dptilt,f107val,
                               minlat=minlat,
                               dr=dr)

            mlats, mlts = swipen.scalargrid
            mlatv, mltv = swipen.vectorgrid
            mlatvn, mlatvs = np.split(mlatv, 2)
            mltvn, mltvs = np.split(mltv, 2)
            mlatsn, mlatss = np.split(mlats, 2)
            mltsn, mltss = np.split(mlts, 2)

        else:
            swipen.update_model(-swvel, by, bz, dptilt, f107val)
            if nomirror:
                swipes = swipen
            else:
                swipes.update_model(-swvel, -by, bz, -dptilt, f107val)

        phin = swipen.get_potential(mlatvn,mltvn)
        phis = swipes.get_potential(mlatvs,mltvs)

        phin = phin - np.median(phin)
        phis = phis - np.median(phis)

        dPhiN = phin.max()-phin.min()
        dPhiS = phis.max()-phis.min()

        print(j,_ca,B,f"{dPhiN:6.1f}",f"{dPhiS:6.1f}",f"{dPhiN-dPhiS:6.1f}")

        minn,maxn = np.argmin(phin),np.argmax(phin)
        mins,maxs = np.argmin(phis),np.argmax(phis)

        if (j % 20) == 0:
            mlatminn,mltminn = mlatvn[minn],mltvn[minn]
            mlatmins,mltmins = mlatvs[mins],mltvs[mins]
            mlatmaxn,mltmaxn = mlatvn[maxn],mltvn[maxn]
            mlatmaxs,mltmaxs = mlatvs[maxs],mltvs[maxs]

            print("North mlatmin, mltmin, mlatmax, mltmax: ",mlatminn, mltminn, mlatmaxn, mltmaxn)
            print("South mlatmin, mltmin, mlatmax, mltmax: ",mlatmins, mltmins, mlatmaxs, mltmaxs)

        dPhiNs.append(dPhiN)
        dPhiSs.append(dPhiS)
        locns.append([minn,maxn])
        locss.append([mins,maxs])


    dPhiNs = np.array(dPhiNs)
    dPhiSs = np.array(dPhiSs)

    dPhiNs = dPhiNs.reshape(CA.shape)
    dPhiSs = dPhiSs.reshape(CA.shape)

    locns, locss = np.array(locns), np.array(locss)

    dPhiNss.append(dPhiNs)
    dPhiSss.append(dPhiSs)
    locnss.append(locns)
    locsss.append(locns)

########################################
#NH/SH MASTER ASYMM COEFF 
cmap = 'magma'
cmap2 = 'bwr'
# labs = ['NH','SH',r"2*(NH-SH)/(NH+SH)"]
if nomirror:
    labs = [r"$\Delta \Phi_N(B_y, B_z, \Psi)$",
            r"$\Delta \Phi_S(B_y, B_z, \Psi)$",
            r'$2\frac{\Delta \Phi_N - \Delta \Phi_S}{\Delta \Phi_N + \Delta \Phi_S}$']
    seaslabs = [r'$\Psi=-25^\circ$',r'$\Psi=0^\circ$',r'$\Psi=25^\circ$']
else:
    labs = [r"$\Delta \Phi_N(B_y, B_z, \Psi)$",
            r"$\Delta \Phi_S(-B_y, B_z, -\Psi)$",
            r'$2\frac{\Delta \Phi_N - \Delta \Phi_S}{\Delta \Phi_N + \Delta \Phi_S}$']
    seaslabs = ['Local\nWinter','Equinox','Local\nSummer']
seaslabs2 = ['Winter','Equinox','Summer']


fig = plt.figure(14,figsize = (14, 12))

plt.clf()

vmax = 0
vmax2 = 0
for j,dptilt in enumerate(dptilts):

    dPhiNs = dPhiNss[j]
    dPhiSs = dPhiSss[j]

    vmin = 0
    tvmax = np.max([dPhiNs,dPhiSs])
    
    vmax = np.max([tvmax,vmax])

    coeff = (dPhiNs-dPhiSs)/ ( (dPhiNs+dPhiSs) / 2 )

    tvmax2 = np.max(np.abs([np.max(coeff),np.min(coeff)]))
    vmax2 = np.max([tvmax2,vmax2])
    vmin2 = -vmax2


plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=0.4)
    
for j,dptilt in enumerate(dptilts):

    # plt.clf()
    # tottitl = f"CPCP\n$\Psi$={dptilt}$^\circ$, F10.7={f107val}"
    
    # fig.suptitle(tottitl)
    
    dPhiNs = dPhiNss[j]
    dPhiSs = dPhiSss[j]

    # vmin = np.min([dPhiNs,dPhiSs])
    # vmax = np.max([dPhiNs,dPhiSs])
    # vmin = 0
    
    coeff = (dPhiNs-dPhiSs)/ ( (dPhiNs+dPhiSs) / 2 )

    # vmax2 = np.max(np.abs([np.max(coeff),np.min(coeff)]))
    # vmin2 = -vmax2
    # vmax2 = 10
    # vmin2 = -10
    
    ax0 = plt.subplot2grid((4, 14), (j,  0), colspan = 4)
    ax1 = plt.subplot2grid((4, 14), (j,  4), colspan = 4)
    ax2 = plt.subplot2grid((4, 14), (j,  8), colspan = 4)
    # plt.text(r"DAD",-0.2,0.5,rotation='horizontal',clip_on=False,transform=ax0.transAxes)

    plt.sca(ax0)
    # ax0.set_aspect('equal')
    if j == 0:
        ax0.set_title(labs[0])
    if j == 2:
        plt.xlabel("$\phi_c$ [deg]")
    plt.ylabel("$B_T$ [nT]")
    imn = plt.pcolormesh(CA,BT,dPhiNs,vmin=vmin,vmax=vmax,cmap=cmap)
    
    plt.sca(ax1)
    # ax1.set_aspect('equal')
    if j == 0:
        ax1.set_title(labs[1])
    if j == 2:
        plt.xlabel("$\phi_c$ [deg]")
    _ = [ylab.set_visible(False) for ylab in ax1.yaxis.get_ticklabels()]
    ims = plt.pcolormesh(CA,BT,dPhiSs,vmin=vmin,vmax=vmax,cmap=cmap)
    
    plt.sca(ax2)
    # ax2.set_aspect('equal')
    if j == 0:
        ax2.set_title(labs[2])
    if j == 2:
        plt.xlabel("$\phi_c$ [deg]")
    _ = [ylab.set_visible(False) for ylab in ax2.yaxis.get_ticklabels()]
    imd = plt.pcolormesh(CA,BT,coeff,vmin=vmin2,vmax=vmax2,cmap=cmap2)
    
    # if j == 0:
    figtxtcoords = fig.transFigure.inverted().transform(ax0.transAxes.transform((0.,0.5)))
    figtxtcoords[0] -= 0.1
    fig.text(*figtxtcoords,seaslabs[j],rotation='vertical',va='center',ha='center',fontsize=18)

axc  = plt.subplot2grid((4, 100), (0, 88), rowspan=3, colspan = 3)
axc2 = plt.subplot2grid((4, 100), (0, 96), rowspan=3, colspan = 3)

cb1 = plt.colorbar(imn,cax=axc)
cb2 = plt.colorbar(imd,cax=axc2)
    

# Now master panel
coeffs = [(dPhiNs-dPhiSs)/ ( (dPhiNs+dPhiSs) / 2 ) for dPhiNs,dPhiSs in zip(dPhiNss,dPhiSss)]

avgs = [np.mean(coeff,axis=0) for coeff in coeffs]

ax = plt.subplot2grid((4, 14), (3, 0), colspan = 12)

linestyles = ['-','-','-']
colors = ['black','gray','lightgray']
linewidths = [2,3,4]
plt.sca(ax)
plt.xlabel("$B_T$ [nT]")
# plt.ylabel("$< C_A >$")
# plt.ylabel("2*( CPCP$_N - $CPCP$_S$ )/( CPCP$_N + $CPCP$_S$ )")
plt.ylabel(r'$2\frac{\Delta \Phi_N - \Delta \Phi_S}{\Delta \Phi_N + \Delta \Phi_S}$')

_ = ax.spines['right'].set_visible(False)
_ = ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
_ = ax.yaxis.set_ticks_position('left')
_ = ax.xaxis.set_ticks_position('bottom')

for j,dptilt in enumerate(dptilts):
    
    idx = j
    avg = avgs[idx]

    ax.plot(BTs,avg,label=seaslabs[idx],color=colors[idx],linestyle=linestyles[idx],lw=linewidths[idx])

#    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=None)
    
ax.legend(ncols=3)


if savefig:
    # tottitl = paramstr + "\n" + titls[0]
    fname = "MASTER_"+tottitl.replace('$','').replace('_','').replace(' ','').replace('$^\circ$','deg')+mirrorstr+'.png'
    fname = fname.replace("\\Psi","ψ").replace("^\\circ","deg").replace("\n","_").replace("{SW}","sw").replace("[","_").replace("]","").replace('CPCP_','')
    print("Saving fig to "+fname)
    plt.savefig(plotdir+fname,dpi=300)
        
        
