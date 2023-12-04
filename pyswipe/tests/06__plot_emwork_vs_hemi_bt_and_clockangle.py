# IPython log file

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
mpl.rcParams.update({'font.size': 14})
# mpl.rcParams.update({'font.family': 'sans-serif'})
# mpl.rcParams.update({'font.sans-serif': 'Arial'})
mpl.rcParams.update({'text.usetex': False})

mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
mpl.rcParams.update({'savefig.directory': plotdir})

import matplotlib.pyplot as plt
plt.ion()

d2r = np.pi/180


# FINE diffs
dptilts = [-25,0,25]

# BTs = [3,5]                          # nT
# BTs = [1,7]                          # nT
BTs = np.arange(0.,10.1,0.2)
cas = np.arange(-180,180,15)

# TEST
BTs = np.arange(1.,10.1,1.0)
cas = np.arange(-180,180,30)

# TEST
BTs = np.arange(1.,10.1,0.5)
cas = np.arange(-180,180,15)

# BTs = np.arange(1.,10.1,1.0)
# cas = np.arange(-180,180,30)

CA, BT = np.meshgrid(cas,BTs,indexing='ij')

print(f"N iters: {CA.size}")

swvel = 450
f107val = 120

savefig = True

minlat = 60

dr = 1                          # deltalat for vector grid, which is equal-area

##############################
#Grid area for integrated EM calc
# Ripped from pyamps.amps.get_integrated_upward_current
REFRE = 6371.2 # Reference radius used in geomagnetic modeling

swipen = SWIPE(-swvel,0.,0.,dptilts[0],f107val,
               minlat=minlat,
               dr=dr)

# get surface area element in each cell:
mlat, mlt = swipen.scalargrid
mlt_sorted = np.sort(np.unique(mlt))
mltres = (mlt_sorted[1] - mlt_sorted[0]) * np.pi / 12
mlat_sorted = np.sort(np.unique(mlat))
mlatres = (mlat_sorted[1] - mlat_sorted[0]) * np.pi / 180
R = (REFRE + swipen.height) * 1e3  # radius in meters
dS = R**2 * np.cos(mlat * np.pi/180) * mlatres * mltres

dS_n,dS_s = np.split(dS, 2)
assert np.array_equal(dS_n,dS_s),"Uh oh, calculations later assume dS_n == dS_s!"
dS = dS_n.ravel()

mlat_n, mlt_n = map(lambda x:np.split(x, 2)[0], swipen.scalargrid)
mlat_s, mlt_s = map(lambda x:np.split(x, 2)[1], swipen.scalargrid)


##############################
#Loop

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

            swipes = SWIPE(-swvel,-by,bz,-dptilt,f107val,
                           minlat=minlat,
                           dr=dr)

        else:
            swipen.update_model(-swvel, by, bz, dptilt, f107val)
            swipes.update_model(-swvel, -by, bz, -dptilt, f107val)

        phin      = swipen.get_emwork(mlat_n, mlt_n).ravel()
        phis      = swipes.get_emwork(mlat_s, mlt_s).ravel()

        dPhiN = np.sum(phin*dS)/1e3/1e9  # 1e3 to junk 'milli' prefix in mW/m², 1e12 to go to GW
        dPhiS = np.sum(phis*dS)/1e3/1e9

        print(j,_ca,B,f"{dPhiN:6.1f}",f"{dPhiS:6.1f}",f"{dPhiN-dPhiS:6.1f}")

        dPhiNs.append(dPhiN)
        dPhiSs.append(dPhiS)

    dPhiNs = np.array(dPhiNs)
    dPhiSs = np.array(dPhiSs)

    dPhiNs = dPhiNs.reshape(CA.shape)
    dPhiSs = dPhiSs.reshape(CA.shape)

    dPhiNss.append(dPhiNs)
    dPhiSss.append(dPhiSs)

########################################
#NH/SH MASTER ASYMM COEFF 
cmap = 'magma'
cmap2 = 'bwr'
# labs = ['NH','SH',r"2*(NH-SH)/(NH+SH)"]
labs = [r"$W_N(B_y, B_z, \Psi)$",
        r"$W_S(-B_y, B_z, -\Psi)$",
        r'$A_{EM} = 2\frac{W_N - W_S}{W_N + W_S}$']
seaslabs = ['Local\nWinter','Equinox','Local\nSummer']
seaslabs2 = ['W','E','S']

extend = 'max'


CAuse = CA.copy()
BTuse = BT.copy()
xticks = [-180,-90,0,90,180]
xtickstr = ['-180','-90','0','90','']
shading = None

doroll = True

rollfunc = lambda x: np.roll(x,-int(cas.size//2),axis=0)
if doroll:

    CAmod = CA.copy()
    CAmod[CAmod < 0] = CAmod[CAmod < 0] + 360


    CAuse = CAmod

    BTuse = BT.copy()
    

    CAuse = rollfunc(CAuse)
    BTuse = rollfunc(BTuse)

    CAuse = np.vstack([CAuse,np.array([360]*BTs.size)])
    BTuse = np.vstack([BTuse,BTuse[0,:]])

    CAuse = np.concatenate([CAuse.T,CAuse[:,0][:,np.newaxis].T]).T
    BTuse = np.broadcast_to(np.append(BTs-np.diff(BTs)[0]/2,BTs[-1]+np.diff(BTs)[0]/2),CAuse.shape)
    shading = 'flat'
    xticks = [0,90,180,270,360]
    xtickstr = ['0','90','180','270','']

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


vmax = 150

fig = plt.figure(14,figsize = (11, 10))

plt.clf()

plt.subplots_adjust(top=0.95,bottom=0.08,left=0.13, wspace=0.3, hspace=0.4,right=0.95)
for j,dptilt in enumerate(dptilts):

    dPhiNs = dPhiNss[j].copy()
    dPhiSs = dPhiSss[j].copy()

    if doroll:
        dPhiNs = rollfunc(dPhiNs)
        dPhiSs = rollfunc(dPhiSs)

    coeff = (dPhiNs-dPhiSs)/ ( (dPhiNs+dPhiSs) / 2 )

    ax0 = plt.subplot2grid((4, 14), (j,  0), colspan = 4)
    ax1 = plt.subplot2grid((4, 14), (j,  4), colspan = 4)
    ax2 = plt.subplot2grid((4, 14), (j,  8), colspan = 4)

    plt.sca(ax0)
    # ax0.set_aspect('equal')
    if j == 0:
        ax0.set_title(labs[0])
    if j == 2:
        plt.xlabel(r"$\theta_c$ [deg]")
    plt.ylabel("$B_T$ [nT]")
    imn = plt.pcolormesh(CAuse,BTuse,dPhiNs,vmin=vmin,vmax=vmax,cmap=cmap,
                         shading=shading)
    
    plt.sca(ax1)
    # ax1.set_aspect('equal')
    if j == 0:
        ax1.set_title(labs[1])
    if j == 2:
        plt.xlabel(r"$\theta_c$ [deg]")
    _ = [ylab.set_visible(False) for ylab in ax1.yaxis.get_ticklabels()]
    ims = plt.pcolormesh(CAuse,BTuse,dPhiSs,vmin=vmin,vmax=vmax,cmap=cmap,
                         shading=shading)
    
    plt.sca(ax2)
    # ax2.set_aspect('equal')
    if j == 0:
        ax2.set_title(labs[2])
    if j == 2:
        plt.xlabel(r"$\theta_c$ [deg]")
    _ = [ylab.set_visible(False) for ylab in ax2.yaxis.get_ticklabels()]
    imd = plt.pcolormesh(CAuse,BTuse,coeff,vmin=vmin2,vmax=vmax2,cmap=cmap2,
                         shading=shading)
    
    # if j == 0:
    figtxtcoords = fig.transFigure.inverted().transform(ax0.transAxes.transform((0.,0.5)))
    figtxtcoords[0] -= 0.1
    fig.text(*figtxtcoords,seaslabs[j],rotation='vertical',va='center',ha='center',fontsize=18)

    for axl in [ax0,ax1,ax2]:
        axl.set_xticks(xticks,labels=xtickstr)

axc  = plt.subplot2grid((4, 100), (0, 88), rowspan=3, colspan = 3)
axc2 = plt.subplot2grid((4, 100), (0, 96), rowspan=3, colspan = 3)

cb1 = plt.colorbar(imn,cax=axc,extend=extend)
cb2 = plt.colorbar(imd,cax=axc2)
    

# Now master panel
coeffs = [(dPhiNs-dPhiSs)/ ( (dPhiNs+dPhiSs) / 2 ) for dPhiNs,dPhiSs in zip(dPhiNss,dPhiSss)]

avgs = [np.mean(coeff,axis=0) for coeff in coeffs]

avgsphic = [np.mean(coeff,axis=1) for coeff in coeffs]

ax = plt.subplot2grid((4, 14), (3, 0), colspan = 7)
ax2 = plt.subplot2grid((4, 14), (3, 7), colspan = 7)

linestyles = ['-','-','-']
colors = ['black','gray','lightgray']
linewidths = [2,3,4]

zorders = [10,5,0]

## Average over clock angle
plt.sca(ax)
plt.xlabel("$B_T$ [nT]")
plt.ylabel(r'$A_{EM}$')

_ = ax.spines['right'].set_visible(False)
_ = ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
_ = ax.yaxis.set_ticks_position('left')
_ = ax.xaxis.set_ticks_position('bottom')

for j,dptilt in enumerate(dptilts):
    
    idx = j
    avg = avgs[idx]

    ax.plot(BTs,avg,label=seaslabs2[idx],color=colors[idx],
            linestyle=linestyles[idx],
            lw=linewidths[idx],
            zorder=zorders[idx])

ax.legend(ncols=3)


## Average over B_T
plt.sca(ax2)
plt.xlabel(r"$\theta_c$ [deg]")

_ = ax2.spines['right'].set_visible(False)
_ = ax2.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
_ = ax2.yaxis.set_ticks_position('left')
_ = ax2.xaxis.set_ticks_position('bottom')

for j,dptilt in enumerate(dptilts):
    
    idx = j
    avg = avgsphic[idx]

    showie = rollfunc(np.append(avg,avg[0])) if doroll else np.append(avg,avg[0])

    ax2.plot(CAuse[:,0],showie,label=seaslabs[idx],color=colors[idx],linestyle=linestyles[idx],lw=linewidths[idx],
             zorder=zorders[idx])

ax.set_ylim(ax2.get_ylim())
tickleme = lambda i,x: "{:.1f}".format(x) if (((i+1) %2) == 0) else " "
ax.set_yticks(ax2.get_yticks(),labels=[tickleme(ix,x) for ix,x in enumerate(ax2.get_yticks())])

ax2.set_yticks(ax2.get_yticks(),labels=[" "]*len(ax2.get_yticks()))
ax2.set_xticks([0,90,180,270,360])


if savefig:
    tottitl = f"EMWork\nF10.7={f107val}"

    fname = "MASTER2_"+tottitl.replace('$','').replace('_','').replace(' ','').replace('$^\circ$','deg')+'.png'
    fname = fname.replace("\\Psi","ψ").replace("^\\circ","deg").replace("\n","_").replace("{SW}","sw").replace("[","_").replace("]","").replace('CPCP_','')
    print("Saving fig to "+fname)
    plt.savefig(plotdir+fname,dpi=300)
        
        
    
