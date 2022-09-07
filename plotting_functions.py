######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

#Own libs
from .units import *
from . import arepo

######################################################################
#                        PDF/CDF FUNCTIONS                           #
######################################################################

def plot_ndensity_PDFs(avgs, 
                       colors=None, 
                       linestyles=None, 
                       linewidths=None,
                       labels=None,  
                       cumulative=False, 
                       fit_spline=True, 
                       only_spline=True,
                       save=None,
                       **kwargs):

    print("FIESTA >> Plotting PDFs/CDFs of the number density distribution...")

    #Figure properties
    nsols = len(avgs)
    if colors is None:
        cmap = plt.cm.tab10
        colors = cmap(np.arange(nsols)%cmap.N)
    if linestyles is None:
        linestyles = ["-"]*nsols
    if linewidths is None:
        linewidths = [1.5]*nsols
    if labels is None:
        labels = [fr"${i}$" for i in np.arange(1,nsols+1)]

    #Main figure
    fig = plt.figure(figsize=(8,8))
    if "figure" in kwargs:
        plt.setp(fig,**kwargs["figure"])
        
    ax = fig.add_subplot(111)
        
    #Grid
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])
    
    #Axes limits
    ax.set_xlim(-2,6)
    ax.set_ylim(1e-3,1)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])
        
    #Axes scales
    ax.set_yscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.xaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    ax.yaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.yaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    if "xtick_params" in kwargs:
        ax.xaxis.set_tick_params(**kwargs["xtick_params"])
    if "ytick_params" in kwargs:
        ax.yaxis.set_tick_params(**kwargs["ytick_params"])
    if "tick_params" in kwargs:
        ax.tick_params(**kwargs["tick_params"])
     
    #Axes labels
    ax.set_xlabel(r"log(n) [cm$^{-3}$]",fontsize=15)
    if(cumulative):
        ax.set_ylabel(r"Cumulative probability density",fontsize=15)
    else:
        ax.set_ylabel(r"Probability density",fontsize=15)
    if "xlabel" in kwargs:
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(**kwargs["ylabel"])
        
    #Figure title
    ax.set_title("",fontsize=15)
    if "title" in kwargs:
        ax.set_title(**kwargs["title"])

    ############### Plotting start ################

    bars = []
    lines = []
        
    for avg, c, ls, lw, la in zip(avgs, colors, linestyles, linewidths, labels):
        
        ndens = avg.ndensity()
        mass = avg.mass[avg.gas_ids]
        log_ndens = np.log10(ndens, out=np.zeros_like(ndens), where=(ndens>0))
        hist, bin_edges = np.histogram(log_ndens, bins=25, weights=mass, density=True)

        #mass-weighted PDF of the density distribution (see Burkhart, 2018)
        #rho_avg = np.average(AVG.rho,weights=AVG.mass[AVG.gas_ids])
        #s = np.log10(AVG.rho/rho_avg, out=np.zeros_like(AVG.rho), where=(AVG.rho>0))
        #hist, bin_edges = np.histogram(s, bins=25, weights=AVG.mass[AVG.gas_ids], density=True)

        x = bin_edges[:-1]
        y = hist

        if(cumulative):
            y = np.cumsum(y*np.diff(bin_edges))

        #Plotting histogram of PDF
        if(not only_spline):

            width = (max(log_ndens)-min(log_ndens))/len(x)
            bar = ax.bar(x, y, width=width, color=c, linestyle=ls, linewidth=lw, label=la, alpha=0.1)
            bars.append(bar)
        
        #Fitting cubic spline to the whole PDF
        if(fit_spline):

            from scipy.interpolate import PchipInterpolator

            spline = PchipInterpolator(x,y)
            xval = np.linspace(min(x),max(x),1000)
            yval = spline(xval)
            line, = ax.plot(xval, yval, color=c, linestyle=ls, linewidth=lw, label=la)
            lines.append(line)
            
    ax.axvline(x=2.0,color='k',linestyle='--')

    ############### Plotting end ################
        
    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    #Figure legend
    ax.legend(fontsize=15)
    if "legend" in kwargs:
        ax.legend(**kwargs["legend"])
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig, ax, bars, lines
        
######################################################################
#                          SINK FUNCTIONS                            #
######################################################################

def plot_sink_mass_evolutions(base_file_paths,
                              min_nums, 
                              max_nums,
                              colors=None,
                              linestyles=None,
                              linewidths=None,
                              labels=None,
                              save=None,
                              **kwargs):

    print("FIESTA >> Plotting sink mass evolution...")

    #Figure properties
    nsols = len(base_file_paths)
    if colors is None:
        cmap = plt.cm.tab10
        colors = cmap(np.arange(nsols)%cmap.N)
    if linestyles is None:
        linestyles = ["-"]*nsols
    if linewidths is None:
        linewidths = [1.5]*nsols
    if labels is None:
        labels = [fr"${i}$" for i in np.arange(1,nsols+1)]

    #Main figure
    fig = plt.figure(figsize=(8,8))
    if "figure" in kwargs:
        plt.setp(fig,**kwargs["figure"])
        
    ax = fig.add_subplot(111)
        
    #Grid
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])
    
    #Axes limits
    ax.set_xlim(-2,6)
    ax.set_ylim(1e-3,1)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])
        
    #Axes scales
    ax.set_yscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.xaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    ax.yaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.yaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    if "xtick_params" in kwargs:
        ax.xaxis.set_tick_params(**kwargs["xtick_params"])
    if "ytick_params" in kwargs:
        ax.yaxis.set_tick_params(**kwargs["ytick_params"])
    if "tick_params" in kwargs:
        ax.tick_params(**kwargs["tick_params"])
     
    #Axes labels
    ax.set_xlabel(r"Time [Myrs]",fontsize=15)
    ax.set_ylabel(r"Total sink mass  [M$_\odot$]",fontsize=15)
    if "xlabel" in kwargs:
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(**kwargs["ylabel"])
        
    #Figure title
    ax.set_title("",fontsize=15)
    if "title" in kwargs:
        ax.set_title(**kwargs["title"])

    ############### Plotting start ################

    lines = []

    for bfp, minn, maxn, c, ls, lw, la in zip(base_file_paths, min_nums, max_nums, colors, linestyles, linewidths, labels):

        nums = ["{0:03}".format(i) for i in range(minn,maxn)]
        counter = 0
        time_arr = []
        totsinkmass_arr = []

        print("FIESTA >> Started reading {}".format(bfp))
        for num in nums:
            file_path = bfp + num
            try:
                avg = arepo.ArepoVoronoiGrid(file_path)
            except FileNotFoundError:
                break
            time = avg.time*utime/(31536000.0*1.e6) #Myrs
            print(counter,end=",")
            counter+=1
            time_arr.append(time)
            totsinkmass = np.sum(avg.mass[avg.sink_ids])
            totsinkmass_arr.append(totsinkmass)
        print("FIESTA >> Finished reading {}".format(base_file_path))

        #ax.scatter(time_array,totsinkmass_array,color=color,s=3,zorder=1)
        line, = ax.plot(time_arr, totsinkmass_arr, color=c, linestyle=ls, linewidth=lw, label=la)
        lines.append(line)

    #ax.axhline(y=250, color='grey', linewidth=1, linestyle='--', zorder=-1)
    #ax.axhline(y=500, color='grey', linewidth=1, linestyle='--', zorder=-1)
    #ax.axhline(y=750, color='grey', linewidth=1, linestyle='--', zorder=-1)
    #ax.axhline(y=1000, color='grey', linewidth=1, linestyle='--', zorder=-1)
    
    ############### Plotting end ################ 
    
    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    #Figure legend
    if "legend" in kwargs:
        ax.legend(**kwargs["legend"])
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
        ax.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.18))
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig, ax, lines

######################################################################
#                          GENERIC FUNCTIONS                         #
######################################################################

def plot_log_histogram(arrays,
                       bins='auto',
                       colors=None, 
                       linestyles=None, 
                       linewidths=None,
                       labels=None,
                       save=None,
                       **kwargs):

    print("FIESTA >> Plotting histogram...")

    #Figure properties
    nsols = len(arrays)
    if colors is None:
        cmap = plt.cm.tab10
        colors = cmap(np.arange(nsols)%cmap.N)
    if linestyles is None:
        linestyles = ["-"]*nsols
    if linewidths is None:
        linewidths = [1.5]*nsols
    if labels is None:
        labels = [fr"${i}$" for i in np.arange(1,nsols+1)]

    #Figure properties
    nsols = len(arrays)
    if colors is None:
        cmap = plt.cm.tab10
        colors = cmap(np.arange(nsols)%cmap.N)
    if linestyles is None:
        linestyles = ["-"]*nsols
    if linewidths is None:
        linewidths = [1.5]*nsols
    if labels is None:
        labels = [fr"${i}$" for i in np.arange(1,nsols+1)]

    #Main figure
    fig = plt.figure(figsize=(8,8))
    if "figure" in kwargs:
        plt.setp(fig,**kwargs["figure"])
        
    ax = fig.add_subplot(111)
        
    #Grid
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])
    
    #Axes limits
    ax.set_xlim(1e-2,1e+2)
    ax.set_ylim(1e-3,1)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])
        
    #Axes scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.xaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    ax.yaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.yaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    if "xtick_params" in kwargs:
        ax.xaxis.set_tick_params(**kwargs["xtick_params"])
    if "ytick_params" in kwargs:
        ax.yaxis.set_tick_params(**kwargs["ytick_params"])
    if "tick_params" in kwargs:
        ax.tick_params(**kwargs["tick_params"])
     
    #Axes labels
    ax.set_xlabel(r"$x$",fontsize=15)
    ax.set_ylabel(r"$y$",fontsize=15)
    if "xlabel" in kwargs:
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(**kwargs["ylabel"])
        
    #Figure title
    ax.set_title("",fontsize=15)
    if "title" in kwargs:
        ax.set_title(**kwargs["title"])

    ############### Plotting start ################

    hists = []

    arrays_flattened = np.array([item for sublist in arrays for item in sublist])
    bins=np.logspace(np.log10(arrays_flattened.min()), np.log10(arrays_flattened.max()), bins)

    for arr, c, ls, lw, la in zip(arrays, colors, linestyles, linewidths, labels):
        hist = ax.hist(arr, bins=bins, edgecolor=c, label=la, linestyle = ls, linewidth = lw,
                       histtype='step', stacked=True, density=True)
        hists.append(hist)

    ############### Plotting end ################

    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    #Figure legend
    ax.legend(fontsize=15)
    if "legend" in kwargs:
        ax.legend(**kwargs["legend"])
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig, ax, hists

######################################################################
######################################################################

def plot_boxplot(arrays,
                 write_mean=True,
                 write_median=True,
                 write_minmax=True,
                 write_total=True,
                 colors=None, 
                 linestyles=None, 
                 linewidths=None,
                 labels=None,
                 save=None,
                 **kwargs):

    print("FIESTA >> Plotting boxplot...")

    #Figure properties
    nsols = len(arrays)
    if colors is None:
        cmap = plt.cm.tab10
        colors = cmap(np.arange(nsols)%cmap.N)
    if linestyles is None:
        linestyles = ["-"]*nsols
    if linewidths is None:
        linewidths = [1.5]*nsols
    if labels is None:
        labels = [fr"${i}$" for i in np.arange(1,nsols+1)]

    #Main figure
    fig = plt.figure(figsize=(8,8))
    if "figure" in kwargs:
        plt.setp(fig,**kwargs["figure"])
        
    ax = fig.add_subplot(111)
        
    #Grid
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])
    
    #Axes limits
    ax.set_ylim(0.2,len(arrays)+0.5)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])
        
    #Axes scales
    ax.set_xscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.xaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    ax.yaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
    ax.yaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
    if "xtick_params" in kwargs:
        ax.xaxis.set_tick_params(**kwargs["xtick_params"])
    if "ytick_params" in kwargs:
        ax.yaxis.set_tick_params(**kwargs["ytick_params"])
    if "tick_params" in kwargs:
        ax.tick_params(**kwargs["tick_params"])
     
    #Axes labels
    ax.set_xlabel(r"$x$",fontsize=15)
    ax.set_ylabel(r"$y$",fontsize=15)
    if "xlabel" in kwargs:
        ax.set_xlabel(**kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(**kwargs["ylabel"])
        
    #Figure title
    ax.set_title("",fontsize=15)
    if "title" in kwargs:
        ax.set_title(**kwargs["title"])

    ############### Plotting start ################

    #Reversing arrays so they appear in order top to bottom
    arrays=arrays[::-1]
    colors=colors[::-1]
    linestyles=linestyles[::-1]
    linewidths=linewidths[::-1]
    labels=labels[::-1]

    #Note: setting whiskers to very high value to force min/max to be within whiskers range
    boxplot = ax.boxplot(arrays,
                         patch_artist = True, 
                         showfliers=True,
                         showmeans = True, meanline = True,
                         whis=1e18,
                         labels = labels,
                         vert = False)

    for patch, c, ls, lw in zip(boxplot['boxes'], colors, linestyles, linewidths):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_linestyle(ls)
        patch.set_linewidth(lw)

    for whisker in boxplot['whiskers']:
        whisker.set(color ='black', linewidth = 1, linestyle='-')

    n = np.arange(len(arrays))
    lens = np.array([len(array) for array in arrays])

    if(write_mean):
        for (i,mean) in zip(n,boxplot['means']):
            mean.set(color ='black', linewidth = 1, linestyle='--')
            x = mean.get_xdata()[0]
            xround = np.round(x,2)
            ax.annotate(r'${}$'.format(xround), (x, i+1-0.45), fontsize=8,  ha='center', va='center')

    if(write_median):
        for (i,median) in zip(n, boxplot['medians']):
            median.set(color ='black', linewidth = 1, linestyle='-')
            x = median.get_xdata()[0]
            xround = np.round(x,2)
            ax.annotate(r'${}$'.format(xround), (x, i+1-0.45), fontsize=8,  ha='center', va='center')

    if(write_minmax):
        for (i,num) in zip(n,lens):
            #min
            x = boxplot['caps'][(2*i)].get_xdata()[0]
            xround = np.round(x,3)
            ax.annotate(r'${}$'.format(xround), (x, i+1-0.45), fontsize=8,  ha='center', va='center')
            #max
            x = boxplot['caps'][(2*i)+1].get_xdata()[0]
            xround = np.round(x,3)
            ax.annotate(r'${}$'.format(xround), (x, i+1-0.45), fontsize=8,  ha='center', va='center')
    
    if(write_total):
        for (i,num) in zip(n,lens):
            ax.annotate(r'${}$'.format(num), (ax.get_xlim()[1], i+1), fontsize=8,  ha='left', va='center')

    ############### Plotting end ################

    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig, ax, boxplot