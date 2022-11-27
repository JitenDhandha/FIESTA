######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains useful plotting routines for statistical analysis.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
#Numpy
import numpy as np
#Scipy
from scipy.interpolate import PchipInterpolator
#Astropy
from astropy import units as u
#Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Fiesta
from fiesta import units as ufi
from fiesta import arepo
from fiesta import _utils as utils

######################################################################
#                        plot_ndensity_PDFs                          #
######################################################################

def plot_ndensity_PDFs(avgs,
                       ndensity_unit=u.cm**-3,
                       bins=None,
                       colors=None,
                       linestyles=None,
                       linewidths=None,
                       labels=None,
                       cumulative=False,
                       fit_spline=False,
                       only_spline=False,
                       save=None,
                       **kwargs):
    """
    
    Plot the mass-weighted logarithmic probability distribution function (PDF)
    or cumulative distribution function (CDF) histograms of the number density 
    of AREPO Voronoi grids.

    Parameters
    ----------

    avgs : `list` of `~fiesta.arepo.ArepoVoronoiGrid`'s
        The list of AREPO Voronoi grid instances to plot the PDFs/CDFs of.

    ndensity_unit :  `~astropy.units.Unit`, optional
        The unit of number density to use. Default value is ``u.cm**-3``.

    bins: `list`, optional
        The number of bins corresponding to each PDF/CDF. If ``None``
        (default), they are set to ``25``.

    colors: `list`, optional
        A list of *matplotlib*-compatible color strings corresponding
        to each line plotted. If ``None`` (default), colors are picked
        by cycling through *matplotlib* tab10 colour palette.

    linestyles : `list`, optional
        A list of *matplotlib*-compatible linestyle strings corresponding
        to each line plotted. If ``None`` (default), they are set to ``-``.

    linewidths : `list`, optional
        A list of linewidths corresponding to each line plotted. 
        If ``None`` (default), they are set to ``1.5``.

    labels : `list`, optional
        A list of labels corresponding to each line plotted.
        If ``None`` (default), they are numbered ``1, ..., n``.

    cumulative : `bool`, optional
        If ``True``, plots the CDF, else plots the PDF.
        Default value is ``False``.

    fit_spline : `bool`, optional
        If ``True``, fits a spline interpolator through each PDF/CDF
        histogram. Default value is ``False``.

    only_spline : `bool`, optional
        If ``True``, plots only the spline interpolator and not the 
        PDF/CDF histograms. Default value is ``False``.

    save : `str`, optional
        File path to save the plot.
        If ``None`` (default), plot is not saved.

    **kwargs : `dict`, optional
        Additional *matplotlib*-based keyword arguments to control 
        finer details of the plot.

    Returns
    -------

    fig : `~matplotlib.figure.Figure`
        Main `~matplotlib.figure.Figure` instance.

    """

    #Check unit
    utils.check_unit(ndensity_unit, u.cm**-3)

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
    if bins is None:
        bins = [25]*nsols

    #Main figure
    fig = plt.figure(figsize=(8,8))
    if "figure" in kwargs:
        plt.setp(fig,**kwargs["figure"])
        
    ax = fig.add_subplot(111)
        
    #Grid
    if "grid" in kwargs:
        ax.grid(**kwargs["grid"])
        
    #Axes scales
    ax.set_yscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
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
    ax.set_xlabel(r"Number density log($n$) [{}]".format(ndensity_unit.to_string()),fontsize=15)
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
        
    for avg, bin, c, ls, lw, la in zip(avgs, bins, colors, linestyles, linewidths, labels):
        
        ndens = avg.get_ndensity().to_value(ndensity_unit)
        mass = avg.mass[avg.gas_ids].to_value(u.g)
        log_ndens = np.log10(ndens, out=np.zeros_like(ndens), where=(ndens>0))
        hist, bin_edges = np.histogram(log_ndens, bins=bin, weights=mass, density=True)

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

            spline = PchipInterpolator(x,y)
            xval = np.linspace(min(x),max(x),1000)
            yval = spline(xval)
            line, = ax.plot(xval, yval, color=c, linestyle=ls, linewidth=lw, label=la)
            lines.append(line)
            
    ax.axvline(x=2.0,color='k',linestyle='--')

    ############### Plotting end ################

    #Axes limits
    ax.set_xlim(-2,6)
    ax.set_ylim(1e-3,1)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])
        
    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    #Figure legend
    ax.legend(fontsize=15)
    if "legend" in kwargs:
        ax.legend(**kwargs["legend"])
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig
        
######################################################################
#                    plot_sink_mass_evolutions                       #
######################################################################

def plot_sink_mass_evolutions(base_file_paths,
                              min_nums,
                              max_nums,
                              time_unit=u.s,
                              mass_unit=u.g,
                              colors=None,
                              linestyles=None,
                              linewidths=None,
                              labels=None,
                              save=None,
                              **kwargs):

    """
    
    Plot the total sink mass of AREPO Voronoi grids as a function of time.

    Parameters
    ----------

    base_file_paths : `list` of `str`'s
        The list of base file path strings to read :class:`~fiesta.arepo.ArepoVoronoiGrid` 
        from. The format of the AREPO snapshots is usually ``<base_file_path><num>`` where 
        ``<num>`` is an positive integer corresponding to the snapshot number.

    min_nums: `list`
        The list of starting value of ``<num>`` for each AREPO simulation.

    max_nums : `list`
        The list of ending value of ``<num>`` for each AREPO simulation.

    time_unit :  `~astropy.units.Unit`, optional
        The unit of time to use. Default value is ``u.s``.

    mass_unit :  `~astropy.units.Unit`, optional
        The unit of mass to use. Default value is ``u.g``.

    colors: `list`, optional
        A list of *matplotlib*-compatible color strings corresponding
        to each line plotted. If ``None`` (default), colors are picked
        by cycling through *matplotlib* tab10 colour palette.

    linestyles : `list`, optional
        A list of *matplotlib*-compatible linestyle strings corresponding
        to each line plotted. If ``None`` (default), they are set to ``-``.

    linewidths : `list`, optional
        A list of linewidths corresponding to each line plotted. 
        If ``None`` (default), they are set to ``1.5``.

    labels : `list`, optional
        A list of labels corresponding to each line plotted.
        If ``None`` (default), they are numbered ``1, ..., n``.

    save : `str`, optional
        File path to save the plot.
        If ``None`` (default), plot is not saved.

    **kwargs : `dict`, optional
        Additional *matplotlib*-based keyword arguments to control 
        finer details of the plot.

    Returns
    -------

    fig : `~matplotlib.figure.Figure`
        Main `~matplotlib.figure.Figure` instance.

    """

    #Check unit
    utils.check_unit(time_unit, u.s)
    utils.check_unit(mass_unit, u.g)

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
        
    #Axes scales
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
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
    ax.set_xlabel(r"Time [{}] ".format(time_unit.to_string()),fontsize=15)
    ax.set_ylabel(r"Total sink mass [{}] ".format(mass_unit.to_string()),fontsize=15)
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

        for num in nums:
            file_path = bfp + num
            avg = arepo.ArepoVoronoiGrid(file_path)
            time = avg.time.to_value(time_unit)
            time_arr.append(time)
            totsinkmass = np.sum(avg.mass[avg.sink_ids].to_value(mass_unit))
            totsinkmass_arr.append(totsinkmass)

        #ax.scatter(time_array,totsinkmass_array,color=color,s=3,zorder=1)
        line, = ax.plot(time_arr, totsinkmass_arr, color=c, linestyle=ls, linewidth=lw, label=la)
        lines.append(line)
    
    ############### Plotting end ################ 

    #Axes limits
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])
    
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

    return fig

######################################################################
#                        plot_log_histogram                          #
######################################################################

def plot_log_histogram(arrays,
                       bins='auto',
                       colors=None,
                       linestyles=None,
                       linewidths=None,
                       labels=None,
                       save=None,
                       **kwargs):

    """
    
    A general function to plot histograms of data on a log scale.

    Parameters
    ----------

    arrays : `list` of `list`'s
        The list of data arrays to generate histograms of.

    bins: `int` or `sequence` or `str`
        See `~matplotlib.axes.Axes.hist` documentation for detail.
        Default value is ``auto``.

    colors: `list`, optional
        A list of *matplotlib*-compatible color strings corresponding
        to each line plotted. If ``None`` (default), colors are picked
        by cycling through *matplotlib* tab10 colour palette.

    linestyles : `list`, optional
        A list of *matplotlib*-compatible linestyle strings corresponding
        to each line plotted. If ``None`` (default), they are set to ``-``.

    linewidths : `list`, optional
        A list of linewidths corresponding to each line plotted. 
        If ``None`` (default), they are set to ``1.5``.

    labels : `list`, optional
        A list of labels corresponding to each histogram plotted.
        If ``None`` (default), they are numbered ``1, ..., n``.

    save : `str`, optional
        File path to save the plot.
        If ``None`` (default), plot is not saved.

    **kwargs : `dict`, optional
        Additional *matplotlib*-based keyword arguments to control 
        finer details of the plot.

    Returns
    -------

    fig : `~matplotlib.figure.Figure`
        Main `~matplotlib.figure.Figure` instance.

    """

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
        
    #Axes scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
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

    #Axes limits
    ax.set_xlim(1e-2,1e+2)
    ax.set_ylim(1e-3,1)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])

    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    #Figure legend
    ax.legend(fontsize=15)
    if "legend" in kwargs:
        ax.legend(**kwargs["legend"])
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig

######################################################################
#                           plot_boxplot                             #
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

    """
    
    A general function to generate boxplot (or box and whiskers plot) of data.

    Parameters
    ----------

    arrays : `list` of `list`'s
        The list of data arrays to generate boxplot of.

    write_mean: bool
        If ``True`` (default), the mean value is mentioned below each mean line.

    write_median: `bool`
         If ``True`` (default), the median value is mentioned below each median line.

    write_minmax : `bool`
        If ``True`` (default), the minimum and maximum values are mentioned below
        the end whiskers.

    write_total : `bool`
        If ``True`` (default), the length of data arrays is mentioned at the right side
        of the plot, outside the axes.

    colors: `list`, optional
        A list of *matplotlib*-compatible color strings corresponding
        to each line plotted. If ``None`` (default), colors are picked
        by cycling through *matplotlib* tab10 colour palette.

    linestyles : `list`, optional
        A list of *matplotlib*-compatible linestyle strings corresponding
        to each line plotted. If ``None`` (default), they are set to ``-``.

    linewidths : `list`, optional
        A list of linewidths corresponding to each line plotted. 
        If ``None`` (default), they are set to ``1.5``.

    labels : list, optional
        A list of labels corresponding to each box plotted.
        If ``None`` (default), they are numbered ``1, ..., n``.

    save : str, optional
        The name of the file to save the plot as. If ``None`` (default), plot is
        not saved.

    **kwargs : dict, optional 
        Additional *matplotlib*-based keyword arguments to control 
        finer details of the plot.

    Returns
    -------

    fig : `~matplotlib.figure.Figure`
        Main `~matplotlib.figure.Figure` instance.

    """

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
        
    #Axes scales
    ax.set_xscale('log')
    if "xscale" in kwargs:
        ax.set_xscale(**kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(**kwargs["yscale"])    
        
    #Axes ticks
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

    #Axes limits
    ax.set_ylim(0.2,len(arrays)+0.5)
    if "xlim" in kwargs:
        ax.set_xlim(**kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(**kwargs["ylim"])

    #Text
    if "text" in kwargs:
        ax.text(**kwargs["text"],transform=ax.transAxes)
        
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return fig