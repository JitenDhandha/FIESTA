######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains classes to interact with AREPO data, 
both structured and unstructured meshes, such as reading/writing
routines and plotting algoritms. Currently only supports 3D data.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import copy
#Numpy
import numpy as np
#Scipy
from scipy import interpolate as interp
#Astropy
from astropy.io import fits
from astropy import units as u
#Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#FIESTA
from fiesta import units as ufi
from fiesta import properties as prop
from fiesta import _areporeadwrite
from fiesta import _utils as utils

######################################################################
#                       CLASS: ArepoVoronoiGrid                      #
######################################################################

class ArepoVoronoiGrid:

    """
    Class to deal with AREPO snapshosts in the form of
    native unstructured meshes.

    Attributes
    ----------
        
    data : `dict`
        Dictionary containing all the data in the snapshot 
        (partly redundant in lieu of other variables).

    header : `dict`
        Dictionary containing all the information in the header of 
        the snapshot (partly redundant in lieu of other variables).

    size : `~astropy.units.Quantity`
        Size of simulation box along one axis where ``size**3`` 
        encompasses the whole 3D simulation.

    time : `~astropy.units.Quantity`
        Time corresponding to the snapshot.

    ntot : `int`
        Total number of particles.

    ngas : `int`
        Number of gas particles.

    nsink : `int`
        Number of sink particles.

    gas_ids : `~numpy.ndarray`
        1D array of indices corresponding to gas particles.

    sink_ids : `~numpy.ndarray`
        1D array of indices corresponding to sink particles.

    pos : `~astropy.units.Quantity`
        1D array of position of all particles.

    vel : `~astropy.units.Quantity`
        1D array of velocities of all particles.

    mass : `~astropy.units.Quantity`
        1D array of mass of all particles.

    chem : `~astropy.units.Quantity`
        1D array of fractional chemical abundances of all particles.

    rho : `~astropy.units.Quantity`
        1D array of mass density of gas particles.

    utherm : `~astropy.units.Quantity`
        1D array of thermal energy per unit mass of all particles.

    Parameters
    ----------

    file_path : `str`
        File path of the AREPO snapshot.

    io_flags : `dict`, optional
        Dictionary of input/output flags for reading the file. If ``None`` (default),
        it is set to: ``io_flags = {'mc_tracer' : False, 'time_steps' : False, 
        'sgchem' : True, 'variable_metallicity': False, 'sgchem_NL99' : False}``.

    verbose : `bool`, optional
        If ``True``, prints output during file reading. Default value is ``False``.
    
    """
    
    def __init__(self, file_path, io_flags=None, verbose=False):


        if(verbose):
            print(utils._prestring() + "Loading AREPO Voronoi grid from \"{}\" ...".format(file_path))

        #Setting IO flags
        if io_flags is None:
            io_flags = {'mc_tracer'           : False,
                        'time_steps'          : False,
                        'sgchem'              : True,
                        'variable_metallicity': False,
                        'sgchem_NL99'         : False}

        self.data, self.header = _areporeadwrite.read_snapshot(file_path, io_flags)
        self.size = float(self.header['boxsize'][0]) << ufi.AREPO_LENGTH
        self.time = float(self.header['time'][0]) << ufi.AREPO_TIME
        self.ntot = int(sum(self.header['num_particles']))
        self.ngas = int(self.header['num_particles'][0])
        self.nsink = int(self.header['num_particles'][5])
        self.gas_ids = np.arange(0,self.ngas)
        self.sink_ids = np.arange(self.ntot-self.nsink,self.ntot)
        self.pos = self.data['pos'] << ufi.AREPO_LENGTH
        self.vel = self.data['vel'] << ufi.AREPO_VELOCITY
        self.mass = self.data['mass'] << ufi.AREPO_MASS
        self.chem = self.data['chem'] << u.dimensionless_unscaled
        self.rho = self.data['rho'] << ufi.AREPO_DENSITY
        self.utherm = self.data['u_therm'] << ufi.AREPO_ENERGY/ufi.AREPO_MASS
        
        if(verbose):
            print(utils._prestring() + "Size: {}".format(self.size))
            print(utils._prestring() + "# of particles: {}".format(self.ntot))
            print(utils._prestring() + "# of gas particles: {}".format(self.ngas))
            print(utils._prestring() + "# of sink particles: {}".format(self.nsink))
            if(self.header['flag_doubleprecision']):
                  print(utils._prestring() + "Precision: double")
            else:
                  print(utils._prestring() + "Precision: float")
                  
        if(verbose):
            print(utils._prestring() + "Completed loading AREPO Voronoi grid from \"{}\"".format(file_path))

        #Additional internal variables
        self._tree = None

    ######################################################################
    #                             GETTERS                                #
    ######################################################################

    def get_size(self, unit=u.cm):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.size`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.cm``.

        """
        utils.check_unit(unit, u.cm)
        return self.size.to(unit)

    def get_time(self, unit=u.s):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.time`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.s``.

        """
        utils.check_unit(unit, u.s)
        return self.time.to(unit)

    def get_ntot(self):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.ntot`.

        """
        return self.ntot

    def get_ngas(self):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.ngas`.

        """
        return self.ngas

    def get_nsink(self):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.nsink`.

        """
        return self.nsink

    def get_gas_ids(self):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.gas_ids`.

        """
        return self.gas_ids

    def get_sink_ids(self):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.sink_ids`.

        """
        return self.sink_ids

    def get_pos(self, unit=u.cm):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.pos`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.cm``.

        """
        utils.check_unit(unit, u.cm)
        return self.pos.to(unit)

    def get_vel(self, unit=u.cm/u.s):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.vel`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.cm/u.s``.

        """
        utils.check_unit(unit, u.cm/u.s)
        return self.vel.to(unit)

    def get_mass(self, unit=u.g):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.mass`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.g``.

        """
        utils.check_unit(unit, u.g)
        return self.mass.to(unit)

    def get_chem(self):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.chem`.

        """
        return self.chem

    def get_rho(self, unit=u.g/u.cm**3):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.rho`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.g/u.cm**3``.

        """
        utils.check_unit(unit, u.g/u.cm**3)
        return self.rho.to(unit)

    def get_utherm(self, unit=u.erg/u.g):
        """

        Returns `~fiesta.arepo.ArepoVoronoiGrid.utherm`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.erg/u.g``.

        """
        utils.check_unit(unit, u.erg/u.g)
        return self.utherm.to(unit)

    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
     
    def write_file(self, file_path, io_flags=None, verbose=False):
        """
        
        Write out an AREPO snapshot file.

        Parameters
        ----------

        file_path : `str`, optional
            File path of the AREPO snapshot.

        io_flags : `dict`, optional
            Dictionary of input/output flags for writing the file. If ``None`` (default),
            it is set to: ``io_flags = {'mc_tracer' : False, 'time_steps' : False, 
            'sgchem' : True, 'variable_metallicity': False, 'sgchem_NL99' : False}``.

        verbose : `bool`, optional
            If ``True``, prints output during file writing. Default value is ``False``.

        """

        #Setting IO flags
        if io_flags is None:
            io_flags = {'mc_tracer'           : False,
                        'time_steps'          : False,
                        'sgchem'              : True,
                        'variable_metallicity': False,
                        'sgchem_NL99'         : False}

        if(verbose):
            print(utils._prestring() + "Writing AREPO Voronoi grid to \"{}\" ...".format(file_path))

        _areporeadwrite.write_snapshot(file_path, self.header, self.data, io_flags)

        if(verbose):
            print(utils._prestring() + "Completed writing AREPO Voronoi grid to \"{}\"".format(file_path))

    ######################################################################
    #                     USEFUL PHYSICAL PROPERTIES                     #
    ######################################################################

    def get_ndensity(self, unit=u.cm**-3):
        """

        Returns the number density of gas particles.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.cm**-3``.

        """

        ndensity = prop.calc_number_density(self.rho)
        utils.check_unit(unit, u.cm**-3)
        return ndensity.to(unit)

    def get_temperature(self, unit=u.K):
        """

        Returns the temperature of all particles.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.K``.

        """
        ndensity = self.get_ndensity()
        _, _, _, _, nTOT = prop.calc_chem_density(self.chem, ndensity)
        temperature = prop.calc_temperature(self.rho, self.utherm, nTOT)
        utils.check_unit(unit, u.K)
        return temperature.to(unit)
                
    ######################################################################
    #                       PLOTTING FUNCTIONS                           #
    ######################################################################
        
    def plot_projection(self, 
                        projection='z',
                        length_unit=u.cm,
                        log=False,
                        bins=1000,
                        bounds=None,
                        filaments=None,
                        sinks=None,
                        cmap='magma',
                        save=None,
                        **kwargs):
        
        """
        
        Plot mass-weighted 2d histogram of the snapshot projected along an axis.

        Parameters
        ----------
        
        projection : `str`, optional
            Axis of projection ``x``, ``y`` or ``z`` (default).

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use. Default value is ``u.cm``.

        log: `bool`, optional
            If ``True``, histogram is normalized on a logarithmic scale.
            If ``False`` (default), histogram is normalized on a linear scale.

        bins : `int`, optional
            Number of bins of the 2D histogram. 
            See `~matplotlib.axes.Axes.hist2d` documentation for more detail. 
            Default value is ``1000``.

        bounds : `~astropy.units.Quantity`, optional
            Bounds to restrict the area of the projection to bin, in the format
            ``[xmin, xmax, ymin, ymax]``. If not a `~astropy.units.Quantity` object, the
            array is assumed to be in ``length_unit``.

        filaments: `list` of `~fiesta.disperse.Filament`'s, optional
            1D array of `~fiesta.disperse.Filament` objects to plot with the projection.
            
        sinks : `list` of `list`'s, optional
            List of sink positions to plot.

        cmap : `str` or `~matplotlib.colors.Colormap`, optional
            Colormap of the histogram. Default value is ``'magma'``.

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

        #Shedding units for ease of use
        utils.check_unit(length_unit, u.cm)
        pos = self.pos.to_value(length_unit)
        weights = self.mass.value #units don't matter since they cancel in weighting

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)
        
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
        if(projection=='x' or projection=='X'):
            ax.set_xlabel(r"$y$ [{}]".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$z$ [{}]".format(length_unit.to_string()),fontsize=15)
        elif(projection=='y' or projection=='Y'):
            ax.set_xlabel(r"$x$ [{}]".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$z$ [{}]".format(length_unit.to_string()),fontsize=15)
        elif(projection=='z' or projection=='Z'):
            ax.set_xlabel(r"$x$ [{}]".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$y$ [{}]".format(length_unit.to_string()),fontsize=15)
        else:
            raise ValueError(utils._prestring() + "Invalid projection.")
        if "xlabel" in kwargs:
            ax.set_xlabel(**kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(**kwargs["ylabel"])
            
        #Figure title
        ax.set_title("",fontsize=15)
        if "title" in kwargs:
            ax.set_title(**kwargs["title"])

        ############### Plotting start ################

        #Colors!
        cmap = copy.copy(mpl.cm.get_cmap(cmap))
        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()

        if(projection=='x' or projection=='X'):
            (xaxis, yaxis) = (1,2)
        elif(projection=='y' or projection=='Y'):
            (xaxis, yaxis) = (0,2)
        elif(projection=='z' or projection=='Z'):
            (xaxis, yaxis) = (0,1)
        x = pos[:,xaxis]
        y = pos[:,yaxis]

        if bounds is not None:
            bounds = u.Quantity(bounds,unit=length_unit).value
            xmin, xmax, ymin, ymax = bounds
            BOUNDS_MASK = ((x > xmin) &
                           (x < xmax) &
                           (y > ymin) &
                           (y < ymax))
            x = x[BOUNDS_MASK]
            y = y[BOUNDS_MASK]
            weights = weights[BOUNDS_MASK]

        hist2d = ax.hist2d(x, y, cmap=cmap, norm=norm, bins=bins, weights=weights)

        #Filament plot
        if filaments is not None:
            for fil in filaments:
                if(fil.scale != "physical"):
                    raise ValueError(utils._prestring() + "Filament must be in \"physical\" units.")
                else:
                    #Shedding units for ease of use
                    x = fil.samps[:,xaxis].to_value(length_unit)
                    y = fil.samps[:,yaxis].to_value(length_unit)
                    ax.scatter(x[0], y[0], c='red', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], c='red', s=2, zorder=2)
                    ax.plot(x, y, linewidth=1, c='gold')
                    
        #Sink plot
        if sinks is not None:
                #Shedding units for ease of use
                x = sinks[:,xaxis].to_value(length_unit)
                y = sinks[:,yaxis].to_value(length_unit)
                ax.scatter(x, y, s=2, c='red', zorder=2)

        ############### Plotting end ################

        #Axes limits
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
                  
######################################################################
#                        CLASS: ArepoCubicGrid                       #
######################################################################
                  
class ArepoCubicGrid:
    
    """
    Class to deal with AREPO cubic/cuboidal grids that have been 
    ray-casted from snapshots.

    Attributes
    ----------

    density : `~astropy.units.Quantity`
        3D array containing density at each grid point in AREPO 
        ``umass/ulength**3`` units.

    nx : `~astropy.units.Quantity`
        Number of grid points along :math:`x`-axis of the grid
        in units of ``u.pix``.

    ny : `~astropy.units.Quantity`
        Number of grid points along :math:`y`-axis of the grid
        in units of ``u.pix``.

    nz : `~astropy.units.Quantity`
        Number of grid points along :math:`z`-axis of the grid
        in units of ``u.pix``.

    scale : `str`
        Either ``pixel`` if the grid is not scaled to the source
        AREPO snapshot, else ``physical``.

    xmin : `~astropy.units.Quantity`
        Minimum value of the :math:`x`-axis of the grid, in either ``pixel``
        or ``physical`` units.

    xmin : `~astropy.units.Quantity`
        Maximum value of the :math:`x`-axis of the grid, in either ``pixel``
        or ``physical`` units.

    ymin : `~astropy.units.Quantity`
        Minimum value of the :math:`y`-axis of the grid, in either ``pixel``
        or ``physical`` units.

    ymax : `~astropy.units.Quantity`
        Maximum value of the :math:`y`-axis of the grid, in either ``pixel``
        or ``physical`` units.

    zmin : `~astropy.units.Quantity`
        Minimum value of the :math:`z`-axis of the grid, in either ``pixel``
        or ``physical`` units.

    zmax : `~astropy.units.Quantity`
        Maximum value of the :math:`z`-axis of the grid, in either ``pixel``
        or ``physical`` units.

    xlength : `~astropy.units.Quantity`
        Length of one voxel along :math:`x`-axis, in either ``pixel``
        or ``physical`` units.

    ylength : `~astropy.units.Quantity`
        Length of one voxel along :math:`y`-axis, in either ``pixel``
        or ``physical`` units.

    zlength : `~astropy.units.Quantity`
        Length of one voxel along :math:`z`-axis, in either ``pixel``
        or ``physical`` units.

    Parameters
    ----------

    file_path : `str`
        File path of the AREPO grid.

    AREPO_bounds : `list` or `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The bounds of the source AREPO snapshot corresponding
        to the grid, in the format ``[xmin, xmax, ymin, ymax,
        zmin, zmax]``. If ``None`` (default), then
        ``scale='pixel'``, else ``scale='physical'``. If not an
        `~astropy.units.Quantity`, assumed to be in `~fiesta.units.AREPO_LENGTH`
        units.

    verbose : bool, optional
        If ``True``, prints output during file reading. Default value is
        ``False``.
    
    """

    def __init__(self, file_path, AREPO_bounds=None, verbose=False):
        
        
        if(verbose):
            print(utils._prestring() + "Loading AREPO Cubic grid from \"{}\" ...".format(file_path))
        
        #Read 3D data
        self.density = _areporeadwrite.read_grid(file_path) << ufi.AREPO_DENSITY
        #Save dimensions
        self.nx, self.ny, self.nz = u.Quantity(self.density.shape, copy=False, dtype=int)

        #Setting default scale
        self.scale = "pixel"
        bounds = u.Quantity([0,self.nx.value,0,self.ny.value,0,self.nz.value], dtype=int, unit=u.pix)
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = bounds
        self.xlength = (self.xmax - self.xmin)/(self.nx)
        self.ylength = (self.ymax - self.ymin)/(self.ny)
        self.zlength = (self.zmax - self.zmin)/(self.nz)

        if(verbose):
            print(utils._prestring() + "Completed loading AREPO Cubic grid from \"{}\"".format(file_path))

        if AREPO_bounds is not None:
            self.set_scale(AREPO_bounds)

    ######################################################################
    #                             GETTERS                                #
    ######################################################################

    def get_density(self, unit=u.g/u.cm**3):
        """

        Returns `~fiesta.arepo.ArepoCubicGrid.density`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.g/u.cm**3``.

        """
        utils.check_unit(unit, u.g/u.cm**3)
        return self.density.to(unit)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################

    def write_to_FITS(self, FITS_file_path, density_units=ufi.AREPO_DENSITY, verbose=False):
        """
        
        Write out the AREPO grid into a .FITS file.

        Parameters
        ----------

        FITS_file_path : `str`
            FITS file path to write the grid into.

        verbose : `bool`, optional
            If ``True``, prints output during file writing.
            Default value is ``False``.

        """
        
        if(verbose):
            print(utils._prestring() + "Writing AREPO Cubic grid to FITS at \"{}\" ...".format(FITS_file_path))
            
        hdu = fits.PrimaryHDU(data=self.density.to(density_units).value)
        hdu.writeto(FITS_file_path, overwrite=True)

        if(verbose):
            print(utils._prestring() + "Completed writing AREPO Cubic grid to FITS at \"{}\".".format(FITS_file_path))
            
    def write_mask_to_FITS(self, mask_FITS_file_path, mask_bounds, verbose=False):
        """
        
        Write a mask .FITS file. Useful for masking certain regions before
        feeding into DisPerSE.

        Parameters
        ----------

        mask_FITS_file_path : `str`
            FITS file path to write the masked grid into.

        mask_bounds : `list` of `int`'s
            The grid indices *outside* which everything is masked, in
            the format ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        verbose : bool, optional
            If ``True``, prints output during file writing.
            Default value is ``False``.

        """

        if(verbose):
            print(utils._prestring() + "Writing mask to FITS at \"{}\" ...".format(mask_FITS_file_path))
            
        mask_data = np.full((self.nx.value,self.ny.value,self.nz.value), 1, dtype=int)
        xmin, xmax, ymin, ymax, zmin, zmax = mask_bounds
        mask_data[xmin:xmax,ymin:ymax,zmin:zmax] = 0
        hdu = fits.PrimaryHDU(data=mask_data)
        hdu.writeto(mask_FITS_file_path, overwrite=True)

        if(verbose):
            print(utils._prestring() + "Completed writing mask to FITS at \"{}\".".format(mask_FITS_file_path))

    ######################################################################
    #                         PHYSICAL PROPERTIES                        #
    ######################################################################

    def set_scale(self, AREPO_bounds):
        """
        
        Set the physical scale of the grid by specifying the bounds 
        of the source AREPO snapshot that the grid corresponds to.

        Parameters
        ----------

        AREPO_bounds : `list` or `~numpy.ndarray` or `~astropy.units.Quantity`, optional
            The bounds of the source AREPO snapshot corresponding
            to the grid, in the format ``[xmin, xmax, ymin, ymax,
            zmin, zmax]``. If ``None`` (default), then
            ``scale='pixel'``, else ``scale='physical'``. If not an
            `~astropy.units.Quantity`, assumed to be in `~fiesta.units.AREPO_LENGTH`
            units.

        verbose : bool, optional
            If ``True``, prints output during file reading. Default value is
            ``False``.

        """

        self.scale = "physical"
        bounds = u.Quantity(AREPO_bounds, unit=ufi.AREPO_LENGTH)
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = bounds
        self.xlength = (self.xmax - self.xmin)/(self.nx)
        self.ylength = (self.ymax - self.ymin)/(self.ny)
        self.zlength = (self.zmax - self.zmin)/(self.nz)

    def get_ndensity(self, unit=u.cm**-3):
        """

        Returns the number density of the grid.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.cm**-3``.

        """

        ndensity = prop.calc_number_density(self.density)
        utils.check_unit(unit, u.cm**-3)
        return ndensity.to(unit)

    
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
    
    def plot_grid(self,
                  length_unit=None,
                  ndensity_unit=u.cm**-3,
                  log=False,
                  mask_below=None,
                  filaments=None,
                  sinks=None,
                  cmap='magma',
                  save=None,
                  **kwargs):

        """
        
        Plot the 3D number density grid. Recommended use is with
        ``mask_below`` parameter so that not all grid points are 
        plotted. Can also plot filaments or AREPO sinks together!

        Parameters
        ----------

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use. If ``None`` (default),
            takes the value ``u.pix`` if ``scale='pixel'``, 
            else ``u.cm`` if ``scale='physical'``.

        ndensity_unit : `~astropy.units.Unit`, optional
            The unit of number density to use. 
            Default value is ``u.cm**-3``.
        
        log: `bool`, optional
            If ``True``, the number density is plotted on a log-scale.
            If ``False`` (default), it is plotted on a linear scale.

        mask_below : `float` or `~astropy.units.Quantity`, optional
            The number density below which all grid points are omitted from the plot.
            This ensures only certain dense regions are plotted.
            If `float`, then assumed to be in ``ndensity_unit`` units.

        filaments : list of `~fiesta.disperse.Filament`'s
            List of `~fiesta.disperse.Filament`'s to plot along with the grid. 
            Requires the filaments to have the same ``scale`` as the grid.

        sinks : `list` of `list`'s, optional
            List of sink positions to plot. Requires the grid to have ``scale='physical'``.

        cmap : `str` or `~matplotlib.colors.Colormap`, optional
            Colormap of the plot. Default is ``'magma'``.

        save : `str`, optional
            File path to save the plot.
            If ``None`` (default), plot is not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : `~matplotlib.figure.Figure`
            Main `~matplotlib.figure.Figure` instance.

        """

        #Shedding units for ease of use
        utils.check_unit(ndensity_unit, u.cm**-3)
        if(self.scale=="physical"):
            internal_unit = u.cm
        elif(self.scale=="pixel"):
            internal_unit = u.pix
        if length_unit is None:
            length_unit = internal_unit
        utils.check_unit(length_unit, internal_unit)
        data = self.get_ndensity().to_value(ndensity_unit)
        xmin = self.xmin.to_value(length_unit)
        xmax = self.xmax.to_value(length_unit)
        ymin = self.ymin.to_value(length_unit)
        ymax = self.ymax.to_value(length_unit)
        zmin = self.zmin.to_value(length_unit)
        zmax = self.zmax.to_value(length_unit)
        nx = self.nx.value
        ny = self.ny.value
        nz = self.nz.value

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111,projection='3d')
        
        #Axes ticks
        ax.xaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
        ax.xaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
        ax.yaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
        ax.yaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
        ax.zaxis.set_tick_params(which='major', width=1, length=5, labelsize=15)
        ax.zaxis.set_tick_params(which='minor', width=1, length=2.5, labelsize=10)
        if "xtick_params" in kwargs:
            ax.xaxis.set_tick_params(**kwargs["xtick_params"])
        if "ytick_params" in kwargs:
            ax.yaxis.set_tick_params(**kwargs["ytick_params"])
        if "ztick_params" in kwargs:
            ax.zaxis.set_tick_params(**kwargs["ytick_params"])

        #Axes labels
        ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
        ax.set_ylabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
        ax.set_zlabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
        if "xlabel" in kwargs:
            ax.set_xlabel(**kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(**kwargs["ylabel"])
        if "zlabel" in kwargs:
            ax.set_zlabel(**kwargs["zlabel"])
            
        #Figure title
        ax.set_title("",fontsize=15)
        if "title" in kwargs:
            ax.set_title(**kwargs["title"])

        ############### Plotting start ################

        #Colours!
        #(1 = 3D scatter, 2 = 2D contour projections)
        cmap1 = copy.copy(mpl.cm.get_cmap(cmap))
        cmap2 = mpl.cm.get_cmap('gray')
        if(log):
            cmap1.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()
        alpha1 = 0.1
        alpha2 = 0.4

        #Masking below a certain threshold
        if mask_below is not None:
            mask_below = u.Quantity(mask_below,unit=ndensity_unit).value
            data = np.ma.masked_array(data,mask=data<mask_below)

        #Checking if empty
        if mask_below is not None:
            if data.count() == 0:
                raise RuntimeError(utils._prestring() + "Nothing below the masking threshold!")

        xr = np.linspace(xmin, xmax, num=nx, endpoint=True)
        yr = np.linspace(ymin, ymax, num=ny, endpoint=True)
        zr = np.linspace(zmin, zmax, num=nz, endpoint=True)

        #Note: Matplotlib and Numpy define axis 0 and 1 OPPOSITE!!!
        #Hence, scatter plot requires np.swapaxes whereas contourf plots
        #are transposed.

        #3D PLOT
        X, Y, Z = np.meshgrid(xr,yr,zr)
        plot_data = np.swapaxes(data,0,1).flatten() #See above for swapaxes.
        scatter = ax.scatter(X, Y, Z, c=plot_data, cmap=cmap1, alpha=alpha1, norm=norm, s=1)

        #CONTOUR PLOTS PROJECTIONS
        #x projection
        y, z = np.meshgrid(yr,zr)
        contour_x = np.sum(data,axis=0).T #See above for transpose.
        ax.contourf(contour_x, y, z, zdir='x', levels=20, offset=xmin, cmap=cmap2, norm=norm, alpha=alpha2)
        
        #y projection
        x, z = np.meshgrid(xr,zr)
        contour_y = np.sum(data,axis=1).T #See above for transpose.
        ax.contourf(x, contour_y, z, zdir='y', levels=20, offset=ymax, cmap=cmap2, norm=norm, alpha=alpha2)
        
        #z projection
        x, y = np.meshgrid(xr,yr)
        contour_z = np.sum(data,axis=2).T #See above for transpose.
        ax.contourf(x, y, contour_z, zdir='z', levels=20, offset=zmin, cmap=cmap2, norm=norm, alpha=alpha2)

        #FILAMENT PLOT
        if filaments is not None:
            for fil in filaments:
                if(fil.scale != self.scale):
                    raise ValueError(utils._prestring() + "Filament and ArepoCubicGrid scales don't match!")
                else:
                    #Shedding units for ease of use
                    x = fil.samps[:,0].to_value(length_unit)
                    y = fil.samps[:,1].to_value(length_unit)
                    z = fil.samps[:,2].to_value(length_unit)
                    ax.scatter(x[0], y[0], z[0], c='red', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], z[-1], c='red', s=2, zorder=2)
                    ax.plot(x, y, z, linewidth=1, c='gold')

        #SINK PLOT
        if sinks is not None:
            if(self.scale != "physical"):
                raise ValueError(utils._prestring() + "ArepoCubicGrid needs to be in a physical scale to plot the sinks!")
            else:
                #Shedding units for ease of use
                x = sinks[:,0].to_value(length_unit)
                y = sinks[:,1].to_value(length_unit)
                z = sinks[:,2].to_value(length_unit)
                ax.scatter(x, y, z, s=2, c='red', zorder=2)

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.1)
        cbar_label = r'Number density $n$ [{}]'.format(ndensity_unit)
        cbar.set_label(cbar_label, size=15, color='black')
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.outline.set_edgecolor('black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        ############### Plotting end ################

        #Axes limits
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_zlim(zmin,zmax)
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])
        if "zlim" in kwargs:
            ax.set_zlim(**kwargs["zlim"])

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig
        
    ######################################################################
    
    def plot_slice(self, 
                   x=None, 
                   y=None, 
                   z=None,
                   length_unit=None,
                   ndensity_unit=u.cm**-3,
                   log=False,
                   mask_below=None,
                   cmap='magma',
                   save=None,
                   **kwargs):

        """
        
        Plot a slice of 3D number density grid.

        Parameters
        ----------

        x : `int`, optional
            Integer corresponding to :math:`x`-slice of the grid.
            One of ``x``, ``y`` or ``z`` is required.

        y : `int`, optional
            Integer corresponding to :math:`y`-slice of the grid.
            One of ``x``, ``y`` or ``z`` is required.

        z : `int`, optional
            Integer corresponding to :math:`z`-slice of the grid.
            One of ``x``, ``y`` or ``z`` is required.

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use. If ``None`` (default),
            takes the value ``u.pix`` if ``scale='pixel'``, 
            else ``u.cm`` if ``scale='physical'``.

        ndensity_unit : `~astropy.units.Unit`, optional
            The unit of number density to use. 
            Default value is ``u.cm**-3``.
        
        log: `bool`, optional
            If ``True``, the number density is plotted on a log-scale.
            If ``False`` (default), it is plotted on a linear scale.

        mask_below : `float` or `~astropy.units.Quantity`, optional
            The number density below which all grid points are omitted from the plot.
            This ensures only certain dense regions are plotted.
            If `float`, then assumed to be in ``ndensity_unit`` units.

        cmap : `str` or `~matplotlib.colors.Colormap`, optional
            Colormap of the plot. Default is ``'magma'``.

        save : `str`, optional
            File path to save the plot.
            If ``None`` (default), plot is not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : `~matplotlib.figure.Figure`
            Main `~matplotlib.figure.Figure` instance.

        """

        #Shedding units for ease of use
        utils.check_unit(ndensity_unit, u.cm**-3)
        if(self.scale=="physical"):
            internal_unit = u.cm
        elif(self.scale=="pixel"):
            internal_unit = u.pix
        if length_unit is None:
            length_unit = internal_unit
        utils.check_unit(length_unit, internal_unit)
        data = self.get_ndensity().to_value(ndensity_unit)
        xmin = self.xmin.to_value(length_unit)
        xmax = self.xmax.to_value(length_unit)
        ymin = self.ymin.to_value(length_unit)
        ymax = self.ymax.to_value(length_unit)
        zmin = self.zmin.to_value(length_unit)
        zmax = self.zmax.to_value(length_unit)
        nx = self.nx.value
        ny = self.ny.value
        nz = self.nz.value

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)
        
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
        if x is not None:
            ax.set_xlabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15)
        elif y is not None:
            ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15)
        elif z is not None:
            ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15)
        else:
            raise ValueError(utils._prestring() + "Please select an axis and an integer slice to plot.")
        if "xlabel" in kwargs:
            ax.set_xlabel(**kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(**kwargs["ylabel"])
            
        #Figure title
        if x is not None:
            ax.set_title(r"Slice of grid at x = {}".format(x),fontsize=15)
        elif y is not None:
            ax.set_title(r"Slice of grid at y = {}".format(y),fontsize=15)
        elif z is not None:
            ax.set_title(r"Slice of grid at z = {}".format(z),fontsize=15)
        if "title" in kwargs:
            ax.set_title(**kwargs["title"])


        ############### Plotting start ################

        #Colors!
        cmap = copy.copy(mpl.cm.get_cmap(cmap))
        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()

        #Masking below a certain threshold
        if mask_below is not None:
            mask_below = u.Quantity(mask_below,unit=ndensity_unit).value
            data = np.ma.masked_array(data,mask=data<mask_below)
    
        #Note: Matplotlib and Numpy define axis 0 and 1 OPPOSITE!!!
        #Hence, imshow plot is transposed.

        if x is not None:
            plot_data = data[x,:,:].T
            extent=[ymin,ymax,zmin,zmax]
        elif y is not None:
            plot_data = data[:,y,:].T
            extent=[xmin,xmax,zmin,zmax]
        elif z is not None:
            plot_data = data[:,:,z].T
            extent=[xmin,xmax,ymin,ymax]

        #Checking if empty
        if mask_below is not None:
            if plot_data.count() == 0:
                raise RuntimeError(utils._prestring() + "Nothing below the masking threshold!")

        #Making the plots
        ax.set_facecolor('black')
        imshow = ax.imshow(plot_data, cmap=cmap, norm=norm, origin='lower', extent=extent)

        #Colorbar
        cbar = fig.colorbar(imshow, ax=ax,fraction=0.046, pad=0.02)
        cbar_label = r'Number density $n$ [{}]'.format(ndensity_unit)
        cbar.set_label(cbar_label,fontsize=15)
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)

        ############### Plotting end ################

        #Axes limits
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

    ######################################################################

    def plot_projection(self, 
                        projection='z',
                        length_unit=None,
                        ndensity_unit=u.cm**-3,
                        log=False,
                        mask_below=None,
                        filaments=None,
                        sinks=None,
                        cmap='magma',
                        save=None,
                        **kwargs):

        """
        
        Plot the 2D projection of the number density grid.
        Can also plot filaments or AREPO sinks together!

        Parameters
        ----------

        projection : `str`, optional
            Axis of projection of the grid, either ``x``, ``y`` or ``z``.

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use. If ``None`` (default),
            takes the value ``u.pix`` if ``scale='pixel'``, 
            else ``u.cm`` if ``scale='physical'``.

        ndensity_unit : `~astropy.units.Unit`, optional
            The unit of number density to use. 
            Default value is ``u.cm**-3``.
        
        log: `bool`, optional
            If ``True``, the number density is plotted on a log-scale.
            If ``False`` (default), it is plotted on a linear scale.

        mask_below : `float` or `~astropy.units.Quantity`, optional
            The number density below which all grid points are omitted from the plot.
            This ensures only certain dense regions are plotted.
            If `float`, then assumed to be in ``ndensity_unit`` units.

        filaments : list of `~fiesta.disperse.Filament`'s
            List of `~fiesta.disperse.Filament`'s to plot along with the grid. 
            Requires the filaments to have the same ``scale`` as the grid.

        sinks : `list` of `list`'s, optional
            List of sink positions to plot. Requires the grid to have ``scale='physical'``.

        cmap : `str` or `~matplotlib.colors.Colormap`, optional
            Colormap of the plot. Default is ``'magma'``.

        save : `str`, optional
            File path to save the plot.
            If ``None`` (default), plot is not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : `~matplotlib.figure.Figure`
            Main `~matplotlib.figure.Figure` instance.

        """

        #Shedding units for ease of use
        if(self.scale=="physical"):
            internal_unit = u.cm
        elif(self.scale=="pixel"):
            internal_unit = u.pix
        if length_unit is None:
            length_unit = internal_unit
        utils.check_unit(length_unit, internal_unit)
        if(self.scale=="physical"):
            ndensity_length_unit = ndensity_unit**(-1/3)
            xlength = self.xlength.to_value(ndensity_length_unit)
            ylength = self.ylength.to_value(ndensity_length_unit)
            zlength = self.zlength.to_value(ndensity_length_unit)
        elif(self.scale=="pixel"):
            ndensity_length_unit = u.pix
            xlength = self.xlength.value
            ylength = self.ylength.value
            zlength = self.zlength.value
        utils.check_unit(ndensity_unit, u.cm**-3)
        
        data = self.get_ndensity().to_value(ndensity_unit)
        xmin = self.xmin.to_value(length_unit)
        xmax = self.xmax.to_value(length_unit)
        ymin = self.ymin.to_value(length_unit)
        ymax = self.ymax.to_value(length_unit)
        zmin = self.zmin.to_value(length_unit)
        zmax = self.zmax.to_value(length_unit)
        nx = self.nx.value
        ny = self.ny.value
        nz = self.nz.value

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)

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
        if(projection=='x' or projection=='X'):
            ax.set_xlabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15)
        elif(projection=='y' or projection=='Y'):
            ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15)
        elif(projection=='z' or projection=='Z'):
            ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15)
            ax.set_ylabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15)
        else:
            raise ValueError(utils._prestring() + "Invalid projection.")
        if "xlabel" in kwargs:
            ax.set_xlabel(**kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(**kwargs["ylabel"])

        #Figure title
        ax.set_title("",fontsize=15)
        if "title" in kwargs:
            ax.set_title(**kwargs["title"])


        ############### Plotting start ################

        #Colors!
        cmap = copy.copy(mpl.cm.get_cmap(cmap))
        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()

        #Masking below a certain threshold
        if mask_below is not None:
            mask_below = u.Quantity(mask_below,unit=ndensity_unit).value
            data = np.ma.masked_array(data,mask=data<mask_below)

        #Note: Matplotlib and Numpy define axis 0 and 1 OPPOSITE!!!
        #Hence, the projected data is transposed.

        if(projection=='x' or projection=='X'):
            (xaxis, yaxis) = (1,2)
            plot_data = np.sum(data,axis=0).T * xlength
            extent=[ymin,ymax,zmin,zmax]
        elif(projection=='y' or projection=='Y'):
            (xaxis, yaxis) = (0,2)
            plot_data = np.sum(data,axis=1).T * ylength
            extent=[xmin,xmax,zmin,zmax]
        elif(projection=='z' or projection=='Z'):
            (xaxis, yaxis) = (0,1)
            plot_data = np.sum(data,axis=2).T * zlength
            extent=[xmin,xmax,ymin,ymax]
            
        #Checking if empty
        if mask_below is not None:
            if plot_data.count() == 0:
                raise RuntimeError(utils._prestring() + "Nothing below the masking threshold!")

        #Making the plot
        ax.set_facecolor('black')
        cb = ax.imshow(plot_data, extent=extent, cmap=cmap, norm=norm, origin='lower', zorder=0)

        #Filament plot
        if filaments is not None:
            for fil in filaments:
                if(fil.scale != self.scale):
                    raise ValueError(utils._prestring() + "Filament and ArepoCubicGrid scales don't match!")
                else:
                    #Shedding units for ease of use
                    x = fil.samps[:,xaxis].to_value(length_unit)
                    y = fil.samps[:,yaxis].to_value(length_unit)
                    ax.scatter(x[0], y[0], c='red', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], c='red', s=2, zorder=2)
                    ax.plot(x, y, linewidth=1, c='gold')

        #Sink plot
        if sinks is not None:
            if(self.scale != "physical"):
                raise ValueError(utils._prestring() + "ArepoCubicGrid needs to be in a physical scale to plot the sinks!")
            else:
                #Shedding units for ease of use
                x = sinks[:,xaxis].to_value(length_unit)
                y = sinks[:,yaxis].to_value(length_unit)
                ax.scatter(x, y, s=2, c='red', zorder=2)

        #Colorbar
        cbar = fig.colorbar(cb, ax=ax,fraction=0.046, pad=0.02)
        cbar.set_label(r'Column density [{}]'.format((ndensity_length_unit*ndensity_unit).to_string()),fontsize=15)
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)

        ############### Plotting end ################

        #Axes limits
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