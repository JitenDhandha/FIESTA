######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains classes to interact with AREPO data, 
both structured and unstructured meshes, such reading/writing
routines and plotting algoritms. Currently only supports 3D data.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#FIESTA
from fiesta import units as ufi
from fiesta import properties as prop
from fiesta import _areporeadwrite

######################################################################
#                       CLASS: ArepoVoronoiGrid                      #
######################################################################

class ArepoVoronoiGrid:

    """
    Class to deal with AREPO snapshosts in the form of
    native unstructured meshes.

    Attributes
    ----------

    file_path : str
        Path of snapshot file.
        
    data : dict
        Dictionary containing all the data in the snapshot (not useful in lieu of other variables).

    header : dict
        Dictionary containing all the information in the header of the snapshot (not useful in lieu of other variables).

    size : int
        Size of one axis of the simulation box in AREPO units where ``size**3`` encompasses the whole 3D simulation.

    time : float
        Time corresponding ot the AREPO snapshot.

    ntot : int
        Total number of particles in the snapshot.

    ngas : int
        Number of gas particles in the snapshot.

    nsink : int
        Number of sink particles in the snapshot.

    gas_ids : numpy.ndarray
        1D array containing all indices corresponding to gas particles.

    sink_ids : numpy.ndarray
        1D array containing all indices corresponding to sink particles.

    pos : numpy.ndarray
        1D array containing position of all particles in AREPO ``ulength`` units.

    vel : numpy.ndarray
        1D array containing velocities of all particles in AREPO ``uvel`` units.

    mass : numpy.ndarray
        1D array containing mass of all particles in ARPEO ``umass`` units.

    chem : numpy.ndarray
        1D array containing chemistry of all particles as fractional chemical abundances.

    rho : numpy.ndarray
        1D array containing density of all gas particles in AREPO ``umass/ulength**3`` units.

    utherm : numpy.ndarray
        1D array containing energy per unit mass of all particles in AREPO ``1/ulength**3`` units.

    cloud_size : float
        Size of one axis of the bounding box surrounding the molecular cloud.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_Tmax : float
        Maximum temperature of the molecular cloud in Kelvin (usually ~20-30K).
        Only relevant when :attr:`detect_cloud` is called.

    cloud_ids : numpy.ndarray
        1D array containing all indices corresponding to cloud particles.
        Only relevant when :attr:`detect_cloud` is called.

    non_cloud_ids : numpy.ndarray
        1D array containing all indices corresponding to non-cloud particles.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_ntot : int
        Number of cloud particles.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_pos : numpy.ndarray
        1D array containing position of cloud particles in AREPO ``ulength`` units.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_vel : numpy.ndarray
        1D array containing velocities of cloud particles in AREPO ``uvel`` units.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_mass : numpy.ndarray
        1D array containing mass of cloud particles in ARPEO ``umass`` units.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_chem : numpy.ndarray
        1D array containing chemistry of cloud particles as fractional chemical abundances.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_rho : numpy.ndarray
        1D array containing density of cloud particles in AREPO ``umass/ulength**3`` units.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_utherm : numpy.ndarray
        1D array containing energy per unit mass of cloud particles in AREPO ``1/ulength**3`` units.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_density : numpy.ndarray
        1D array containing density of cloud particles in g/cm^3.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_ndensity : numpy.ndarray
        1D array containing number density of cloud particles in 1/cm^3.
        Only relevant when :attr:`detect_cloud` is called.

    cloud_temperature : numpy.ndarray
        1D array containing temperature of cloud particles in Kelvin.
        Only relevant when :attr:`detect_cloud` is called.

    Parameters
    ----------

    file_path : str, optional
        The path of the AREPO snapshot file to read in and construct
        the object.

    verbose : bool, optional
        If ``True``, prints output during file reading.
    
    """
    
    def __init__(self, file_path=None, verbose=False):
        
        #Create variables
        self.file_path = str
        self.data = {}
        self.header = {}
        self.size = int
        self.time = float
        self.ntot = int
        self.ngas = int
        self.nsink = int
        self.gas_ids = []
        self.sink_ids = []
        self.pos = []
        self.vel = []
        self.mass = []
        self.chem = []
        self.rho = []
        self.utherm = []
        
        #Set variables
        if file_path is not None:
            self.read_file(file_path, verbose)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
    
    def read_file(self, file_path, verbose=False):
        """
        
        Read in an AREPO snapshot file.

        Parameters
        ----------

        file_path : str, optional
            The path of the AREPO snapshot file to read in and construct
            the object.

        verbose : bool, optional
            If ``True``, prints output during file reading.

        """

        if(verbose):
            print("FIESTA >> Loading AREPO Voronoi grid from \"{}\" ...".format(file_path))
        
        self.file_path = file_path
        self.data, self.header = _areporeadwrite.read_snapshot(self.file_path)
        self.size = int(self.header['boxsize'][0])
        self.time = float(self.header['time'][0])
        self.ntot = int(sum(self.header['num_particles']))
        self.ngas = int(self.header['num_particles'][0])
        self.nsink = int(self.header['num_particles'][5])
        self.gas_ids = np.arange(0,self.ngas)
        self.sink_ids = np.arange(self.ntot-self.nsink,self.ntot)
        self.pos = self.data['pos']
        self.vel = self.data['vel']
        self.mass = self.data['mass']
        self.chem = self.data['chem']
        self.rho = self.data['rho']
        self.utherm = self.data['u_therm']
        
        if(verbose):
            print("FIESTA >> Size: {}^3".format(self.size))
            print("FIESTA >> # of particles: {}".format(self.ntot))
            print("FIESTA >> # of gas particles: {}".format(self.ngas))
            print("FIESTA >> # of sink particles: {}".format(self.nsink))
            if(self.header['flag_doubleprecision']):
                  print("FIESTA >> Precision: double")
            else:
                  print("FIESTA >> Precision: float")
                  
        if(verbose):
            print("FIESTA >> Completed loading AREPO Voronoi grid from \"{}\"".format(file_path))

    def write_file(self, file_path, io_flags=None, verbose=False):
        """
        
        Write out an AREPO snapshot file.

        Parameters
        ----------

        file_path : str
            The name and path of the AREPO snapshot file to write the AREPO
            Voronoi grid into.

        io_flags : dict
            Dictionary containing the input/output AREPO flags. If ``None`` (default), 
            set to ``{'mc_tracer': False, 'time_steps': False, 'sgchem': True, 
            'variable_metallicity': False}``.

        verbose : bool, optional
            If ``True``, prints output during file writing.

        """

        #Setting IO flags
        if io_flags is None:
            io_flags = {'mc_tracer'           : False,
                        'time_steps'          : False,
                        'sgchem'              : True,
                        'variable_metallicity': False}
        _areporeadwrite.io_flags = io_flags

        if(verbose):
            print("FIESTA >> Writing AREPO Voronoi grid to \"{}\" ...".format(file_path))

        _areporeadwrite.write_snapshot(file_path, self.header, self.data)

        if(verbose):
            print("FIESTA >> Completed writing AREPO Voronoi grid to \"{}\"".format(file_path))

    ######################################################################
    #                     USEFUL PHYSICAL PROPERTIES                     #
    ######################################################################

    def get_density(self):
        """

        Returns the density of all gas particles in units of g/cm^3.

        Returns
        -------

        density : numpy.ndarray
            1D array containing density of gas particles in g/cm^3.

        """

        density = self.rho * ufi.udensity
        return density

    def get_ndensity(self):
        """

        Returns the number density of all gas particles in units of 1/cm^3.

        Returns
        -------

        ndensity : numpy.ndarray
            1D array containing number density of gas particles in 1/cm^3.

        """

        ndensity = prop.calc_number_density(self.rho)
        return ndensity

    def get_temperature(self):
        """

        Returns the temperature of all particles in Kelvin.

        Returns
        -------

        ndensity : numpy.ndarray
            1D array containing temperature of particles in Kelvin.

        """
        nTOT = prop.calc_chem_density(self.chem, self.get_ndensity())[4]
        temperature = prop.calc_temperature(self.rho, self.utherm, nTOT)
        return temperature
                
    ######################################################################
    #                       PLOTTING FUNCTIONS                           #
    ######################################################################
        
    def plot_projection(self, 
                        projection='z',
                        scale=1.0,
                        log=False,
                        ROI=None,
                        cmap='magma',
                        bins=1000,
                        save=None,
                        **kwargs):
        
        """
        
        Plot mass-weighted 2d histogram projection of the AREPO Voronoi grid 
        along an axis.

        Parameters
        ----------
        
        projection : str, optional
            Axis of projection ``x``, ``y`` or ``z`` (default).

        scale : float, optional
            The number to scale the plot axes with. Useful for converting plots to 
            commonly used length scales such as ``units.uparsec`` (parsec) or 
            ``units.ulength`` (cm).

        log: bool, optional
            If ``True``, histogram is normalized on a logarithmic scale.
            If ``False`` (default), histogram is normalized on a linear scale.

        ROI : list or numpy.ndarray, optional
            Delinate a Region of Interest (ROI) in the projection as a red
            box defined by the bounds ``[xmin, xmax, ymin, ymax]``.

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap of the histogram. Default is ``'magma'``.

        bins : None or int or [int, int] or array-like or [array, array], optional
            See *matplotlib.axes.Axes.hist2d* documentation for more detail. 
            Default value is ``1000``.

        save : str, optional
            The name of the file to save the plot as. If ``None`` (default), plot is
            not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : matplotlib.figure.Figure
            Main *matplotlib.figure.Figure* instance.

        ax : matplotlib.axes.Axes
            Main *matplotlib.axes.Axes* instance.

        his2d: matplotlib.axes.Axes.hist2d
            Main *matplotlib.axes.Axes.hist2d* instance.

        """

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)
        
        #Axes limits
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])
        
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
            ax.set_xlabel(r"$y$",fontsize=15)
            ax.set_ylabel(r"$z$",fontsize=15)
        elif(projection=='y' or projection=='Y'):
            ax.set_xlabel(r"$x$",fontsize=15)
            ax.set_ylabel(r"$z$",fontsize=15)
        elif(projection=='z' or projection=='Z'):
            ax.set_xlabel(r"$x$",fontsize=15)
            ax.set_ylabel(r"$y$",fontsize=15)
        else:
            raise ValueError("FIESTA >> Invalid projection.")
        if "xlabel" in kwargs:
            ax.set_xlabel(**kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(**kwargs["ylabel"])
            
        #Figure title
        ax.set_title("",fontsize=15)
        if "title" in kwargs:
            ax.set_title(**kwargs["title"])

        ############### Plotting start ################

        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()

        xpos = self.pos[:,0]*scale
        ypos = self.pos[:,1]*scale
        zpos = self.pos[:,2]*scale

        if(projection=='x' or projection=='X'):
            hist2d = ax.hist2d(ypos, zpos, cmap=cmap, norm=norm, bins=bins, weights=self.mass)
        elif(projection=='y' or projection=='Y'):
            hist2d = ax.hist2d(xpos, zpos, cmap=cmap, norm=norm, bins=bins, weights=self.mass)
        elif(projection=='z' or projection=='Z'):
            hist2d = ax.hist2d(xpos, ypos, cmap=cmap, norm=norm, bins=bins, weights=self.mass)

        if ROI is not None:
            #This part plots a red square around the ROI (Region of Interest) in the simulation
            #The format of ROI is [xmin,xmax,ymin,ymax]
            xmin, xmax = ROI[0], ROI[1]
            ymin, ymax = ROI[2], ROI[3]
            ax.vlines(x=xmin, ymin=ymin, ymax=ymax, color='red')
            ax.vlines(x=xmax, ymin=ymin, ymax=ymax, color='red')
            ax.hlines(y=ymin, xmin=xmin, xmax=xmax, color='red')
            ax.hlines(y=ymax, xmin=xmin, xmax=xmax, color='red')

        ############### Plotting end ################

        #Text
        if "text" in kwargs:
            ax.text(**kwargs["text"],transform=ax.transAxes)

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig, ax, hist2d

    def plot_inital_velocity(self,
                             projection='z',
                             scale=1.0,
                             cmap='RdBu',
                             save=None,
                             **kwargs):

        """
        
        Plot a 3D heatmap of the velocities of a molecular cloud.
        Requires :attr:`detect_cloud` to be run before.

        Parameters
        ----------
        
        projection : str, optional
            Velocity component to plot: ``x``, ``y`` or ``z`` (default).

        scale : float, optional
            The number to scale the plot axes with. Useful for converting plots to 
            commonly used length scales such as ``units.uparsec`` (parsec) or 
            ``units.ulength`` (cm).

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap of the plot. Default is ``'RdBu'``.

        save : str, optional
            The name of the file to save the plot as. If ``None`` (default), plot is
            not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : matplotlib.figure.Figure
            Main *matplotlib.figure.Figure* instance.

        ax : matplotlib.axes.Axes
            Main *matplotlib.axes.Axes* instance.

        scatter: matplotlib.axes.Axes.scatter
            Main *matplotlib.axes.Axes.scatter* instance.

        """

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111,projection='3d')
        
        #Axes limits
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])
        if "zlim" in kwargs:
            ax.set_zlim(**kwargs["zlim"])
        
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
        ax.set_xlabel(r"$x$",fontsize=15,labelpad=5)
        ax.set_ylabel(r"$y$",fontsize=15,labelpad=5)
        ax.set_zlabel(r"$z$",fontsize=15,labelpad=5)
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
        
        x = self.cloud_pos[:,0]*scale
        y = self.cloud_pos[:,1]*scale
        z = self.cloud_pos[:,2]*scale

        if(projection.lower()=='x'):
            vel = self.cloud_vel[:,0]*ufi.uvel/100000.0 #converting to km/s
        elif(projection.lower()=='y'):
            vel = self.cloud_vel[:,1]*ufi.uvel/100000.0 #converting to km/s
        elif(projection.lower()=='z'):
            vel = self.cloud_vel[:,2]*ufi.uvel/100000.0 #converting to km/s
        else:
            raise ValueError("FIESTA >> Invalid projection.")

        scatter = ax.scatter(x,y,z, c=vel, s=0.5, cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.1)
        cbar.set_label(r'Velocity projection ${}$ [km/s]'.format(projection), size=15, color='black')
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.outline.set_edgecolor('black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        ############### Plotting end ################

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig, ax, scatter

    ######################################################################
    #                        CLOUD-BASED FUNCTIONS                       #
    ######################################################################

    def detect_cloud(self, cloud_size, cloud_Tmax=30, verbose=False):

        """
        
        Detects the presence of a molecular cloud centered in the simulation box,
        bounded within a cubical region, and with a maximum temperature. The code
        is currently based on the assumption that the cloud is spherical which is
        usually the case when setting initial conditions. This function sets various 
        instance variables to access the cloud, as follows:

        ``cloud_size``
        ``cloud_Tmax``
        ``cloud_ids``
        ``non_cloud_ids``
        ``cloud_pos``
        ``cloud_vel``
        ``cloud_mass``
        ``cloud_chem``
        ``cloud_rho``
        ``cloud_utherm``
        ``cloud_density``
        ``cloud_ndensity``
        ``cloud_temperature``

        Parameters
        ----------
        
        cloud_size : float
            Size of the cubical region, in AREPO units, within which the cloud
            is bounded assuming it is centered in the simulation box.
            This is especially useful when the simulation box is much bigger 
            than the initial cloud.

        cloud_Tmax : float, optional
            Maximum temperature of the cloud in Kelvin (usually ~20-30K). This 
            ensures that non cloud particles within the ``cloud_size`` box
            are not erroneously detected as part of the cloud.

        verbose : bool, optional
            If ``True``, prints output during cloud detection.

        """

        if(verbose):
            print("FIESTA >> Detecting cloud...")
        
        #CREATE VARIABLES FOR THE CLOUD IN THE GRID
        #
        self.cloud_size = float
        self.cloud_Tmax = float
        #
        self.cloud_ids = []
        self.non_cloud_ids = []
        self.cloud_ntot = int
        #
        self.cloud_pos = []  #in AREPO units
        self.cloud_vel = []  #in AREPO units
        self.cloud_mass = []  #in AREPO units
        self.cloud_chem = []  #in AREPO units
        self.cloud_rho = []  #in AREPO units
        self.cloud_utherm = []  #in AREPO units
        #
        self.cloud_density = [] #in g/cm^-3
        self.cloud_ndensity = []  #in cm^-3
        self.cloud_temperature = []  #in K
        
        
        #DEFINING BOUNDARIES OF THE CLOUD (a cubical region containing it)
        self.cloud_size = cloud_size
        cloud_min = int(self.size/2 - self.cloud_size/2)
        cloud_max = int(self.size/2 + self.cloud_size/2)
        
        #SETTING MAXIMUM TEMPERATURE FOR THE CLOUD
        self.cloud_Tmax = cloud_Tmax
    
        
        #EXTRACTING THE CLOUD FROM THE GAS
        if(verbose):
            print("FIESTA >> Started extracting cloud particle indices...")
        temps = self.get_temperature() #For checking temperature of cell
        for i in range(self.ntot):
            if(i%500000==0):
                print(str(i)+"/"+str(self.ntot)+" total cells") #counter
            pos = self.pos[i]
            #This is a first filter based on cloud size (the box within which the cloud is)
            if(cloud_min<pos[0]<cloud_max and cloud_min<pos[1]<cloud_max and cloud_min<pos[2]<cloud_max):
                #This is a second filter based on temperature (to remove "outliers" from cloud size box)
                temp = temps[i]
                if(temp<cloud_Tmax):
                    self.cloud_ids.append(i)
        self.non_cloud_ids = np.delete(np.arange(self.ntot), self.cloud_ids)
        if(verbose):
            print("FIESTA >> Completed extracting cloud particle indices.")
        self.cloud_ntot = len(self.cloud_ids)
        if(verbose):
            print("FIESTA >> Note: The number of cloud particles is {}".format(self.cloud_ntot))
        
        
        #SETTING CLOUD VARIABLES FOR EASE OF USE
        self.cloud_pos = self.pos[self.cloud_ids]
        self.cloud_vel = self.vel[self.cloud_ids]
        self.cloud_mass = self.mass[self.cloud_ids]
        self.cloud_chem = self.chem[self.cloud_ids]
        self.cloud_rho = self.rho[self.cloud_ids]
        self.cloud_utherm = self.utherm[self.cloud_ids]
        self.cloud_density = self.get_density()[self.cloud_ids]
        self.cloud_ndensity = self.get_ndensity()[self.cloud_ids]
        self.cloud_temperature = self.get_temperature()[self.cloud_ids]

    def update_cloud_velocity(self, tvg, virial_ratio=1.0, verbose=False):

        """
        
        Set velocity of the molecular cloud particles in accordance with a
        :class:`~fiesta.turbvel.TurbulentVelocityGrid`. 
        Requires :attr:`detect_cloud` to be run before. The cubic grid is first 
        scaled to :attr:`cloud_size` and then linearly interpolated to fill in 
        the AREPO Voronoi grid of cloud particles.

        Parameters
        ----------
        
        tvg : :class:`~fiesta.turbvel.TurbulentVelocityGrid`
            The turbulent velocity grid to use for assigning velocities to the cloud.

        virial_ratio : float, optional
            The virial ratio :math:`\\alpha_\\mathrm{vir} = \\frac{E_\\mathrm{kin}}{E_\\mathrm{grav}}`
            for scaling the velocities, since the turbulent velocity grid is normalized and unitless.
            A ratio of ``1.0`` (default) means equilibrium condition.

        verbose : bool, optional
            If ``True``, prints output during when setting cloud velocities.

        """

        #CREATING INTERPOLATION FUNCTION
        if(verbose):
            print("FIESTA >> Started creating linear interpolation function for velocities...")
        #Creating grid for interpolating
        cloud_min = int(self.size/2 - self.cloud_size/2)
        cloud_max = int(self.size/2 + self.cloud_size/2)
        scaling = float(self.cloud_size)/float(tvg.size) #Scaling velocity gridsize to cloud gridsize
        range_vals = np.arange(cloud_min,cloud_max,scaling)
        #Interpolating the data
        interp_x = RegularGridInterpolator([range_vals]*3,tvg.vx,method='linear')
        interp_y = RegularGridInterpolator([range_vals]*3,tvg.vy,method='linear')
        interp_z = RegularGridInterpolator([range_vals]*3,tvg.vz,method='linear')
        if(verbose):
            print("FIESTA >> Completed creating linear interpolation function for velocities {}.")

        #CREATING VELOCITY ARRAY
        if(verbose):
            print("FIESTA >> Started interpolating velocities for the AREPO grid...")
        #Filling with all zero velocity first
        vel_new = np.zeros((self.ntot,3)) 
        vx_s = interp_x(self.cloud_pos)
        vy_s = interp_y(self.cloud_pos)
        vz_s = interp_z(self.cloud_pos)
        vel_new[self.cloud_ids] = np.array([vx_s,vy_s,vz_s]).T
        if(verbose):
            print("FIESTA >> Completed interpolating velocities for the AREPO grid.")
        #Note that the velocity array is in arbitrary units so will have to be scaled    

        #SCALING VELOCITIES TO A SPECIFIC VIRIAL RATIO
        if(verbose):
            print("FIESTA >> Started scaling velocities to satisfy virial ratio {}...".format(virial_ratio))
        #Finding the scaling factor first
        E_k = prop.calc_kinetic_energy(self.cloud_mass, vel_new[self.cloud_ids])
        E_g = prop.calc_gravitational_potential_energy(self.cloud_mass, self.cloud_rho)
        current_ratio = 2.0 * E_k / E_g
        scaling = np.sqrt(virial_ratio/current_ratio)
        if(verbose):
            print("FIESTA >> Note: the scaling factor is {}.".format(scaling))
        #Scaling the velocity here
        vel_new *= scaling
        if(verbose):
            print("FIESTA >> Completed scaling velocities to satisfy virial ratio {}.".format(virial_ratio))

        #CHUCKING VELOCITIES INTO VARIABLES
        self.vel = vel_new
        self.cloud_vel = self.vel[self.cloud_ids]
        self.data['vel'] = self.vel

                  
######################################################################
#                        CLASS: ArepoCubicGrid                       #
######################################################################
                  
class ArepoCubicGrid:
    
    """
    Class to deal with AREPO cubic/cuboidal grids that have been 
    ray-casted from snapshots.

    Attributes
    ----------

    file_path : str
        Path of grid file.

    density : numpy.ndarray
        3D array containing density at each grid point in AREPO 
        ``umass/ulength**3`` units.

    nx : int
        Number of grid points along x-axis of the grid.

    ny : int
        Number of grid points along y-axis of the grid.

    nz : int
        Number of grid points along z-axis of the grid.

    scale : str
        Either ``pixel`` if the grid is not scaled to the source
        AREPO snapshot, else ``AREPO``.

    xmin : float
        Minimum value of the x-axis of the grid, in either ``pixel``
        or ``AREPO`` units.

    xmin : float
        Maximum value of the x-axis of the grid, in either ``pixel``
        or ``AREPO`` units.

    ymin : float
        Minimum value of the y-axis of the grid, in either ``pixel``
        or ``AREPO`` units.

    ymax : float
        Maximum value of the y-axis of the grid, in either ``pixel``
        or ``AREPO`` units.

    zmin : float
        Minimum value of the z-axis of the grid, in either ``pixel``
        or ``AREPO`` units.

    zmin : float
        Maximum value of the z-axis of the grid, in either ``pixel``
        or ``AREPO`` units.

    xlength : float
        Length of one voxel along x-axis in either ``1`` pixel
        or AREPO ``ulength`` units.

    ylength : float
        Length of one voxel along y-axis in either ``1`` pixel
        or AREPO ``ulength`` units.

    zlength : float
        Length of one voxel along z-axis in either ``1`` pixel
        or AREPO ``ulength`` units.

    Parameters
    ----------

    file_path : str, optional
        The path of the AREPO grid file to read in and construct
        the object.

    AREPO_bounds : list or numpy.ndarray, optional
        The bounds of the source AREPO snapshot corresponding
        to the grid, in the format ``[xmin, xmax, ymin, ymax,
        zmin, zmax]``. If ``None`` (default), then
        ``scale='pixel'``, else ``scale='AREPO'``.

    verbose : bool, optional
        If ``True``, prints output during file reading.
    
    """

    def __init__(self, file_path=None, AREPO_bounds=None, verbose=False):
        
        
        #Create variables
        #Data variables
        self.file_path = str
        self.density =[]
        self.nx = int
        self.ny = int
        self.nz = int
        #Scale-related variables
        self.scale = str
        self.xmin = float
        self.xmax = float
        self.ymin = float
        self.ymax = float
        self.zmin = float
        self.zmax = float
        self.xlength = float
        self.ylength = float
        self.zlength = float

        #Set variables
        if file_path is not None:
            self.read_from_binary(file_path, verbose)

            if AREPO_bounds is not None:
                self.set_scale("AREPO", AREPO_bounds)

        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################

    def read_from_binary(self, file_path, verbose=False):
        """
        
        Read in an AREPO grid file.

        Parameters
        ----------

        file_path : str
            The path of the AREPO grid file to read in and construct
            the object.

        verbose : bool, optional
            If ``True``, prints output during file reading.

        """

        if(verbose):
            print("FIESTA >> Loading AREPO Cubic grid from \"{}\" ...".format(file_path))
            
        self.file_path = file_path
        #Read 3D data
        self.density = _areporeadwrite.read_grid(self.file_path)
        #Save dimensions
        self.nx, self.ny, self.nz = self.density.shape

        #Default scale
        self.set_scale("pixel")

        if(verbose):
            print("FIESTA >> Completed loading AREPO Cubic grid from \"{}\"".format(file_path))

    def write_to_FITS(self, FITS_file_path, verbose=False):
        """
        
        Write out the AREPO grid into a .FITS file.

        Parameters
        ----------

        FITS_file_path : str
            The  name and path of the .FITS file to write the 
            AREPO Cubic grid into.

        verbose : bool, optional
            If ``True``, prints output during file writing.

        """
        
        if(verbose):
            print("FIESTA >> Writing AREPO Cubic grid to FITS at \"{}\" ...".format(FITS_file_path))
            
        hdu = fits.PrimaryHDU(data=self.density)
        hdu.writeto(FITS_file_path, overwrite=True)

        if(verbose):
            print("FIESTA >> Completed writing AREPO Cubic grid to FITS at \"{}\"".format(FITS_file_path))
            
    def write_mask_to_FITS(self, mask_FITS_file_path, mask_bounds, verbose=False):
        """
        
        Write a mask .FITS file. Useful for masking certain regions before
        feeding into DisPerSE.

        Parameters
        ----------

        mask_FITS_file_path : str
            The  name and path of the .FITS file to write the 
            masked grid into.

        mask_bounds : str
            The region *outside* which everything is masked, in
            the format ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        verbose : bool, optional
            If ``True``, prints output during file writing.

        """

        if(verbose):
            print("FIESTA >> Writing mask to FITS at \"{}\" ...".format(mask_FITS_file_path))
            
        mask_data = np.full((self.nx,self.ny,self.nz), 1, dtype=int)
        xmin, xmax = (mask_bounds[0], mask_bounds[1])
        ymin, ymax = (mask_bounds[2], mask_bounds[3])
        zmin, zmax = (mask_bounds[4], mask_bounds[5])
        mask_data[xmin:xmax,ymin:ymax,zmin:zmax] = 0
        hdu = fits.PrimaryHDU(data=mask_data)
        hdu.writeto(mask_FITS_file_path, overwrite=True)

        if(verbose):
            print("FIESTA >> Completed writing mask to FITS at \"{}\"".format(mask_FITS_file_path))

    ######################################################################
    #                         PHYSICAL PROPERTIES                        #
    ######################################################################

    def set_scale(self, scale, AREPO_bounds=None):
        """
        
        Set the scale of the grid to either pixels or AREPO units.

        Parameters
        ----------

        scale : str
            Either "AREPO" or "pixel".

        AREPO_bounds : list or numpy.ndarray, optional
            If ``scale='AREPO'``, the bounds of the source 
            AREPO snapshot corresponding to the grid, in the 
            format ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        """

        if(scale.lower()=="AREPO".lower()):
            self.scale = "AREPO"
            if AREPO_bounds is None:
                raise ValueError("FIESTA >> Invalid AREPO_bounds.")
            else:
                #Setting x, y and z min/max values
                self.xmin, self.xmax = (AREPO_bounds[0], AREPO_bounds[1])
                self.ymin, self.ymax = (AREPO_bounds[2], AREPO_bounds[3])
                self.zmin, self.zmax = (AREPO_bounds[4], AREPO_bounds[5])

                #Actual length of each grid cell
                self.xlength = float(self.xmax - self.xmin)/float(self.nx)
                self.ylength = float(self.ymax - self.ymin)/float(self.ny)
                self.zlength = float(self.zmax - self.zmin)/float(self.nz)                

        elif(scale.lower()=="pixel".lower()):
            self.scale = "pixel"
            self.xmin, self.xmax = (0, self.nx)
            self.ymin, self.ymax = (0, self.ny)
            self.zmin, self.zmax = (0, self.nz)
            self.xlength = 1
            self.ylength = 1
            self.zlength = 1   

        else:
            print("Incorrect scale option!")

    def get_density(self):
        """

        Returns the density of the grid in units of g/cm^3.

        Returns
        -------

        density : numpy.ndarray
            3D array containing density at each grid point in g/cm^3.

        """

        density = self.density * ufi.udensity
        return density

    def get_ndensity(self):
        """

        Returns the number density of the grid in units of 1/cm^3.

        Returns
        -------

        ndensity : numpy.ndarray
            3D array containing number density at each grid point in 1/cm^3.

        """

        ndensity = prop.calc_number_density(self.density)
        return ndensity
    
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
    
    def plot_grid(self,
                  scale=1.0,
                  log=False,
                  cmap='viridis',
                  mask_below=None,
                  filaments=None,
                  sinks=None,
                  save=None,
                  **kwargs):

        """
        
        Plot the 3D number density grid. Recommended use is with
        ``mask_below`` parameter so that not all grid points are 
        plotted. Can also plot filaments or AREPO sinks together!

        Parameters
        ----------

        scale : float, optional
            The number to scale the plot axes with. Useful for converting plots to 
            commonly used length scales such as ``units.uparsec`` (parsec) or 
            ``units.ulength`` (cm).
        
        log: bool, optional
            If ``True``, the number density is plotted on a log-scale.
            If ``False`` (default), it is plotted on a linear scale.

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap of the plot. Default is ``'viridis'``.

        mask_below : float, optional
            The number density below which all grid points are omitted from the plot.
            This ensures only certain dense regions are plotted (making the plot
            easier to see).

        filaments : list of :class:`~fiesta.disperse.Filament`
            List of filaments to plot along with the grid. Requires the filaments
            to have the same ``scale`` as the grid.

        sinks : list, optional
            List of sink positions to plot. Requires the grid to have ``scale='AREPO'``.

        save : str, optional
            The name of the file to save the plot as. If ``None`` (default), plot is
            not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : matplotlib.figure.Figure
            Main *matplotlib.figure.Figure* instance.

        ax : matplotlib.axes.Axes
            Main *matplotlib.axes.Axes* instance.

        scatter: matplotlib.axes.Axes.scatter
            Main *matplotlib.axes.Axes.scatter* instance.

        """

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111,projection='3d')
        
        #Axes limits
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        ax.set_zlim(self.zmin,self.zmax)
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])
        if "zlim" in kwargs:
            ax.set_zlim(**kwargs["zlim"])
        
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
        ax.set_xlabel(r"$x$",fontsize=15,labelpad=5)
        ax.set_ylabel(r"$y$",fontsize=15,labelpad=5)
        ax.set_zlabel(r"$z$",fontsize=15,labelpad=5)
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
        cmap1 = cmap
        cmap2 = mpl.cm.get_cmap('gray')
        alpha1 = 0.1
        alpha2 = 0.4

        data = self.get_ndensity()

        #Masking below a certain threshold
        if mask_below is not None:
            data = np.ma.masked_array(data,mask=data<mask_below)
           
        #Taking logarithm
        if(log):
            data = np.log10(data, out=np.zeros_like(data), where=(data>0))

        xr = np.linspace(self.xmin, self.xmax, num=self.nx, endpoint=True)*scale
        yr = np.linspace(self.ymin, self.ymax, num=self.ny, endpoint=True)*scale
        zr = np.linspace(self.zmin, self.zmax, num=self.nz, endpoint=True)*scale

        #Note: Matplotlib and Numpy define axis 0 and 1 OPPOSITE!!!
        #Hence, scatter plot requires np.swapaxes whereas contourf plots
        #are transposed.

        #3D PLOT
        X, Y, Z = np.meshgrid(xr,yr,zr)
        plot_data = np.swapaxes(data,0,1).flatten() #See above for swapaxes.
        scatter = ax.scatter(X, Y, Z, c=plot_data, cmap=cmap1, alpha=alpha1, s=1)

        #CONTOUR PLOTS PROJECTIONS
        #x projection
        y, z = np.meshgrid(yr,zr)
        contour_x = np.sum(data,axis=0).T #See above for transpose.
        ax.contourf(contour_x, y, z, zdir='x', levels=20, offset=self.xmin, cmap=cmap2, alpha=alpha2)
        
        #y projection
        x, z = np.meshgrid(xr,zr)
        contour_y = np.sum(data,axis=1).T #See above for transpose.
        ax.contourf(x, contour_y, z, zdir='y', levels=20, offset=self.ymax, cmap=cmap2, alpha=alpha2)
        
        #z projection
        x, y = np.meshgrid(xr,yr)
        contour_z = np.sum(data,axis=2).T #See above for transpose.
        ax.contourf(x, y, contour_z, zdir='z', levels=20, offset=self.zmin, cmap=cmap2, alpha=alpha2)

        #FILAMENT PLOT
        if filaments is not None:
            for fil in filaments:
                if(fil.scale != self.scale):
                    raise ValueError("FIESTA >> Filament and ArepoCubicGrid scales don't match!")
                else:
                    x = fil.samps[:,0]*scale
                    y = fil.samps[:,1]*scale
                    z = fil.samps[:,2]*scale
                    ax.scatter(x[0], y[0], z[0], c='black', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], z[-1], c='black', s=2, zorder=2)
                    ax.plot(x, y, z, linewidth=1, c='red')

        #SINK PLOT
        if sinks is not None:
            if(self.scale != "AREPO"):
                raise ValueError("FIESTA >> ArepoCubicGrid needs to be in AREPO scale to plot the sinks!")
            else:
                x = sinks[:,0]*scale
                y = sinks[:,1]*scale
                z = sinks[:,2]*scale
                ax.scatter(x, y, z, s=2, c='cyan', zorder=2)

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.1)
        if(log):
            cbar_label = r'Number density log($n$) [cm$^-3$]'
        else:
            cbar_label = r'Number density $n$ [cm$^-3$]'
        cbar.set_label(cbar_label, size=15, color='black')
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.outline.set_edgecolor('black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        ############### Plotting end ################

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig, ax, scatter
        
    ######################################################################
    
    def plot_slice(self, 
                   x=None, 
                   y=None, 
                   z=None,
                   scale=1.0,
                   log=False,
                   cmap='magma',
                   save=None,
                   **kwargs):

        """
        
        Plot a slice of 3D number density grid.

        Parameters
        ----------

        x : int, optional
            Integer corresponding to x-slice of the grid.
            One of ``x``, ``y`` or ``z`` is required.

        y : int, optional
            Integer corresponding to y-slice of the grid.
            One of ``x``, ``y`` or ``z`` is required.

        z : int, optional
            Integer corresponding to z-slice of the grid.
            One of ``x``, ``y`` or ``z`` is required.

        scale : float, optional
            The number to scale the plot axes with. Useful for converting plots to 
            commonly used length scales such as ``units.uparsec`` (parsec) or 
            ``units.ulength`` (cm).
        
        log: bool, optional
            If ``True``, the number density is plotted on a log-scale.
            If ``False`` (default), it is plotted on a linear scale.

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap of the plot. Default is ``'magma'``.

        save : str, optional
            The name of the file to save the plot as. If ``None`` (default), plot is
            not saved.

        **kwargs : dict, optional
            Additional *matplotlib*-based keyword arguments to control 
            finer details of the plot.

        Returns
        -------

        fig : matplotlib.figure.Figure
            Main *matplotlib.figure.Figure* instance.

        ax : matplotlib.axes.Axes
            Main *matplotlib.axes.Axes* instance.

        imshow : matplotlib.axes.Axes.imshow
            Main *matplotlib.axes.Axes.imshow* instance.

        """

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)
        
        #Axes limits
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])
        
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
            ax.set_xlabel(r"$y$",fontsize=15)
            ax.set_ylabel(r"$z$",fontsize=15)
        elif y is not None:
            ax.set_xlabel(r"$x$",fontsize=15)
            ax.set_ylabel(r"$z$",fontsize=15)
        elif z is not None:
            ax.set_xlabel(r"$x$",fontsize=15)
            ax.set_ylabel(r"$y$",fontsize=15)
        else:
            raise ValueError("FIESTA >> Please select an axis and an integer slice to plot.")
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

        data = self.get_ndensity()

        #Taking logarithm
        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()
    
        #Note: Matplotlib and Numpy define axis 0 and 1 OPPOSITE!!!
        #Hence, imshow plot is transposed.

        if x is not None:
            plot_data = data[x,:,:].T #See above for transpose.
            extent=[self.ymin,self.ymax,self.zmin,self.zmax]
        elif y is not None:
            plot_data = data[:,y,:].T #See above for transpose.
            extent=[self.xmin,self.xmax,self.zmin,self.zmax]
        elif z is not None:
            plot_data = data[:,:,z].T #See above for transpose.
            extent=[self.xmin,self.xmax,self.ymin,self.ymax]

        imshow = ax.imshow(plot_data, cmap=cmap, norm=norm, origin='lower', extent=extent)

        cbar = fig.colorbar(imshow, ax=ax,fraction=0.046, pad=0.02)
        if(log):
            cbar_label = r'Number density log($n$) [cm$^-3$]'
        else:
            cbar_label = r'Number density $n$ [cm$^-3$]'
        cbar.set_label(cbar_label,fontsize=15)
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)

        ############### Plotting end ################

        #Text
        if "text" in kwargs:
            ax.text(**kwargs["text"],transform=ax.transAxes)

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig, ax, imshow

    ######################################################################

    def plot_projection(self, 
                        projection='z',
                        log=False,
                        cmap=None,
                        mask_below=None,
                        network=None, #Passing a Network would mean filaments are plotted
                        bifurcations=False, 
                        avg=None,     #Passing an ArepoVoronoiGrid would mean sinks are plotted.
                        min_sink_mass=0,
                        ROI=None,
                        save=None,
                        **kwargs):

        print("FIESTA >> Plotting 2D projection of ArepoCubicGrid {}...".format(self))


        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)
        
        #Axes limits
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])

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
            ax.set_xlabel(r"$y$",fontsize=15)
            ax.set_ylabel(r"$z$",fontsize=15)
        elif(projection=='y' or projection=='Y'):
            ax.set_xlabel(r"$x$",fontsize=15)
            ax.set_ylabel(r"$z$",fontsize=15)
        elif(projection=='z' or projection=='Z'):
            ax.set_xlabel(r"$x$",fontsize=15)
            ax.set_ylabel(r"$y$",fontsize=15)
        else:
            raise ValueError("FIESTA >> Invalid projection.")
        if "xlabel" in kwargs:
            ax.set_xlabel(**kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(**kwargs["ylabel"])

        #Figure title
        ax.set_title("",fontsize=15)
        if "title" in kwargs:
            ax.set_title(**kwargs["title"])


        ############### Plotting start ################

        #Colours!
        if cmap is None:
            cmap = plt.cm.gist_gray
        else:
            cmap = mpl.cm.get_cmap(cmap)

        data = self.get_ndensity()

        #Masking below a certain threshold
        if mask_below is not None:
            data = np.ma.masked_array(data,mask=data<mask_below)

        #Taking logarithm
        if(log):
            if mask_below is not None:
                data = np.log10(data, out=np.zeros_like(data), where=(data>0))
                norm = mpl.colors.Normalize()
            else:
                cmap.set_bad((0,0,0))
                norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()

        #The transpose is CRUCIAL due to how imshow works!
        if(projection=='x' or projection=='X'):
            (xaxis, yaxis) = (1,2)
            plot_data = np.sum(data,axis=0).T
            extent=[self.ymin,self.ymax,self.zmin,self.zmax]
        elif(projection=='y' or projection=='Y'):
            (xaxis, yaxis) = (0,2)
            plot_data = np.sum(data,axis=1).T
            extent=[self.xmin,self.xmax,self.zmin,self.zmax]
        elif(projection=='z' or projection=='Z'):
            (xaxis, yaxis) = (0,1)
            plot_data = np.sum(data,axis=2).T
            extent=[self.xmin,self.xmax,self.ymin,self.ymax]


        #MAKING THE PLOTS
        if mask_below is None:
            cb = ax.imshow(plot_data, extent=extent, cmap=cmap, norm=norm, origin='lower', zorder=0)
        else:
            ax.set_facecolor('black')
            cb = ax.contourf(plot_data, levels=10, extent=extent, cmap=cmap, norm=norm, origin='lower', zorder=0)


        #IF NETWORK IS PASSED
        if network is not None:
            if(network.scale != self.scale):
                raise ValueError("FIESTA >> Network and ArepoCubicGrid scales don't match!")
            else: 
                #Plotting filaments
                for fil in network.fils:
                    x = fil.samps[:,xaxis]
                    y = fil.samps[:,yaxis]
                    ax.plot(x, y, linewidth=1.5, zorder=1)
                    ax.scatter(x[0], y[0], linewidth=1, c='black', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], linewidth=1, c='black', s=2, zorder=2)
                    #if(annotate):
                    #    ax.text(pos_x[fil.nsamp//2], pos_y[fil.nsamp//2], fil.idx, color='orange', fontsize=12) 
                #Plotting bifurcations
                if(bifurcations):
                    for bp in network.bifurcations:
                        x = bp.pos[xaxis]
                        y = bp.pos[yaxis]
                        ax.scatter(x, y, c='cyan', s=2, zorder=2)


        #IF VORONOI GRID IS PASSED
        if avg is not None:
            if(self.scale != "AREPO"):
                raise ValueError("FIESTA >> ArepoCubicGrid needs to be in AREPO scale to plot the sinks!")
            else:
                sink_mass = avg.mass[avg.sink_ids]
                sink_pos = avg.pos[avg.sink_ids]
                #Only plot above a certain minimum sink mass (in solar masses)
                trunc_sink_mass = sink_mass[sink_mass>min_sink_mass]
                trunc_sink_pos = sink_pos[sink_mass>min_sink_mass]
                x = trunc_sink_pos[:,xaxis]
                y = trunc_sink_pos[:,yaxis]
                s = 2.0 + 5.0*trunc_sink_mass/trunc_sink_mass.max()
                ax.scatter(x, y, s=s, c='red',zorder=2, alpha=0.5)

        #A red square around the ROI (Region of Interest) in the simulation
        if ROI is not None:
            #The format of ROI is [xmin,xmax,ymin,ymax]
            xmin, xmax = ROI[0], ROI[1]
            ymin, ymax = ROI[2], ROI[3]
            ax.vlines(x=xmin, ymin=ymin, ymax=ymax, color='red')
            ax.vlines(x=xmax, ymin=ymin, ymax=ymax, color='red')
            ax.hlines(y=ymin, xmin=xmin, xmax=xmax, color='red')
            ax.hlines(y=ymax, xmin=xmin, xmax=xmax, color='red')


        cbar = fig.colorbar(cb, ax=ax,fraction=0.046, pad=0.02)
        cbar.set_label(r'Column density',fontsize=15)
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)

        ############### Plotting end ################

        #Text
        if "text" in kwargs:
            ax.text(**kwargs["text"],transform=ax.transAxes)

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)