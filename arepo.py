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
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

#Own libs
from .units import *
from .physical_functions import *
from .plotting_functions import *
from . import binary_read as rsnap
from . import binary_write as wsnap

######################################################################
#                       CLASS: ArepoVoronoiGrid                      #
######################################################################

class ArepoVoronoiGrid:
    
    ######################################################################
    #                           CONSTRUCTOR                              #
    ######################################################################
    
    def __init__(self, file_path=None, verbose=False):
        
        #FILE PATH
        self.file_path = str
        
        #CREATE VARIABLES FOR THE WHOLE GRID
        #Raw data
        self.data = []
        self.header = []
        #Extracted data
        self.size = int
        self.time = float
        self.ntot = int
        self.ngas = int
        self.nsink = int
        self.gas_ids = []
        self.sink_ids = []
        self.pos = []  #in code units
        self.vel = []  #in code units
        self.mass = []  #in code units
        self.chem = []  #in code units
        self.rho = []  #in code units
        self.utherm = []  #in code units
        
        #SET VARIABLES
        if file_path is not None:
            self.read_file(file_path, verbose)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
    
    #READING DATA FROM FILE AND SETTING VARIABLES
    def read_file(self, file_path, verbose=False):

        if(verbose):
            print("FIESTA >> Loading AREPO Voronoi grid from \"{}\" ...".format(file_path))
        
        self.file_path = file_path
        self.data, self.header = rsnap.read_snapshot(self.file_path)
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
        
    #WRITING DATA TO FILE
    def write_file(self, file_path, io_flags=None, verbose=False):

        #Setting IO flags
        if io_flags is None:
            io_flags = {'mc_tracer'           : False,
                        'time_steps'          : False,
                        'sgchem'              : True,
                        'variable_metallicity': False}
        wsnap.io_flags = io_flags

        if(verbose):
            print("FIESTA >> Writing AREPO Voronoi grid to \"{}\" ...".format(file_path))

        wsnap.write_snapshot(file_path, self.header, self.data)

        if(verbose):
            print("FIESTA >> Completed writing AREPO Voronoi grid to \"{}\"".format(file_path))

    ######################################################################
    #                     USEFUL PHYSICAL PROPERTIES                     #
    ######################################################################

    #Number density. output: 1d array, units: in cm^-3
    def ndensity(self):
        ndensity = calc_number_density(self.rho)
        return ndensity

    #Temperature. output: 1d array, units: in K
    def temperature(self):
        nTOT = calc_chem_density(self.chem, self.ndensity())[4]
        temperature = calc_temperature(self.rho, self.utherm, nTOT)
        return temperature
                
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
        
    def plot_projection(self, 
                        projection='z', 
                        log=False,
                        ROI=None,
                        cmap=None,
                        bins=1000,
                        save=None,
                        **kwargs):
        
        print("FIESTA >> Plotting {} projection mass-weighted 2d histogram of ArepoVoronoiGrid {}...".format(projection,self))

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

        if cmap is None:
            cmap = plt.cm.magma
        else:
            cmap = mpl.cm.get_cmap(cmap)            
        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()

        xpos = self.pos[:,0]
        ypos = self.pos[:,1]
        zpos = self.pos[:,2]

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

    ######################################################################

    #Plotting heat map of the inital velocity
    def plot_inital_velocity(self,
    						 projection='z',
    						 cmap=None,
                             save=None,
    						 **kwargs):

        print("FIESTA >> Plotting {} velocity component of ArepoVoronoiGrid {}...".format(projection,self))

        #Figure properties
        if cmap is None:
            cmap = plt.cm.RdBu

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
        
        x = self.cloud_pos[:,0]
        y = self.cloud_pos[:,1]
        z = self.cloud_pos[:,2]

        if(projection.lower()=='x'):
            vel = self.cloud_vel[:,0]*uvel/100000.0 #converting to km/s
        elif(projection.lower()=='y'):
            vel = self.cloud_vel[:,1]*uvel/100000.0 #converting to km/s
        elif(projection.lower()=='z'):
            vel = self.cloud_vel[:,2]*uvel/100000.0 #converting to km/s
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

        return fig, ax, scatter, cbar

    ######################################################################
    #                        CLOUD-BASED FUNCTIONS                       #
    ######################################################################

    #Detecting cloud centered in the simulation box of cloud_size and having
    #a maximum temperature of cloud_Tmax
    def detect_cloud(self, cloud_size=600, cloud_Tmax=30, verbose=False):

        if(verbose):
            print("FIESTA >> Detecting cloud in ArepoVoronoiGrid {}...".format(self))
        
        #CREATE VARIABLES FOR THE CLOUD IN THE GRID
        #
        self.cloud_size = int
        self.cloud_Tmax = float
        #
        self.cloud_ids = []
        self.non_cloud_ids = []
        self.cloud_ntot = int
        #
        self.cloud_pos = []  #in code units
        self.cloud_vel = []  #in code units
        self.cloud_mass = []  #in code units
        self.cloud_chem = []  #in code units
        self.cloud_rho = []  #in code units
        self.cloud_utherm = []  #in code units
        #
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
        temps = self.temperature() #For checking temperature of cell
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
        self.cloud_ndensity = self.ndensity()[self.cloud_ids]
        self.cloud_temperature = self.temperature()[self.cloud_ids]

    #TODO: Change virial_ratio Q to virial_ratio alpha_vir?

    #Updating velocity in the cloud (requires a turbulent velocity field input)    
    def update_cloud_velocity(self, tvg=None, virial_ratio=0.5, verbose=False):

        #CREATING INTERPOLATION FUNCTION
        if(verbose):
            print("FIESTA >> Started creating interpolation functions for TurbulentVelocityGrid {}...".format(tvg))
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
            print("FIESTA >> Completed creating interpolation functions for TurbulentVelocityGrid {}.".format(tvg))

        #CREATING VELOCITY ARRAY
        if(verbose):
            print("FIESTA >> Started creating velocities for ArepoVoronoiGrid {}...".format(self))
        #Filling with all zero velocity first
        vel_new = np.zeros((self.ntot,3)) 
        vx_s = interp_x(self.cloud_pos)
        vy_s = interp_y(self.cloud_pos)
        vz_s = interp_z(self.cloud_pos)
        vel_new[self.cloud_ids] = np.array([vx_s,vy_s,vz_s]).T
        if(verbose):
            print("FIESTA >> Completed creating velocities for ArepoVoronoiGrid {}.".format(self))
        #Note that the velocity array is in arbitrary units so will have to be scaled    

        #SCALING VELOCITIES TO A SPECIFIC VIRIAL RATIO
        if(verbose):
            print("FIESTA >> Started scaling velocities to satisfy virial ratio {}...".format(virial_ratio))
        #Finding the scaling factor first
        E_k = calc_kinetic_energy(self.cloud_mass, vel_new[self.cloud_ids])
        E_g = calc_gravitational_potential_energy(self.cloud_mass, self.cloud_rho)
        Q = E_k / E_g  #Virial ratio (0.5 for virial equilibrium)
        scaling = np.sqrt(virial_ratio/Q)
        if(verbose):
            print("FIESTA >> Note: the scaling factor is {}.".format(scaling))
        print("NOTE: The scaling factor is "+str(scaling)+".")
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
    
    ######################################################################
    #                           CONSTRUCTORS                             #
    ######################################################################
                
    def __init__(self, file_path=None, AREPO_bounds=None, verbose=False):
        

        ######################################################################
        #                              Vars                                  #
        ######################################################################
        
        #Data properties
        self.file_path = str
        self.density =[[[]]]      #3D array
        self.nx = int             #number of data-points along x
        self.ny = int             #number of data-points along y
        self.nz = int             #number of data-points along z
        self.ndensity = [[[]]]    #3D array, units: cm^-3

        
        #Scale-related stuff
        self.scale = str          #"pixel" or "AREPO"
        self.xmin = float         #minimum x value, units: Euclidean OR AREPO
        self.xmax = float         #maximum x value, units: Euclidean OR AREPO 
        self.ymin = float         #minimum y value, units: Euclidean OR AREPO
        self.ymax = float         #maximum y value, units: Euclidean OR AREPO 
        self.zmin = float         #minimum z value, units: Euclidean OR AREPO
        self.zmax = float         #maximum z value, units: Euclidean OR AREPO 
        self.xlength = float      #length of one grid cube along x, units: Euclidean OR AREPO
        self.ylength = float      #length of one grid cube along y, units: Euclidean OR AREPO
        self.zlength = float      #length of one grid cube along z, units: Euclidean OR AREPO

        ######################################################################
        #                              Init                                  #
        ######################################################################

        #Initialize everything
        if file_path is not None:
            self.read_from_binary(file_path, verbose)

            if AREPO_bounds is not None:
                self.set_scale("AREPO", AREPO_bounds)

        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################

    #Reading in the grid from a binary file
    def read_from_binary(self, file_path, verbose=False):

        if(verbose):
            print("FIESTA >> Loading AREPO Cubic grid from \"{}\" ...".format(file_path))
            
        self.file_path = file_path
        #Read 3D data
        self.density = rsnap.read_grid(self.file_path)
        #Save dimensions
        self.nx, self.ny, self.nz = self.density.shape
        #Calculate ndensity for ease of use
        self.ndensity = calc_number_density(self.density)

        #Default scale
        self.set_scale("Euclidean")

        if(verbose):
            print("FIESTA >> Completed loading AREPO Cubic grid from \"{}\"".format(file_path))

    #Setting a scale for the grid. Useful!
    def set_scale(self, scale, AREPO_bounds=None):

        if(scale.upper()=="AREPO".upper()):
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

        elif(scale.lower()=="Euclidean".lower()):
            self.scale = "Euclidean"
            self.xmin, self.xmax = (0, self.nx)
            self.ymin, self.ymax = (0, self.ny)
            self.zmin, self.zmax = (0, self.nz)
            self.xlength = 1
            self.ylength = 1
            self.zlength = 1   

        else:
            print("Incorrect scale option!") 
    
    #Writing the grid into a FITS file
    def write_to_FITS(self, FITS_file_path, verbose=False):
        
        if(verbose):
            print("FIESTA >> Writing AREPO Cubic grid to FITS at \"{}\" ...".format(FITS_file_path))
            
        hdu = fits.PrimaryHDU(data=self.density)
        hdu.writeto(FITS_file_path, overwrite=True)

        if(verbose):
            print("FIESTA >> Completed writing AREPO Cubic grid to FITS at \"{}\"".format(FITS_file_path))

        #if(check):
        #    print("NOTE: Printing data from the FITS file below...")
        #    data = fits.open(grid_FITS_file_path)[0].data
        #    print(data)
            
            
    #Writing a mask to the FITS file that masks things outside (xmin,xmax), (ymin,ymax) and (zmin,zmax)
    def write_mask_to_FITS(self, mask_FITS_file_path, xmin, xmax, ymin, ymax, zmin, zmax, verbose=False):

        if(verbose):
            print("FIESTA >> Writing mask to FITS at \"{}\" ...".format(mask_FITS_file_path))
            
        mask_data = np.full((self.nx,self.ny,self.nz), 1, dtype=int)
        mask_data[xmin:xmax,ymin:ymax,zmin:zmax] = 0
        hdu = fits.PrimaryHDU(data=mask_data)
        hdu.writeto(mask_FITS_file_path, overwrite=True)

        if(verbose):
            print("FIESTA >> Completed writing mask to FITS at \"{}\"".format(mask_FITS_file_path))

        #if(check):
        #    print("NOTE: Printing mask from the FITS file below...")
        #    data = fits.open(mask_FITS_file_path)[0].data
        #    print(data[xmin:xmin+5,ymin:ymin+5,zmin:zmin+5])
    
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
    
    def plot_grid(self, 
                  log=False,
                  cmap=None,
                  mask_below=None, 
                  network=None,
                  avg=None,
                  min_sink_mass=0.0,
                  save=None,
                  **kwargs):

        print("FIESTA >> Plotting 3D ArepoCubicGrid {}...".format(self))

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
        if cmap is not None:
            cmap1 = cmap
        else:
            cmap1 = mpl.cm.get_cmap('viridis')
        cmap2 = mpl.cm.get_cmap('gray')
        alpha1 = 0.1
        alpha2 = 0.4

        data = self.ndensity

        #Masking below a certain threshold
        if mask_below is not None:
            data = np.ma.masked_array(data,mask=data<mask_below)
           
        #Taking logarithm
        if(log):
            data = np.log10(data, out=np.zeros_like(data), where=(data>0))

        #CONTOUR PLOTS PROJECTIONS
        xr = np.linspace(self.xmin, self.xmax, num=self.nx, endpoint=True)
        yr = np.linspace(self.ymin, self.ymax, num=self.ny, endpoint=True)
        zr = np.linspace(self.zmin, self.zmax, num=self.nz, endpoint=True)

        #x projection
        y, z = np.meshgrid(yr,zr)
        contour_x = np.sum(data,axis=0).T #The transpose is CRUCIAL due to how contourf works!
        ax.contourf(contour_x, y, z, zdir='x', levels=20, offset=self.xmin, cmap=cmap2, alpha=alpha2)
        
        #y projection
        x, z = np.meshgrid(xr,zr)
        contour_y = np.sum(data,axis=1).T #The transpose is CRUCIAL due to how contourf works!
        ax.contourf(x, contour_y, z, zdir='y', levels=20, offset=self.ymax, cmap=cmap2, alpha=alpha2)
        
        #z projection
        x, y = np.meshgrid(xr,yr)
        contour_z = np.sum(data,axis=2).T #The transpose is CRUCIAL due to how contourf works!
        ax.contourf(x, y, contour_z, zdir='z', levels=20, offset=self.zmin, cmap=cmap2, alpha=alpha2)


        #3D PLOT
        X, Y, Z = np.meshgrid(xr,yr,zr)
        plot_data = np.swapaxes(data,0,1).flatten() #The swapping of axes is how matplotlib works. NECESSARY!
        scatter = ax.scatter(X, Y, Z, c=plot_data, cmap=cmap1, alpha=alpha1, s=1)


        #FILAMENT PLOT
        if network is not None:
            if(network.scale != self.scale):
                raise ValueError("FIESTA >> Network and ArepoCubicGrid scales don't match!")
            else:   
                for fil in network.fils:
                    x = fil.samps[:,0]
                    y = fil.samps[:,1]
                    z = fil.samps[:,2]
                    ax.scatter(x[0], y[0], z[0], c='black', s=2, zorder=2)
                    ax.scatter(x[-1], y[-1], z[-1], c='black', s=2, zorder=2)
                    ax.plot(x, y, z, linewidth=1, c='red')

        #SINK PLOT
        if avg is not None:
            if(self.scale != "AREPO"):
                raise ValueError("FIESTA >> ArepoCubicGrid needs to be in AREPO scale to plot the sinks!")
            else:
                sink_mass = avg.mass[avg.sink_ids]
                sink_pos = avg.pos[avg.sink_ids]
                #Only plot above a certain minimum sink mass (in solar masses)
                trunc_sink_mass = sink_mass[sink_mass>min_sink_mass]
                trunc_sink_pos = sink_pos[sink_mass>min_sink_mass]
                x = trunc_sink_pos[:,0]
                y = trunc_sink_pos[:,1]
                z = trunc_sink_pos[:,2]
                s = 2.0 + 5.0*trunc_sink_mass/trunc_sink_mass.max()
                ax.scatter(x, y, z, s=2, c='red', zorder=2)

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

        return fig, ax
        
    ######################################################################
    
    def plot_slice(self, 
                   x=None, 
                   y=None, 
                   z=None,
                   log=False,
                   cmap=None,
                   save=None,
                   **kwargs):

        print("FIESTA >> Plotting 2D slice ArepoCubicGrid {}...".format(self))

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

        if cmap is None:
            cmap = plt.cm.magma
        else:
            cmap = mpl.cm.get_cmap(cmap)

        data = self.ndensity

        #Taking logarithm
        if(log):
            cmap.set_bad((0,0,0))
            norm = mpl.colors.LogNorm()
        else:
            norm = mpl.colors.Normalize()
    
        #The transpose is CRUCIAL due to how imshow works!
        if x is not None:
            plot_data = data[x,:,:].T
            extent=[self.ymin,self.ymax,self.zmin,self.zmax]
        elif y is not None:
            plot_data = data[:,y,:].T
            extent=[self.xmin,self.xmax,self.zmin,self.zmax]
        elif z is not None:
            plot_data = data[:,:,z].T
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

        return fig, ax, imshow, cbar

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

        if cmap is None:
            cmap = plt.cm.gist_gray
        else:
            cmap = mpl.cm.get_cmap(cmap)

        data = self.ndensity

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