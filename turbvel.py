######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

######################################################################
#                            LIBRARIES                               #
######################################################################

import functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

######################################################################
#                      TURBULENT VELOCITY GRID                       #
######################################################################

class TurbulentVelocityGrid:
    
    ######################################################################
    #                           CONSTRUCTOR                              #
    ######################################################################
    
    def __init__(self, file_path=None, verbose=False, check=False):
        
        #FILE PATH
        self.file_path = str
        
        #CREATE VARIABLES
        #Raw data
        self.data = []
        #Extracted data
        self.size = int
        self.vx = [[[]]]
        self.vy = [[[]]]
        self.vz = [[[]]]
        
        #SET VARIABLES
        if file_path is not None:
            self.read_file(file_path, verbose, check)
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
    
    #Reading in data from the turbulent velocity binary file
    def read_file(self, file_path, verbose=False, check=False):

        #READING THE DATA
        #Note: the create-turbulence-field code has to be run with parameter "flt" for data types to match
        #between the numpy reading done below and the C++ code output
        if(verbose):
            print("FIESTA >> Started reading turbulent velocity field from \"{}\" ...".format(file_path))
        self.file_path = file_path
        velocity_data = np.fromfile(self.file_path,'float32')
        if(verbose):
            print("FIESTA >> Completed reading turbulent velocity field from \"{}\"".format(file_path))
        
        #SPLITTING DATA INTO INTO VELOCITY COMPONENTS
        self.size = int(np.cbrt(len(velocity_data)/3))
        velocity_arrays = np.array_split(velocity_data, 3)
        n = self.size
        vx_flat= velocity_arrays[0]
        self.vx = np.reshape(vx_flat,(n,n,n),order='C')
        vy_flat = velocity_arrays[1]
        self.vy = np.reshape(vy_flat,(n,n,n),order='C')
        vz_flat = velocity_arrays[2]
        self.vz = np.reshape(vz_flat,(n,n,n),order='C')
        
        #CHECKING THE READING OF DATA
        if(check):
            print("FIESTA >> Printing data for checking below...")
            #Compare these with the dim/2 slices from the C++ code
            #Note that the slices are all for velocity_z and correspond to dim/2 slices along each x,y,z axis
            print("FIESTA >> Printing vz dim/2 x-slice now")
            print(self.vz[:,:,int(n/2)])
            print("FIESTA >> Printing vz dim/2 y-slice now")
            print(self.vz[:,int(n/2),:])
            print("FIESTA >> Printing vz dim/2 z-slice now")
            print(self.vz[int(n/2),:,:])
                  
    
    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
                  
    #Plotting heat map of the velocity components
    def plot_projection(self,
                        projection='z',
                        cmap=None,
                        save=None,
                        **kwargs):

        print("FIESTA >> Plotting {} component of turbulent velocity field {}...".format(projection,self))

        #Figure properties
        if cmap is None:
            cmap = plt.cm.RdBu

        #Main figure
        fig = plt.figure(figsize=(10,10))
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
        ax.set_xlabel(r"x",fontsize=15,labelpad=5)
        ax.set_ylabel(r"y",fontsize=15,labelpad=5)
        ax.set_zlabel(r"z",fontsize=15,labelpad=5)
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

        def cartesian_product_broadcasted(*arrays):
            """
            http://stackoverflow.com/a/11146645/190597 (senderle)
            """
            broadcastable = np.ix_(*arrays)
            broadcasted = np.broadcast_arrays(*broadcastable)
            dtype = np.result_type(*arrays)
            rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
            out = np.empty(rows * cols, dtype=dtype)
            start, end = 0, rows
            for a in broadcasted:
                out[start:end] = a.reshape(-1)
                start, end = end, end + rows
            return out.reshape(cols, rows).T
                  
        #Only plotting the outer faces of the velocity grid since N^3 scatter graph is extremely computationally heavy
        n = self.size
        x, y, z = cartesian_product_broadcasted(*[np.arange(n, dtype='int16')]*3).T
        mask = ((x == 0) | (x == n-1) | (y == 0) | (y == n-1) | (z == 0) | (z == n-1))
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if(projection.lower()=='x'):
            vel = self.vx.ravel()[mask]
        elif(projection.lower()=='y'):
            vel = self.vy.ravel()[mask]
        elif(projection.lower()=='z'):
            vel = self.vz.ravel()[mask]
        else:
            raise ValueError("FIESTA >> Invalid projection.")

        scatter = ax.scatter(x,y,z,c=vel,cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.1)
        cbar.set_label(r'Velocity projection ${}$'.format(projection), size=15, color='black')
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.outline.set_edgecolor('black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        ############### Plotting end ################

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig, ax, scatter, cbar