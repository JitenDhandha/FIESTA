######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains the class to deal with cubic turbulent velocity
grids. Initially written to work with a field generator written by 
Philipp Girichidis, but can be modified for general use.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import functools
#Numpy
import numpy as np
#Astropy
from astropy import units as u
#Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#FIESTA
from fiesta import _utils as utils

######################################################################
#                      TURBULENT VELOCITY GRID                       #
######################################################################

class TurbulentVelocityGrid:

    """
    Main class that deals with cubic turbulent velocity grids.

    Attributes
    ----------
    
    size : `int`
        Size of one axis of the turbulent velocity field where ``size**3`` encompasses the whole cube.

    vx : `~astropy.units.Quantity`
        3D grid of :math:`x`-component of velocity: :math:`v_x`.

    vy : `~astropy.units.Quantity`
        3D grid of :math:`y`-component of velocity: :math:`v_y`.

    vz : `~astropy.units.Quantity`
        3D grid of :math:`z`-component of velocity: :math:`v_z`.

    Parameters
    ----------

    file_path : `str`
        File path of the turbulent velocity grid.

    verbose : `bool`, optional
        If ``True``, prints output during file reading. Default value is ``False``.
    
    check : `bool`, optional
        If ``True``, prints middle slices of `~fiesta.turbvel.TurbulentVelocityGrid.vz`.
        Default value is ``False``.
    
    """
    
    def __init__(self, file_path, verbose=False, check=False):
        
        #Note: the create-turbulence-field code has to be run with parameter "flt" for data types to match
        #between the numpy reading done below and the C++ code output
        if(verbose):
            print(utils._prestring() + "Started reading turbulent velocity field from \"{}\" ...".format(file_path))
        velocity_data = np.fromfile(file_path,'float32')
        if(verbose):
            print(utils._prestring() + "Completed reading turbulent velocity field from \"{}\"".format(file_path))
        
        #Splitting data into velocity components
        self.size = u.Quantity(np.cbrt(len(velocity_data)/3), copy=False, dtype=int, unit=u.pix)
        if(verbose):
            print(utils._prestring() + "The grid is of size {}**3".format(self.size))
        velocity_arrays = np.array_split(velocity_data, 3)
        n = self.size.value
        vx_flat= velocity_arrays[0]
        self.vx = np.reshape(vx_flat,(n,n,n),order='C') << u.dimensionless_unscaled
        vy_flat = velocity_arrays[1]
        self.vy = np.reshape(vy_flat,(n,n,n),order='C') << u.dimensionless_unscaled
        vz_flat = velocity_arrays[2]
        self.vz = np.reshape(vz_flat,(n,n,n),order='C') << u.dimensionless_unscaled
        
        #Checking the reading of the data
        if(check):
            print(utils._prestring() + "Printing data for checking below...")
            #Compare these with the dim/2 slices from the C++ code
            #Note that the slices are all for vz and correspond to dim/2 slices along each x,y,z axis
            print(utils._prestring() + "Printing vz dim/2 x-slice now")
            print(self.vz[:,:,int(n/2)])
            print(utils._prestring() + "Printing vz dim/2 y-slice now")
            print(self.vz[:,int(n/2),:])
            print(utils._prestring() + "Printing vz dim/2 z-slice now")
            print(self.vz[int(n/2),:,:])

    ######################################################################
    #                             GETTERS                                #
    ######################################################################

    def get_size(self):
        """

        Returns `~fiesta.turbvel.TurbulentVelocityGrid.size`.

        """
        return self.size

    def get_vx(self):
        """

        Returns `~fiesta.turbvel.TurbulentVelocityGrid.vx`.

        """
        return self.vx

    def get_vy(self):
        """

        Returns `~fiesta.turbvel.TurbulentVelocityGrid.vy`.

        """
        return self.vy

    def get_vz(self):
        """

        Returns `~fiesta.turbvel.TurbulentVelocityGrid.vz`.

        """
        return self.vz

    ######################################################################
    #                         PLOTTING FUNCTIONS                         #
    ######################################################################
                  
    def plot_velocity_component(self,
                                component='z',
                                outer_faces_only=True,
                                cmap='RdBu',
                                save=None,
                                **kwargs):

        """
        
        Plot a 3D heatmap of the velocities as a cube.

        Parameters
        ----------
        
        component : `str`, optional
            Velocity component to plot: ``x`` for :math:`v_x`, ``y`` 
            for :math:`v_y` or ``z`` (default) for :math:`v_z`.

        outer_faces_only : `bool`, optional
            If ``True`` (default), plots only the outer faces of the velocity grid
            since a ``size**3`` scatter plot is very computationally heavy. 
            If ``False``, plots the whole grid.

        cmap : `str` or `~matplotlib.colors.Colormap`, optional
            Colormap of the plot. Default value is ``'RdBu'``.

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
        n = self.size.value
        vx = self.vx.value
        vy = self.vy.value
        vz = self.vz.value

        #Main figure
        fig = plt.figure(figsize=(10,10))
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
        ax.set_xlabel(r"$x$ [pix]",fontsize=15,labelpad=5)
        ax.set_ylabel(r"$y$ [pix]",fontsize=15,labelpad=5)
        ax.set_zlabel(r"$z$ [pix]",fontsize=15,labelpad=5)
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

        #Projection
        if(component=='x' or component=='X'):
            vel = vx.flatten()
        elif(component=='y' or component=='Y'):
            vel = vy.flatten()
        elif(component=='z' or component=='Z'):
            vel = vz.flatten()
        else:
            raise ValueError(utils._prestring() + "Invalid component.")

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
        
        x, y, z = cartesian_product_broadcasted(*[np.arange(n, dtype='int16')]*3).T

        if(outer_faces_only):
            mask = ((x == 0) | (x == n-1) | (y == 0) | (y == n-1) | (z == 0) | (z == n-1))
            x = x[mask]
            y = y[mask]
            z = z[mask]
            vel = vel[mask]

        scatter = ax.scatter(x, y, z, c=vel, cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.038, pad=0.1)
        cbar.set_label(r'Velocity component ${}$'.format(component), size=15, color='black')
        cbar.ax.tick_params(labelsize=15, color='black')
        cbar.outline.set_edgecolor('black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        ############### Plotting end ################

        #Axes limits
        if "xlim" in kwargs:
            ax.set_xlim(**kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(**kwargs["ylim"])
        if "zlim" in kwargs:
            ax.set_zlim(**kwargs["zlim"])

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)

        return fig