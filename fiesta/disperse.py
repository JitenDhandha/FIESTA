######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains classes for dealing with filamentary networks 
from DisPerSE, and a variety of very useful algorithms such as
readjusting filament spines, characterizing filament lengths and masses,
identifying junctions/hubs in the network and more!
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import copy
#Numpy
import numpy as np
#Scipy
from scipy import stats
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
#Astropy
from astropy import units as u
#Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#FIESTA
from fiesta import units as ufi
from fiesta import _utils as utils

######################################################################
#                        INTERNAL FUNCTIONS                          #
######################################################################

def _magnitude(v):
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)

def _normalize(v):
    return v/_magnitude(v)

#Returns the rotation matrix for the angle between the given vector and [0,0,1]
def _get_rotation_matrix(v):
    n = _normalize(v)
    theta = np.arccos(n[2])
    u = _normalize(np.cross(n,[0,0,1]))
    rot = Rotation.from_rotvec(theta * u)
    return rot

def _get_spheres(points, grid, SPHERE_RADIUS, grid_tree=None):

        #Initialize arrays
        all_sphere_ids = []
        all_sphere_points = []

        #STEP 1: Creating the KDTree from grid (if not pre-evaulated and passed)
        if grid_tree is None:
            grid_tree = KDTree(grid)

        #STEP 2: Querying all cells within a sphere of radius SPHERE_RADIUS
        all_sphere_ids = grid_tree.query_ball_point(points, r=SPHERE_RADIUS)
        for sphere_ids in all_sphere_ids:
            all_sphere_points.append(grid[sphere_ids])

        return all_sphere_ids, all_sphere_points

def _get_cylinders(points, grid, CYL_RADIUS):

        ids = np.arange(len(grid),dtype=int)

        #Initialize arrays
        vectors = []
        rotation_matrices =[]
        all_cylinder_ids = []
        all_cylinder_points = []
        
        #STEP 1: Find vectors between successive points
        vectors = np.diff(points,axis=0)
        
        for i in range(len(points)-1):

            #STEP 2: Extract a cubical region of side 4*CYL_RADIUS around point
            SEARCH_RADIUS = 2*CYL_RADIUS
            REGION_MASK = ((grid[:,0] > points[i,0]-SEARCH_RADIUS) &
                           (grid[:,0] < points[i,0]+SEARCH_RADIUS) &
                           (grid[:,1] > points[i,1]-SEARCH_RADIUS) &
                           (grid[:,1] < points[i,1]+SEARCH_RADIUS) &
                           (grid[:,2] > points[i,2]-SEARCH_RADIUS) &
                           (grid[:,2] < points[i,2]+SEARCH_RADIUS))
            region_points = grid[REGION_MASK]
            region_ids = ids[REGION_MASK]

            #STEP 3: Translate and rotate the region and filament to the origin
            rot = _get_rotation_matrix(vectors[i])
            points_at_origin = rot.apply(points-points[i])
            region_at_origin = rot.apply(region_points-points[i])
            rotation_matrices.append(rot)
        
            #STEP 4: Querying all cells within a cylinder of radius CYL_RADIUS
            CYLINDER_MASK = ((region_at_origin[:,0]**2 + region_at_origin[:,1]**2 <= CYL_RADIUS**2) &
                             (region_at_origin[:,2] > min(points_at_origin[i,2],points_at_origin[i+1,2])) &
                             (region_at_origin[:,2] < max(points_at_origin[i,2],points_at_origin[i+1,2])))
            cylinder_ids = region_ids[CYLINDER_MASK]
            cylinder_points = region_points[CYLINDER_MASK]
            
            #Adding to array
            all_cylinder_ids.append(cylinder_ids)
            all_cylinder_points.append(cylinder_points)

        return vectors, rotation_matrices, all_cylinder_ids, all_cylinder_points

######################################################################
#                         CRITICAL POINT                             #
######################################################################

class CriticalPoint:
    """

    Class to deal with critical points output by DisPerSE.

    Attributes
    ----------
        
    idx : `int`
        Unique index for the critical point in the `~fiesta.disperse.Network`.

    pos : `~astropy.units.Quantity`
        Position :math:`(x,y,z)` of the critical point.
        Defaults to units of ``u.pix``.
    
    nfil : `int`
        Number of filaments connected to the critical point.

    scale : `str`
        Either ``pixel`` if the grid is not scaled to the source
        AREPO snapshot, else ``physical``.
    
    """
    def __init__(self):
        self.idx = None
        self.pos = None
        self.nfil = None
        self.scale = None
    
######################################################################
#                              FILAMENT                              #
######################################################################

class Filament:
    """

    Class to deal with filaments output by DisPerSE.

    Attributes
    ----------
        
    idx : `int`
        Unique index for the filament in the `~fiesta.disperse.Network`.

    cps : pair of `~fiesta.disperse.CriticalPoint`
        Critical point at the start and end of the filament.
    
    nsamp : `int`
        Number of sampling points of the filament.

    samps : `~astropy.units.Quantity`
        Array of position :math:`(x,y,z)` of each sampling point 
        of the filament. Defaults to units of ``u.pix``.

    scale : `str`
        Either ``pixel`` if the grid is not scaled to the source
        AREPO snapshot, else ``physical``.

    length : `~astropy.units.Quantity`
        Length of the filament. Set only after calling
        `~fiesta.disperse.Filament.calc_length`.
    
    radius : `~astropy.units.Quantity`
        Radius of the filament. Fixed value set during
        `~fiesta.disperse.Filament.characterize_filament`.

    mass : `~astropy.units.Quantity`
        Mass of the filament. Set only after calling
        `~fiesta.disperse.Filament.calc_mass`.

    arepo_ids : `list` of `int`'s
        List of indices of `~fiesta.arepo.ArepoVoronoiGrid` cells
        that are "part" of the filament. Set only after calling
        `~fiesta.disperse.Filament.characterize_filament`.
    
    """
    def __init__(self):
        
        #Core variables
        self.idx = None
        self.cps = None
        self.nsamp = None
        self.samps = None
        self.scale = None
        #NOTE: The samps contain the critical points already (as first and last element)
        
        #Tools
        self._filfunc = None
        self.arepo_ids = None
        
        #Properties
        self.length = None
        self.radius = None
        self.mass = None
    
    ######################################################################
    #                               filfunc                             #
    ######################################################################
    
    #The filfunc is a function that parametrizes the 1D curve in 3D space:
    #f(s) = (x,y,z) where s = [0,1] -- the normalized distance along the filament.
    
    def _set_filfunc(self):
            
        #Since you can always fit a N-1 degree polynomial to N points
        if(self.nsamp>3):
            degree = 3
        else:
            degree = self.nsamp-1
        #Note the smoothing s=0 so that the fit curve passes through all sampling points
        if(self.scale=="physical"):
            internal_unit = u.cm
        elif(self.scale=="pixel"):
            internal_unit = u.pix
        points = self.samps.to_value(internal_unit)
        func, _ = splprep(points.T,s=0,k=degree)

        self._filfunc = func
    
    def get_points(self, s, unit=None):
        """

        Function to sample any point on the filament spine.

        A *filament function* is one that parameterizes
        the 1D filament curve in 3D space, defined as
        :math:`\mathbf{f}(s) = (x,y,z)` where :math:`s \in [0,1]`
        is the normalized distance of a point along the filament spine
        and :math:`(x,y,z)` is the position vector of that point in space.
        
        Since DisPerSE only outputs a discrete set of sampling points,
        fitting :math:`\mathbf{f}(s)` allows one to sample the filament
        continuously or as finely as required. In this implementation,
        the function is a cubic B-spline fit using `scipy.interpolate.splprep`, 
        with zero-smoothing to ensure it passes through input points.

        Attributes
        ----------
            
        s : `list` or `~numpy.ndarray` of `float`'s
            Array of normalized distances along the filament
            :math:`s \in [0,1]`.

        unit : `~astropy.units.Unit`
            Unit to output the points in.
            If ``None`` (default), takes the value 
            ``u.pix`` if ``scale='pixel'``, else 
            ``u.cm`` if ``scale='physical'``.

        Returns
        ----------

        points : `~astropy.units.Quantity`
            Array of 3d position vectors corresponding 
            to the points at the given normalized distances
            :math:`s` along the filament.
        
        """
        
        if(self.scale=="physical"):
            internal_unit = u.cm
        elif(self.scale=="pixel"):
            internal_unit = u.pix
        if unit is None:
            unit = internal_unit
        utils.check_unit(unit, internal_unit)
        points = (splev(s,self._filfunc) << internal_unit).T
        return points.to(unit)

    ######################################################################
    #                                Getters                             #
    ######################################################################

    def get_length(self, unit=None):
        """

        Returns `~fiesta.disperse.Filament.length`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in.
            If ``None`` (default), takes the value 
            ``u.pix`` if ``scale='pixel'``, else 
            ``u.cm`` if ``scale='physical'``.

        """
        if self.length is None:
            raise ValueError(utils._prestring() + "Length needs to be calculated first using calc_length() function.")
        else:
            if(self.scale=="physical"):
                internal_unit = u.cm
            elif(self.scale=="pixel"):
                internal_unit = u.pix
            if unit is None:
                unit = internal_unit
            utils.check_unit(unit, internal_unit)
            return self.length.to(unit)

    def get_arepo_ids(self):
        """

        Returns `~fiesta.disperse.Filament.arepo_ids`. 

        """
        if self.arepo_ids is None:
            raise ValueError(utils._prestring() + "Filament needs to be characterized first using characterize_filament() function.")
        else:
            return self.arepo_ids

    def get_radius(self, unit=u.cm):
        """

        Returns `~fiesta.disperse.Filament.radius`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.cm``.

        """
        if self.radius is None:
            raise ValueError(utils._prestring() + "Filament needs to be characterized first using characterize_filament() function.")
        else:
            utils.check_unit(unit, u.cm)
            return self.radius.to(unit)

    def get_mass(self, unit=u.g):
        """

        Returns `~fiesta.disperse.Filament.mass`.

        Parameters
        ----------

        unit : `~astropy.units.Unit`, optional
            Unit to output the result in. Default value is ``u.g``.

        """
        if self.mass is None:
            raise ValueError(utils._prestring() + "Mass needs to be calculated first using calc_mass() function.")
        else:
            utils.check_unit(unit, u.g)
            return self.mass.to(unit)
    
    ######################################################################
    #                       ANALYZING THE FILAMENT                       #
    ######################################################################
            
    def correct_spine(self, 
                      avg, 
                      sphere_radius, 
                      method='center of density', 
                      verbose=False, 
                      plot_correction=False, 
                      length_unit=u.cm):

        """

        Function to correct the spine of a filament, based
        on source `~fiesta.arepo.ArepoVoronoiGrid`.

        Since DisPerSE works on a regular cubic grid
        (i.e., `~fiesta.arepo.ArepoCubicGrid`) which has a finite
        resolution that washes over local overdensities, it can be
        useful to correct the spine of the filament, once identified
        on the coarser regular grid, using the original finer 
        Voronoi grid.

        The correction is made by querying all AREPO cells within a 
        given ``sphere_radius`` of each sampling point, and
        then shifting the sampling point based on the 
        ``method`` of correction.

        Note that the filament must have ``scale='physical'`` to
        use this functionality. The function updates the 
        `~fiesta.disperse.Filament.samps` and the *filament
        function* internally!

        Attributes
        ----------
            
        avg : `~fiesta.arepo.ArepoVoronoiGrid`
            Source `~fiesta.arepo.ArepoVoronoiGrid` to use for
            correcting the spine.

        sphere_radius : `~astropy.units.Quantity`
            Spherical radius of correction for each sampling point.

        method : `str`, optional
            Either ``'max density'`` or ``'center of density'`` (default).
            The former shifts the sampling point to the densest cell
            within the given spherical radius, while the latter
            calculates and shifts it to the center of density
            in the sphere.

        verbose : `bool`, optional
            If ``True``, prints output during spine correction.
            Default value is ``False``.
        
        plot_correction : `bool`, optional
            If ``True``, plots filament before and after the 
            spine correction. Useful for checks! Default
            value is ``False``.

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use for the plot.
            Default value is ``u.cm``.
        
        """

        if(verbose):
            print(utils._prestring() + "Correcting Filament spine for idx = {}".format(self.idx))

        if(self.scale!="physical"):
            raise ValueError(utils._prestring() + "Filament must be in \"physical\" units.")

        #AREPO data variables
        AREPO_mass = avg.mass[avg.gas_ids].to_value(u.g)
        AREPO_ndensity = avg.get_ndensity().to_value(u.cm**-3)
        AREPO_pos = avg.pos[avg.gas_ids].to_value(u.cm)
        if avg._tree is None:
            avg._tree = KDTree(AREPO_pos)
        AREPO_tree = avg._tree

        #Getting points
        old_points = self.samps
        old_points_val = self.samps.to_value(u.cm)
        old_spline = self.get_points(np.linspace(0,1,self.nsamp*100,endpoint=True))
        utils.check_quantity(sphere_radius, u.cm, "sphere_radius")
        sphere_radius = sphere_radius.to_value(u.cm)

        #Initialize arrays
        all_sphere_ids, all_sphere_points = _get_spheres(old_points_val, AREPO_pos, sphere_radius, AREPO_tree)
        new_filament = []

        for i in range(len(old_points_val)):

            if all_sphere_points[i].size==0:

                new_filament.append(old_points_val[i])
                continue

            else:

                if(method.lower()=='max density'):
                    corr_idx = np.argmax(AREPO_ndensity[all_sphere_ids[i]])
                    corr_point = all_sphere_points[i][corr_idx]
                elif(method.lower()=='center of density'):
                    corr_point = np.average(all_sphere_points[i],
                                            weights=AREPO_ndensity[all_sphere_ids[i]],
                                            axis=0)
                else:
                    raise ValueError(utils._prestring() + "Invalid method of spine correction.")
                
                corr_point = list(corr_point)
                if corr_point not in new_filament:
                    new_filament.append(corr_point)
        
        new_filament = u.Quantity(new_filament,unit=u.cm).to(ufi.AREPO_LENGTH)

        #Saving the corrected filament
        self.samps = new_filament
        self.nsamp = len(new_filament)
        self._set_filfunc()
        new_spline = self.get_points(np.linspace(0,1,self.nsamp*100,endpoint=True))

        #Plot the correction
        if plot_correction:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection='3d')
            utils.check_unit(length_unit, u.cm)
            ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
            ax.set_ylabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
            ax.set_zlabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
            ax.scatter(*old_points.to_value(length_unit).T, color='black', s=5)
            ax.plot(*old_spline.to_value(length_unit).T, color='grey', linewidth=2)
            ax.scatter(*self.samps.to_value(length_unit).T, color='red', s=5)
            ax.plot(*new_spline.to_value(length_unit).T, color='gold', linewidth=2)
            for sphere_points in all_sphere_points:
                sphere_points = sphere_points << u.cm
                if(not sphere_points.size == 0):
                    ax.scatter(*sphere_points.to_value(length_unit).T, alpha=0.02)

    def characterize_filament(self, 
                              avg, 
                              cylinder_radius, 
                              npoints=None, 
                              verbose=False, 
                              plot_filament=False, 
                              length_unit=u.cm):

        """

        Function to query all simulation cells around the filament, 
        based on source `~fiesta.arepo.ArepoVoronoiGrid`.

        The function splits the filament into equidistant ``npoints``,
        joining them by cylinders of radius ``cylinder_radius``. All 
        AREPO cells within these cylinders are considered "part" of the 
        filament. The indices of the cells are then stored as the instance
        variable `~fiesta.disperse.Filament.arepo_ids`, and the 
        ``cylinder_radius`` sets the instance variable
        `~fiesta.disperse.Filament.radius`.

        Note that double-counting is **not** corrected for in the indices.
        In order to do this, try `np.unique(np.hstack(arepo_ids))`.

        Attributes
        ----------
            
        avg : `~fiesta.arepo.ArepoVoronoiGrid`
            Source `~fiesta.arepo.ArepoVoronoiGrid` to use for
            characterizing the filament.

        cylinder_radius : `float` or `~astropy.units.Quantity`
            Radius of cylinder to query AREPO cells within.
            If not a `~astropy.units.Quantity`, assumed to be in 
            ``u.cm`` units.

        npoints : `int`, optional
            Number of points to split the filament into. If ``None``
            (default), then the sampling points 
            `~fiesta.disperse.Filament.samps` are used.

        verbose : `bool`, optional
            If ``True``, prints output during filament characterization.
            Default value is ``False``.
        
        plot_filament : `bool`, optional
            If ``True``, plots the filament along with all its AREPO cells
            Useful for checks! Default value is ``False``.

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use for the plot.
            Default value is ``u.cm``.
        
        """

        if(verbose):
            print(utils._prestring() + "Characterizing Filament for idx = {}".format(self.idx))

        if(self.scale!="physical"):
            raise ValueError(utils._prestring() + "Filament must be in \"physical\" units.")

        #AREPO data variables
        AREPO_pos = avg.pos[avg.gas_ids].to_value(u.cm)

        #Getting points
        if(npoints is None):
            npoints = self.nsamp
            points = self.samps
        else:
            points = self.get_points(np.linspace(0,1,npoints,endpoint=True))
        points_val = points.to_value(u.cm)
        utils.check_quantity(cylinder_radius, u.cm, "cylinder_radius")
        cylinder_radius = cylinder_radius.to_value(u.cm)
        self.radius = cylinder_radius

        #Initialize arrays
        vectors, rotation_matrices, all_cylinder_ids, all_cylinder_points = _get_cylinders(points_val, AREPO_pos, cylinder_radius)
        self.arepo_ids = all_cylinder_ids

        #Plot the filament
        if plot_filament:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection='3d')
            utils.check_unit(length_unit, u.cm)
            ax.set_xlabel(r"$x$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
            ax.set_ylabel(r"$y$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
            ax.set_zlabel(r"$z$ [{}] ".format(length_unit.to_string()),fontsize=15,labelpad=5)
            ax.scatter(*points.to_value(length_unit).T, color='black', s=5)
            spline = self.get_points(np.linspace(0,1,npoints,endpoint=True))
            ax.plot(*spline.to_value(length_unit).T, color='grey', linewidth=2)
            for cylinder_points in all_cylinder_points:
                cylinder_points = cylinder_points << u.cm
                if(not cylinder_points.size == 0):
                    ax.scatter(*cylinder_points.to_value(length_unit).T, alpha=0.02)


    ######################################################################
    #                        PHYSICAL PROPERTIES                         #
    ######################################################################

    def calc_length(self, verbose=False):
        """

        Calculates the length of the filament, either in pixels
        or physical units depending on 
        `~fiesta.disperse.Filament.scale`, by finely integrating
        the Euclidean distance between very close points on the filament.

        Attributes
        ----------

        verbose : `bool`, optional
            If ``True``, prints output during length calculation.
            Default value is ``False``.
        
        """

        if(verbose):
            print(utils._prestring() + "Calculating Filament length for idx = {}".format(self.idx))

        #Summing Euclidean distance between very close points
        spline = self.get_points(np.linspace(0,1,self.nsamp*500,endpoint=True))
        self.length = np.sqrt((np.diff(spline,axis=0) ** 2).sum(axis=1)).sum() << spline.unit

        if(verbose):
            print(utils._prestring() + "The length of the filament is {}".format(self.length))

    def calc_mass(self, avg, verbose=False):
        """

        Calculates the mass of the filament by summing the mass of all cells given by
        `~fiesta.disperse.Filament.arepo_ids` (corrected for double-counting). 
        
        Note that this requires calling `~fiesta.disperse.Filament.characterize_filament`
        first to set `~fiesta.disperse.Filament.arepo_ids`.

        Attributes
        ----------

        avg : `~fiesta.arepo.ArepoVoronoiGrid`
            Source `~fiesta.arepo.ArepoVoronoiGrid` to use for
            calculating mass of the filament.

        verbose : `bool`, optional
            If ``True``, prints output during mass calculation.
            Default value is ``False``.
        
        """

        if verbose:
            print(utils._prestring() + "Calculating Filament mass for idx = {}".format(self.idx))

        if self.arepo_ids is None:
            raise ValueError(utils._prestring() + "Filament needs to be characterized first using characterize_filament() function.")

        unique_ids = np.unique(np.hstack(self.arepo_ids))
        self.mass = np.sum(avg.mass[unique_ids])

        if(verbose):
            print(utils._prestring() + "The mass of the filament is {}".format(self.mass))

    def plot_density_profile(self, 
                             avg,
                             length_unit=u.cm,
                             ndensity_unit=u.cm**-3,
                             scatter=True,
                             contours=None,
                             save=None,
                             **kwargs):
        
        """
        
        Plot mass-weighted 2d histogram projection of the AREPO Voronoi grid 
        along an axis.

        Parameters
        ----------
        
        avg : `~fiesta.arepo.ArepoVoronoiGrid`
            Source `~fiesta.arepo.ArepoVoronoiGrid` to use for
            plotting the profile

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use. If ``None`` (default),
            takes the value ``u.pix`` if ``scale='pixel'``, 
            else ``u.cm`` if ``scale='physical'``.

        ndensity_unit : `~astropy.units.Unit`, optional
            The unit of number density to use. 
            Default value is ``u.cm**-3``.

        scatter: `bool`, optional
            If ``True`` (default), plots the scatter of radius versus density for
            each cell. Its color can be controlled through ``color`` argument
            in ``**kwargs``.

        contours: `int`, optional
            Number of contours, estimated through `scipy.stats.gaussian_kde`,
            of the scatter. If ``None`` (default), no contour plot is made.
            Its color can be controlled through ``cmap`` argument
            in ``**kwargs``.

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

        if self.arepo_ids is None:
            raise ValueError(utils._prestring() + "Filament needs to be characterized first using characterize_filament() function.")

        #Shedding units for ease of use
        points = avg.pos.to_value(length_unit)
        spline = self.get_points(np.linspace(0,1,self.nsamp*500)).to_value(length_unit)
        ndensity = avg.get_ndensity().to_value(ndensity_unit)
        mass = avg.mass.to_value(u.g)

        #Main figure
        fig = plt.figure(figsize=(8,8))
        if "figure" in kwargs:
            plt.setp(fig,**kwargs["figure"])
            
        ax = fig.add_subplot(111)

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
        ax.set_xlabel(r"Radial distance $r$ [{}]".format(length_unit.to_string()),fontsize=15)
        ax.set_ylabel(r"Number density $n$ [{}]".format(ndensity_unit.to_string()),fontsize=15)
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
        if "color" in kwargs:
            color1 = kwargs["color"]
        else:
            color1 = 'black'
        if "cmap" in kwargs:
            cmap2 = copy.copy(mpl.cm.get_cmap(kwargs["cmap"]))
        else:
            cmap2 = copy.copy(mpl.cm.get_cmap('Blues'))
        alpha1 = 0.8
        alpha2 = 0.8

        def density_estimation(m1, m2, weights=None):

            #see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

            X, Y = np.mgrid[min(m1):max(m1):100j, min(m2):max(m2):100j]                                                   
            positions = np.vstack([X.ravel(), Y.ravel()])                                         
            values = np.vstack([m1, m2])                                                                        
            kernel = stats.gaussian_kde(values,weights=weights)                                                                 
            Z = np.reshape(kernel(positions).T, X.shape)
            return X, Y, Z

        #Getting unique AREPO ids
        unique_ids = np.unique(np.hstack(self.arepo_ids))
        points = points[unique_ids]
        ndensity = ndensity[unique_ids]
        mass = mass[unique_ids]

        #Finding the radial distance of points and contour of scatter
        radius = cdist(points,spline).min(axis=1)
        X, Y, Z = density_estimation(radius,np.log10(ndensity)) #Note: the logarithm

        #Plotting now
        if scatter:
            scatter = ax.scatter(radius, ndensity, s=0.1, c=color1, alpha=alpha1, zorder=1)
        if contours is not None:
            contourf = ax.contourf(X, np.power(10,Y), Z, cmap=cmap2, alpha=alpha2, levels=contours, zorder=0)

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
#                              NETWORK                               #
######################################################################
        
class Network:
    """

    Class to deal with filament network output by DisPerSE.

    Attributes
    ----------

    cps : `list` of `~fiesta.disperse.CriticalPoint`'s
        Array of `~fiesta.disperse.CriticalPoint`'s in the network.

    ncps : int
        Number of critical points in the network.

    fils : `list` of `~fiesta.disperse.Filament`'s
        Array of `~fiesta.disperse.Filament`'s in the network.
    
    nfils : `int`
        Number of filaments in the network

    scale : `str`
        Either ``pixel`` if the grid is not scaled to the source
        AREPO snapshot, else ``physical``.

    Parameters
    ----------
    
    file_path : `str`
        File path of the DisPerSE output (currently only supports
        ASCII format).

    verbose : `bool`, optional
        If ``True``, prints output during file reading. 
        Default value is ``False``.
    
    """
    
    ######################################################################
    #                           CONSTRUCTORS                             #
    ######################################################################
    
    def __init__(self, file_path, verbose=False):

        #Assumes ASCII format (see http://www2.iap.fr/users/sousbie/web/html/indexbea5.html?post/NDskl_ascii-format)

        if(verbose):
            print(utils._prestring() + "Loading Network from DisPerSE ASCII file from \"{}\" ...".format(file_path))

        #FIRST STORING THE DIFFERENT SECTIONS OF THE ASCII OUTPUT
        with open(file_path) as f:
            text = f.read()
        stop1 = text.rfind("[CRITICAL POINTS]")
        stop2 = text.rfind("[FILAMENTS]")
        stop3 = text.rfind("[CRITICAL POINTS DATA]")
        stop4 = text.rfind("[FILAMENTS DATA]")
        critical_points_text = text[stop1:stop2]
        filaments_text = text[stop2:stop3]
        critical_data_text = text[stop3:stop4]
        filaments_data_text = text[stop4:]

        #READING CRITICAL POINT DATA
        critical_points = []
        critical_points_text_array = critical_points_text.splitlines()[2:]
        for i in range(len(critical_points_text_array)):
            line = critical_points_text_array[i]
            if(not line[0]==' '):
                l = line.split(' ')
                #Creating the object
                cp = CriticalPoint()
                cp.idx = len(critical_points) #Setting 'idx' here
                cp.pos = [float(l[3]),float(l[2]),float(l[1])] << u.pix #NOTE: Notice the order of l[1], l[2] and l[3]. This is the discrepancy in Disperse coordinates and AREPO.
                cp.nfil = int(critical_points_text_array[i+1])
                cp.scale = "pixel"
                #Adding to array
                critical_points.append(cp)
        critical_points = np.array(critical_points)

        #READING FILAMENT DATA
        filaments = []
        filaments_text_array = filaments_text.splitlines()[2:]
        for i in range(len(filaments_text_array)):
            line = filaments_text_array[i]
            if(not line[0]==' '):
                l = line.split(' ')
                #Creating the object
                fil = Filament()
                fil.idx = len(filaments)
                fil.nsamp = int(l[2])
                fil.cps = np.array([critical_points[int(l[0])],critical_points[int(l[1])]])
                fil.scale = "pixel"
                fil.samps = []
                while(True):
                    if(i==len(filaments_text_array)-1):
                        break
                    else:
                        i+=1
                        line = filaments_text_array[i]
                    if(not line[0]==' '):
                        i-=1
                        fil.samps = fil.samps << u.pix
                        fil._set_filfunc()
                        #Adding to array
                        filaments.append(fil)
                        break
                    else:
                        l = line.split(' ')
                        pos = [float(l[3]),float(l[2]),float(l[1])] #NOTE: Notice the order of l[1], l[2] and l[3]. This is the discrepancy in Disperse coordinates and AREPO.
                        fil.samps.append(pos)
        filaments = np.array(filaments)
                    
        if(verbose):
            print(utils._prestring() + "Completed loading Network from DisPerSE ASCII file from \"{}\" .".format(file_path))

        self.cps = critical_points
        self.ncps = len(critical_points)
        self.fils = filaments
        self.nfils = len(filaments)
        self.scale = "pixel"

        if(verbose):
            print(utils._prestring() + "Note: The Network has {} filaments and {} critical points.".format(self.nfils,self.ncps))

    ######################################################################
    #                          ACCESS FUNCTIONS                          #
    ######################################################################

    def get_filament(self, idx):
        """

        Function to fetch a filament of specific ID.

        Parameters
        ----------

        idx : `int`
            ID of the filament to fetch.

        Returns
        ----------

        fil : `~fiesta.disperse.Filament`
            Filament corresponding to the given ID.
            
        """
        for fil in self.fils:
            if(fil.idx == idx):
                return fil
        print(utils._prestring() + "No filament with idx = {}.".format(idx))

    def get_cp(self, idx):
        """

        Function to fetch a critical point of specific ID.

        Parameters
        ----------

        idx : `int`
            ID of the critical point to fetch.

        Returns
        ----------

        fil : `~fiesta.disperse.CriticalPoint`
            Critical point corresponding to the given ID.
            
        """
        for cp in self.cps:
            if(cp.idx == idx):
                return cp
        print(utils._prestring() + "No critical point with idx = {}.".format(idx))
        
    ######################################################################
    #                      RESCALE TO ORIGINAL AREPO                     #
    ######################################################################
    
    def set_scale(self, acg):
        """
        
        Function to set the physical scale of the network by passing the
        `~fiesta.arepo.ArepoCubicGrid` that was used as input for DisPerSE.

        Parameters
        ----------

        acg : `~fiesta.arepo.ArepoCubicGrid`
            Source `~fiesta.arepo.ArepoCubicGrid` to use for setting
            the scale.
        
        """

        if(self.scale=="physical"):
            print(utils._prestring()+"Already in \"physical\" units.")
            return
        
        if(acg.scale=="physical"):

            self.scale = acg.scale
            
            #Scale critical points
            for cp in self.cps:
                cp.scale = acg.scale
                cp.pos = u.Quantity([acg.xmin + cp.pos[0].value * acg.xlength,
                                  acg.ymin + cp.pos[1].value * acg.ylength,
                                  acg.zmin + cp.pos[2].value * acg.zlength])
                
            #Scale filaments
            for fil in self.fils:
                fil.scale = acg.scale
                #No need to change the cps here since they are already changed above
                fil.samps = u.Quantity([acg.xmin + fil.samps[:,0].value * acg.xlength,
                                        acg.ymin + fil.samps[:,1].value * acg.ylength,
                                        acg.zmin + fil.samps[:,2].value * acg.zlength]).T
                #Resetting the fil function
                fil._set_filfunc()

        else:

            raise ValueError(utils._prestring() + "ArepoCubicGrid must be in \"physical\" units.")

       
    ######################################################################
    #                    PROPERTIES OF THE NETWORK                       #
    ######################################################################
    
    def calc_filament_lengths(self, verbose=False):
        """

        Calculate the length of all filaments in the network.
         
        Works simply by calling `~fiesta.disperse.Filament.calc_length`
        for each filament.

        Attributes
        ----------

        avg : `~fiesta.arepo.ArepoVoronoiGrid`
            Source `~fiesta.arepo.ArepoVoronoiGrid` to use for
            calculating mass of the filament.

        verbose : `bool`, optional
            If ``True``, prints output during length calculation.
            Default value is ``False``.
        
        """
        if(verbose):
            print(utils._prestring() + "Completed calculating length of all filaments in Network...")
        for fil in self.fils:
            fil.calc_length(verbose)
        if(verbose):
            print(utils._prestring() + "Started calculating length of all filaments in Network...")
        
    def calc_filament_masses(self, avg, verbose=False):
        """

        Calculate the mass of all filaments in the network.
         
        Works simply by calling `~fiesta.disperse.Filament.calc_mass`
        for each filament.

        Attributes
        ----------

        avg : `~fiesta.arepo.ArepoVoronoiGrid`
            Source `~fiesta.arepo.ArepoVoronoiGrid` to use for
            calculating mass of filaments.

        verbose : `bool`, optional
            If ``True``, prints output during mass calculation.
            Default value is ``False``.
        
        """
        if(verbose):
            print(utils._prestring() + "Completed calculating mass of all filaments in Network...")
        for fil in self.fils:
            fil.calc_mass(avg, verbose)
        if(verbose):
            print(utils._prestring() + "Completed calculating mass of all filaments in Network...")
            
    ######################################################################
    #                        FILTERING FILAMENTS                         #
    ######################################################################
    
    def remove_short_filaments(self, min_length, verbose=False):
        """

        Remove filaments below a given minimum length. Useful for 
        discarding spurious filaments below grid resolution.

        Attributes
        ----------

        min_length : `~astropy.units.Quantity`
            Length below which all filaments are discarded from the network.

        verbose : `bool`, optional
            If ``True``, prints output during removal of short filaments.
            Default value is ``False``.
        
        """

        #Shedding units for ease of use
        if(self.scale=="physical"):
            unit = u.cm
        elif(self.scale=="pixel"):
            unit = u.pix
        utils.check_quantity(min_length, unit, "min_length")
        min_length = min_length.to_value(unit)
        
        #First filtering filaments
        fils_to_keep = []
        for i in range(self.nfils):
            fil = self.fils[i]
            if fil.length is None :
                raise ValueError(utils._prestring() + "Filament length is None. Need to calculate filament lengths first")
            if(fil.length.to_value(unit) > min_length):
                fils_to_keep.append(i)

        if(verbose):
            nfils_removed = self.nfils-len(fils_to_keep)
            print(utils._prestring() + "Removed {} short filaments from Network.".format(nfils_removed))

        self.fils = self.fils[fils_to_keep]
        self.nfils = len(self.fils)

        #Now updating the critical points corresponding to these
        cp_idxs_in_fils = [cp.idx for fil in self.fils for cp in fil.cps]
        cps_to_keep = []
        for j in range(self.ncps):
            cp = self.cps[j]
            if(cp.idx in cp_idxs_in_fils):
                cps_to_keep.append(j)
                cp.nfil = cp_idxs_in_fils.count(cp.idx)

        if(verbose):
            ncps_removed = self.ncps-len(cps_to_keep)
            print(utils._prestring() + "Removed {} critical points from Network.".format(ncps_removed))

        self.cps = self.cps[cps_to_keep]
        self.ncps = len(self.cps)

    ######################################################################
    #                        PLOTTING FUNCTIONS                          #
    ######################################################################

    def plot_network(self,
                    fil_idxs=None,
                    length_unit=None,
                    colors=None,
                    splines=False,
                    cylinders=False,
                    avg=None,
                    save=None,
                    **kwargs):
        """
        
        Plot the filament network in 3D.

        Parameters
        ----------

        fil_idxs : `list` of int`'s
            ID's of the filaments to plot. If ``None`` (default),
            all filaments are plotted.

        length_unit : `~astropy.units.Unit`, optional
            The unit of length to use for the plot.
            If ``None`` (default), takes the value 
            ``u.pix`` if ``scale='pixel'``, else 
            ``u.cm`` if ``scale='physical'``.

        colors: `list`, optional
            A list of *matplotlib*-compatible color strings corresponding
            to each filament plotted. If ``None`` (default), colors are picked
            by cycling through *matplotlib* tab10 colour palette.

        splines : `bool`, optional
            If ``True``, plots the spline fit (*filament function*) of each filament.
            Default value is ``False``.

        cylinders : `bool`, optional
            If ``True``, plots the AREPO cells that are part of the filament.
            Default value is ``False``. Requires 
            `~fiesta.disperse.Filament.arepo_ids` to be set.

        avg : `~fiesta.arepo.ArepoVoronoiGrid`, optional
            The source `~fiesta.arepo.ArepoVoronoiGrid` snapshot
            to use for plotting the cells. Omitted if ``cylinders=False``.

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

        #Checking units
        if(self.scale=="physical"):
            internal_unit = u.cm
        elif(self.scale=="pixel"):
            internal_unit = u.pix
        if length_unit is None:
            length_unit = internal_unit
        utils.check_unit(length_unit, internal_unit)

        #Figure properties
        nsols = len(fil_idxs)
        if colors is None:
            cmap = plt.cm.tab10
            colors = cmap(np.arange(nsols)%cmap.N)

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

        if fil_idxs is None:

            fil_idxs = [fil.idx for fil in self.fils]

        for fil_idx, c in zip(fil_idxs, colors):

            fil = self.get_filament(fil_idx)

            points = fil.samps.to_value(length_unit)
            ax.plot(*points.T, color=c)
            ax.scatter(*points.T, s=10, color=c)

            if(splines):
                spline = fil.get_points(np.linspace(0,1,fil.nsamp*100,endpoint=True)).to_value(length_unit)
                ax.plot(*spline.T, color=c, linestyle='--')

            if(cylinders):

                if self.arepo_ids is None:
                    raise ValueError(utils._prestring() + "Filament needs to be characterized first using characterize_filament() function.")
                if(avg is None):
                    raise ValueError(utils._prestring()+"Need to pass ArepoVoronoiGrid to plot the cylinders!")

                ncyls = len(fil.arepo_ids)
                cmap = plt.cm.tab10
                cylcolors = cmap(np.arange(ncyls)%cmap.N)

                for (ids,cylcolor) in zip(fil.arepo_ids,cylcolors):
                    cylpoints = avg.pos[ids].to_value(length_unit)
                    ax.scatter(*cylpoints.T, alpha=0.2, s=10, color=cylcolor)

    
        #z=ax.get_zlim()
        #ax.plot3D(fil.samps.transpose()[0],fil.samps.transpose()[1], z[0], c='grey', alpha=0.5,zorder=-10)
        #ax.scatter(fil.samps.transpose()[0],fil.samps.transpose()[1], z[0], c='grey', alpha=0.5,s=10,zorder=-10)
        #ax.set_zlim(z)
            
        #x=ax.get_xlim()
        #ax.plot3D([x[0]]*len(fil.samps),fil.samps.transpose()[1],fil.samps.transpose()[2], c='grey',alpha=0.5,zorder=-10)
        #ax.scatter([x[0]]*len(fil.samps),fil.samps.transpose()[1],fil.samps.transpose()[2], c='grey',alpha=0.5,zorder=-10)
        #ax.set_xlim(x)
            
        #y=ax.get_ylim()
        #ax.plot3D(fil.samps.transpose()[0],[y[1]]*len(fil.samps),fil.samps.transpose()[2], c='grey',alpha=0.5,zorder=-10)
        #ax.scatter(fil.samps.transpose()[0],[y[1]]*len(fil.samps),fil.samps.transpose()[2], c='grey',alpha=0.5,s=10,zorder=-10)
        #ax.set_ylim(y)

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