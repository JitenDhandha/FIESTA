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
from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Own libs
from .units import *
from .physical_functions import *

######################################################################
#                         CRITICAL POINT                             #
######################################################################

class CriticalPoint:
    def __init__(self):
        self.idx = int
        self.pos = []
        self.nfil = int
        self.scale = str #either "Euclidean" or "AREPO"
    
######################################################################
#                              FILAMENT                              #
######################################################################

class Filament:
    def __init__(self):
        
        #Core variables
        self.idx = int
        self.cps = []
        self.nsamp = int
        self.samps = [] #NOTE: The samps contain the critical points already (as first and last element)
        self.scale = str #either "Euclidean" or "AREPO"
        
        #Tools
        self.charfunc = None    #characteristic function of the filament
        self.AREPO_ids = []
        
        #Calculated properties
        self.length = float #parsec
        self.mass = float #solar masses
    
    ######################################################################
    #                               CHARFUNC                             #
    ######################################################################
    
    #The charfunc is a function that parametrizes the 1D curve in 3D space.
    #The parameter 'u' is the normalized distance along filament.
    #Hence, u=0 corresponds to first critical point and u=1, to the last critical point.
    
    def set_charfunc(self):
            
        #Since you can always fit a N-1 degree polynomial to N points
        if(self.nsamp>3):
            degree = 3
        else:
            degree = self.nsamp-1
        #Note the smoothing s=0 so that the fit curve passes through all sampling points
        func, _ = splprep([self.samps[:,0],self.samps[:,1],self.samps[:,2]],s=0,k=degree)

        self.charfunc = func
    
    def get_charfunc(self,u):
        
        points = np.array(splev(u,self.charfunc)).T
        return points
    
    ######################################################################
    #            CREATING CYLINDERS BETWEEN SAMPLING POINTS              #
    ######################################################################
            
    def make_cylinders(self, avg, plot=False):

        if(self.scale!='AREPO'):
            raise ValueError("FIESTA >> Cannot make cylinders since scale is not in AREPO units.")

        def normalize(v):
            v_unit = v/np.sqrt(v[0]**2+v[1]**2+v[2]**2)
            return v_unit

        def rotate_axes(normal):
            n = normalize(normal)
            theta = np.arccos(n[2])
            u = normalize(np.cross(n,[0,0,1]))
            rot = np.array([[np.cos(theta)+u[0]**2*(1-np.cos(theta)), 
                             u[0]*u[1]*(1-np.cos(theta))-u[2]*np.sin(theta), 
                             u[0]*u[2]*(1-np.cos(theta))+u[1]*np.sin(theta)],
                            [u[1]*u[0]*(1-np.cos(theta))+u[2]*np.sin(theta), 
                             np.cos(theta)+u[1]**2*(1-np.cos(theta)), 
                             u[1]*u[2]*(1-np.cos(theta))-u[0]*np.sin(theta)],
                            [u[2]*u[0]*(1-np.cos(theta))-u[1]*np.sin(theta), 
                             u[2]*u[1]*(1-np.cos(theta))+u[0]*np.sin(theta), 
                             np.cos(theta)+u[2]**2*(1-np.cos(theta))]])
            return rot
        
        # create storage arrays
        normals = np.zeros((len(self.samps),3))
        cylinders = []
        id_nums = []

        # get normal vectors for filaments points
        normals[0] = self.samps[0]-self.samps[1]
        for i in range(1,len(self.samps)-1):
            normals[i] = self.samps[i-1]-self.samps[i+1]
        normals[len(self.samps)-1] = self.samps[len(self.samps)-2]-self.samps[len(self.samps)-1]

        allpos = avg.pos[0:avg.ngas].transpose()
        
        # for each point in a filament
        for i in range(len(self.samps)):

            cylinder = []
            id_num = []
            # compute rotation matrix
            rot = rotate_axes(normals[i])        
            # define bounds of region of interest [[x_min, y_min, z_min], [x_max, y_max, z_max]] (self.samps are in AREPO code units)
            bounds = [[self.samps[i][0]-0.2, self.samps[i][1]-0.2, self.samps[i][2]-0.2], [self.samps[i][0]+0.2, self.samps[i][1]+0.2, self.samps[i][2]+0.2]]
            # convert to AREPO code units for search
            bounds = np.array(bounds)
            # find all points in the region
            pts_nan = np.where((allpos[0] > bounds[0][0]) 
                             & (allpos[1] > bounds[0][1]) 
                             & (allpos[2] > bounds[0][2]) 
                             & (allpos[0] < bounds[1][0]) 
                             & (allpos[1] < bounds[1][1]) 
                             & (allpos[2] < bounds[1][2]), 
                             allpos, np.nan)
            ids_nan = np.where((allpos[0] > bounds[0][0]) 
                             & (allpos[1] > bounds[0][1])
                             & (allpos[2] > bounds[0][2]) 
                             & (allpos[0] < bounds[1][0]) 
                             & (allpos[1] < bounds[1][1]) 
                             & (allpos[2] < bounds[1][2]),
                             np.array([[avg.gas_ids]]*3), np.nan)
            pts = np.array([row[~np.isnan(row)] for row in pts_nan])
            ids = np.array([row[~np.isnan(row)] for row in ids_nan])
            pts = pts.transpose()
            ids = ids.transpose()

            self_rotated = np.zeros((len(self.samps),3))
            # translate such that sample point i is at origin
            self_at_origin = self.samps-self.samps[i]
            for j in range(len(self.samps)):
                # rotate everything
                self_rotated[j] = np.dot(rot, self_at_origin[j])
            # translate back
            self_translated_back = self_rotated + self.samps[i]
            
            pts_rotated = np.zeros((len(pts),3))
            # translate such that sample point i is at origin
            pts_at_origin = pts-self.samps[i]
            #print(self_translated_back[1][2], self_translated_back[0][2])
            for k in range(len(pts)):
                # rotate everything
                pts_rotated[k] = np.dot(rot, pts_at_origin[k])
                if i < len(self.samps)-1:
                    if self_rotated[i+1][2] < self_rotated[i][2]: # does filament have negative gradient?
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] >= self_rotated[i+1][2]) & (pts_rotated[k][2] <= self_rotated[i][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                    else:
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] <= self_rotated[i+1][2]) & (pts_rotated[k][2] >= self_rotated[i][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                elif i == len(self.samps)-1:
                    if self_rotated[i][2] < self_rotated[i-1][2]:
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] >= self_rotated[i][2]) & (pts_rotated[k][2] <= self_rotated[i-1][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                    else:
                        # define cylinder
                        if ((pts_rotated[k][0]**2 + pts_rotated[k][1]**2) <= (0.1)**2) & (pts_rotated[k][2] <= self_rotated[i][2]) & (pts_rotated[k][2] >= self_rotated[i-1][2]):
                            cylinder.append(pts_rotated[k])
                            id_num.append(ids[k])
                else:
                    print('i does not have an acceptable value')
                    
            cylinder = np.array(cylinder)
            id_num = np.array(id_num)
                        
            if cylinder.size == 0:
                #print('No points found for sample point '+ str(i) + ' in ' + str(self))
                pass
            else:
                id_num = id_num.transpose()[0]
            
                # find inverse rotation matrix (inverse of a rotation matrix is its transpose)
                invRot = rot.transpose()
                # invert rotation on points to confirm alignement with filament
                for k in range(len(cylinder)):
                    cylinder[k] = np.dot(invRot, cylinder[k])

                # translate back
                cylinder = np.array(cylinder + self.samps[i])
                
            cylinders.append(cylinder)
            id_nums.append(id_num)
        
    # visual confirmation
        if plot == True:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_title(r"$\mathrm{Particles~adjacent~to~a~filament~spine}$", fontsize='xx-large')
            ax.set_xlabel(r"$\mathrm{x~[pc]}$",fontsize='xx-large')
            ax.set_ylabel(r"$\mathrm{y~[pc]}$",fontsize='xx-large')
            ax.set_zlabel(r"$\mathrm{z~[pc]}$",fontsize='xx-large')
            ax.plot3D(self.samps.transpose()[0],self.samps.transpose()[1],self.samps.transpose()[2],color='k')
            for i in range(len(cylinders)):
                ax.scatter(cylinders[i].transpose()[0], cylinders[i].transpose()[1], cylinders[i].transpose()[2], alpha=0.05)
            
        for c in range(len(cylinders)):
            cylinders[c] = cylinders[c]

        self.AREPO_ids = id_nums

    ######################################################################
    #                      CALCULATING PROPERTIES                        #
    ######################################################################
        
    def calc_length(self):
        
        #First creating sampling points along the filament
        n = self.nsamp
        u_sampling = np.linspace(0,1,n*500,endpoint=True)
        points = self.get_charfunc(u_sampling)
    
        #CALCULATING LENGTH
        length = 0
        for i in range(1,len(points)):
            length += euclidean(points[i-1],points[i])
        self.length = length
        
    def calc_mass(self, avg):
        
        if(self.AREPO_ids == []):
            self.make_cylinders(avg)
        
        unique_ids = np.unique([item for sublist in self.AREPO_ids for item in sublist])
        self.mass = np.sum(np.array(avg.mass)[unique_ids.astype(int)])

######################################################################
#                           BIFURCATIONS                             #
######################################################################
class BifurcationPoint:
    def __init__(self, pos, rank, scale):
        self.pos = pos
        self.rank = rank
        self.scale = scale #either "Euclidean" or "AREPO"
        
######################################################################
#                              NETWORK                               #
######################################################################
        
class Network:
    
    ######################################################################
    #                           CONSTRUCTORS                             #
    ######################################################################
    
    def __init__(self, file_path=None, verbose=False):
        self.file_path = str
        self.ntotcps = int
        self.ntotfils = int
        self.cps = []
        self.fils = []
        self.scale = str #either "Euclidean" or "AREPO"

        #These are extra properties calculated if required
        self.bifurcations = []
        self.nbifurcation = int
        
        if(not file_path==None):
            self.read_ASCII_file(file_path, verbose)

    ######################################################################
    #                          ACCESS FUNCTIONS                          #
    ######################################################################

    def get_filament(self, fil_idx):
        for fil in self.fils:
            if(fil.idx == fil_idx):
                return fil
        raise ValueError("FIESTA >> No filamet with ID = {}.".format(fil_idx))

    def get_cp(self, cp_idx):
        for cp in self.cps:
            if(cp.idx == cp_idx):
                return cp
        raise ValueError("FIESTA >> No critical point with ID = {}.".format(cp_idx))
        
    ######################################################################
    #                       READ AND WRITE FUNCTIONS                     #
    ######################################################################
        
    def read_ASCII_file(self, file_path, verbose=False):
        
        #see http://www2.iap.fr/users/sousbie/web/html/indexbea5.html?post/NDskl_ascii-format

        if(verbose):
            print("FIESTA >> Loading Network from DisPerSE ASCII file from \"{}\" ...".format(file_path))
        
        self.file_path = file_path

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
                cp = CriticalPoint()
                cp.idx = len(critical_points) #Setting 'idx' here
                cp.pos = np.array([float(l[3]),float(l[2]),float(l[1])]) #Setting 'pos' here
                #IMPORTANT NOTE: Notice the order of l[1], l[2] and l[3].
                #This is the discrepancy in Disperse coordinates and AREPO.
                cp.nfil = int(critical_points_text_array[i+1]) #Setting 'nfil' here
                cp.scale = "Euclidean" #Setting 'scale' here
                critical_points.append(cp)
        critical_points = np.array(critical_points)

        #READING FILAMENT DATA
        filaments = []
        filaments_text_array = filaments_text.splitlines()[2:]
        for i in range(len(filaments_text_array)):
            line = filaments_text_array[i]
            if(not line[0]==' '):
                l = line.split(' ')
                fil = Filament()
                fil.idx = len(filaments) #Setting 'idx' here
                fil.nsamp = int(l[2]) #Setting 'nsamp' here
                fil.cps = np.array([critical_points[int(l[0])],critical_points[int(l[1])]]) #Setting 'cps' here
                while(True):
                    if(i==len(filaments_text_array)-1):
                        break
                    else:
                        i+=1
                        line = filaments_text_array[i]
                    if(not line[0]==' '):
                        i-=1
                        fil.samps = np.array(fil.samps)
                        fil.set_charfunc() #Setting 'charfunc' here
                        fil.scale = "Euclidean" #Setting 'scale' here
                        filaments.append(fil)
                        break
                    else:
                        l = line.split(' ')
                        pos = np.array([float(l[3]),float(l[2]),float(l[1])])
                        #IMPORTANT NOTE: Notice the order of l[1], l[2] and l[3].
                        #This is the discrepancy in Disperse coordinates and AREPO.
                        fil.samps.append(pos)
        filaments = np.array(filaments)
                    
        if(verbose):
            print("FIESTA >> Completed loading Network from DisPerSE ASCII file from \"{}\" .".format(file_path))

        self.cps = critical_points
        self.ntotcps = len(critical_points)
        self.fils = filaments
        self.ntotfils = len(filaments)
        self.scale = "Euclidean"

        if(verbose):
            print("FIESTA >> Note: The Network has {} filaments and {} critical points.".format(self.ntotfils,self.ntotcps))
        
    ######################################################################
    #                      RESCALE TO ORIGINAL AREPO                     #
    ######################################################################
    
    def set_scale(self, acg):  #acg = ArepoCubicGrid
        
        self.scale = acg.scale
        
        #Scale critical points
        for cp in self.cps:
            cp.scale = acg.scale
            cp.pos[0] = acg.xmin + cp.pos[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
            cp.pos[1] = acg.ymin + cp.pos[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
            cp.pos[2] = acg.zmin + cp.pos[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
            
        #Scale filaments
        for fil in self.fils:
            fil.scale = acg.scale
            #No need to change the cps here since they are already changed above
            for samp in fil.samps:
                samp[0] = acg.xmin + samp[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
                samp[1] = acg.ymin + samp[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
                samp[2] = acg.zmin + samp[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
            #Resetting the fil function
            fil.set_charfunc()
            
        #Scale bifurcations (if any)
        for bf in self.bifurcations:
            bf.scale = acg.scale
            bf.pos[0] = acg.xmin + bf.pos[0] * float(acg.xmax - acg.xmin)/float(acg.nx)
            bf.pos[1] = acg.ymin + bf.pos[1] * float(acg.ymax - acg.ymin)/float(acg.ny)
            bf.pos[2] = acg.zmin + bf.pos[2] * float(acg.zmax - acg.zmin)/float(acg.nz)
            
    ######################################################################
    #                        FILTERING FILAMENTS                         #
    ######################################################################
    
    #Removing all filaments with only 2 sampling points
    def remove_short_filaments(self, verbose=False):
        
        #First filtering filaments
        fils_to_keep = []
        for i in range(self.ntotfils):
            fil = self.fils[i]
            if(fil.nsamp!=2):
                fils_to_keep.append(i)

        if(verbose):
            nfils_removed = self.ntotfils-len(fils_to_keep)
            print("FIESTA >> Removing {} short filaments from Network.".format(nfils_removed))

        self.fils = self.fils[fils_to_keep]
        self.ntotfils = len(self.fils)

        #Now updating the critical points corresponding to these
        cp_idxs_in_fils = [cp.idx for fil in self.fils for cp in fil.cps]
        cps_to_keep = []
        for j in range(self.ntotcps):
            cp = self.cps[j]
            if(cp.idx in cp_idxs_in_fils):
                cps_to_keep.append(j)
                cp.nfil = cp_idxs_in_fils.count(cp.idx)

        self.cps = self.cps[cps_to_keep]
        self.ntotcps = len(self.cps)
       
    ######################################################################
    #                    PROPERTIES OF THE NETWORK                       #
    ######################################################################
    
    def calc_filament_lengths(self, verbose=False):
        if(verbose):
            print("FIESTA >> Calculating length of filaments in Network {}...".format(self))
        for fil in self.fils:
            fil.calc_length()
        if(verbose):
            print("FIESTA >> Completed calculating length of filaments in Network {}.".format(self))
        
    def calc_filament_masses(self, avg, verbose=False):
        if(verbose):
            print("FIESTA >> Calculating mass of filaments in Network {}...".format(self))
        counter = 0
        for fil in self.fils:
            counter+=1
            if(verbose):
                print("FIESTA >> Filaments done: "+str(counter)+"/"+str(len(self.fils)))
            fil.calc_mass(avg)
        if(verbose):
            print("FIESTA >> Completed calculating mass of filaments in Network {}.".format(self))

    ######################################################################
    #                   BIFURCATIONS OF THE NETWORK                      #
    ######################################################################

    # Finds bifurcation points for a set of filaments, with a variable radius.
    # This parameter has an impact on the evaluated number of filaments.

    def find_bifurcation_points(self, radius=0, verbose=False):

        if(verbose):
            print("FIESTA >> Finding bifurcation points in Network {}...".format(self))

        # initialise arrays
        points = []
        ids = []
        candidates = []
        bifurcations = []
        id1 = []
        # iterate through points in each filament and assign an ID corresponding to the filament they belong to
        for j in range(self.ntotfils):
            for i in self.fils[j].samps:
                points.append(i)
                ids.append(j)
        points = np.array(points)
        ids = np.array(ids)
        # make a KDtree from all the points
        tree = cKDTree(points)
        # get all the points in the KDTree which are within a radius of a point for each sample point
        for k in points:
            candidate = tree.data[tree.query_ball_point(k, radius)]
            candidates.append(candidate)
        # for each set of candidate points corresponding to a ball around a sample point
        for cs in range(len(candidates)):
            # reset arrays
            id1 = []
            # for each of the individual candidates
            for c in candidates[cs]:
                # for each sample point
                for k in range(len(points)):
                    # compare candidate point to sample point and assign IDs to matching points
                    if c[0] == points[k][0] and c[1] == points[k][1] and c[2] == points[k][2]:
                        id1.append(ids[k])
            # compare all ids in the list of candidates to the first
            id1 = np.array(id1)
            id1 = np.unique(id1)
            if len(id1) < 2:
                pass
            # if there are different ids in the set of candidate points around a sample point, it is a bifurcation point
            else:
                bifurcations.append(BifurcationPoint(points[cs],len(id1),self.scale))
        bifurcations = np.array(bifurcations)
        # the number of bifurcation points is half of the points classified as bifurcation points (pairs) for the correct separation threshold
        
        uniq_ids = np.unique([bf.pos for bf in bifurcations],return_index=True,axis=0)[1]
        bifurcations = bifurcations[uniq_ids]

        bifurcation_number = len(bifurcations)/2

        self.bifurcations = bifurcations
        self.nbifurcation = bifurcation_number

        if(verbose):
            print("FIESTA >> Completed finding bifurcation points in Network {}.".format(self))

    ######################################################################
    #                        PLOTTING FUNCTIONS                          #
    ######################################################################

    def plot_network(self, 
                    fil_idxs,
                    colors=None,
                    save=None,
                    splines=False,
                    cylinders=False,
                    avg=None,
                    bifurcations=False,
                    **kwargs):

        print("FIESTA >> Plotting Network {}...".format(self))

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

        for fil_idx, c, in zip(fil_idxs, colors):

            fil = self.get_filament(fil_idx)

            points = fil.samps
            ax.plot(points[:,0], points[:,1], points[:,2], color=c)
            ax.scatter(points[:,0], points[:,1], points[:,2], s=10, color=c)

            if(splines):
                u_sampling = np.linspace(0,1,fil.nsamp*100,endpoint=True)
                points = fil.get_charfunc(u_sampling)
                ax.plot(points[:,0], points[:,1], points[:,2], color=c, linestyle='--')

            if(cylinders):

                if avg is None:
                    raise TypeError("FIESTA >> Empty \"avg\" parameter needed for plotting cylinders.")

                ncyls = len(fil.AREPO_ids)
                cmap = plt.cm.tab10
                cylcolors = cmap(np.arange(ncyls)%cmap.N)

                for (ids,cylcolor) in zip(fil.AREPO_ids,cylcolors):
                    points = avg.pos[[int(idx) for idx in ids]]
                    ax.scatter(points[:,0], points[:,1], points[:,2], alpha=0.2, s=10, color=cylcolor)

        if(bifurcations):
            for bf in self.bifurcations:
                point = bf.pos
                if(bf.rank==2):
                    ax.scatter(point[0], point[1], point[2], s=10,c='green', zorder=10)
                if(bf.rank>2):
                    ax.scatter(point[0], point[1], point[2], s=10,c='red', zorder=10)

        ############### Plotting end ################

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

        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=100)