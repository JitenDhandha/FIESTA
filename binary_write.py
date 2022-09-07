######################################################################
# Authors: Robin Tress
# Last edited: ---
######################################################################

#from io_utility import *
import numpy as np
from numpy import uint32, uint64, float64, float32

io_flags = {'mc_tracer'           : True,\
            'time_steps'          : True,\
            'sgchem'              : True, \
            'variable_metallicity': True,\
            'sgchem_NL99'         : False}


########### SNAPSHOT FILE ##########

header_names = ('num_particles', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback', \
                'num_particles_total', 'flag_cooling', 'num_files', 'boxsize', 'omega0', 'omegaLambda', \
                'hubble0', 'flag_stellarage', 'flag_metals', 'npartTotalHighWord', 'flag_entropy_instead_u', 'flag_doubleprecision', \
                'flag_lpt_ics', 'lpt_scalingfactor', 'flag_tracer_field', 'composition_vector_length', 'buffer')

header_sizes = ((uint32, 6), (float64, 6), (float64, 1), (float64, 1), (uint32, 1), (uint32, 1), \
                (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1), (float64, 1), (float64, 1), \
                (float64, 1), (uint32, 1), (uint32, 1), (uint32, 6), (uint32, 1), (uint32, 1), \
                (uint32, 1), (float32, 1), (uint32, 1), (uint32, 1), (np.uint8, 40))

########### IC FILE ##########
IC_header_names = ('num_particles', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback',              \
                   'num_particles_total', 'flag_cooling', 'num_files', 'boxsize', 'omega0', 'omegaLambda',\
                   'hubble0', 'flag_stellarage', 'flag_metals', 'npartTotalHighWord', 'utime', 'umass',   \
                   'udist', 'flag_entropyICs', 'unused')

IC_header_sizes = ((uint32, 6), (float64, 6),(float64, 1),(float64, 1),(uint32, 1), (uint32, 1),  \
                   (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1),(float64, 1),(float64, 1), \
                   (float64, 1),(uint32, 1), (uint32, 1), (uint32, 6), (float64,1), (float64,1),  \
                   (float64,1), (uint32, 1), (np.uint8, 36))

########## necessary WRITE functions #########

def writeu(f, data):
    """ Write a numpy array to the unformatted fortran file f """
    data_size = data.size * data.dtype.itemsize

    data_size = np.array(data_size, dtype=uint32)

    data_size.tofile(f)

    data.tofile(f)

    data_size.tofile(f)
    
####### binary write functions #########

def write_snapshot(filename, header, data):
    """ Write a binary snapshot file (unformatted fortran) """
    # do some checks on the header
    for name, size in zip(header_names, header_sizes):
      if name not in header:
        print('Missing %s in header file' % name)
        raise Exception('Missing %s in header file' % name)

      if np.array(header[name]).size != size[1]:
        msg = 'Header %s should contain %d elements, %d found' % \
              (name, size[1], np.array(header[name]).size)
        print(msg)
        raise Exception(msg)

    nparts = header['num_particles']

    #should we write in single or double precision
    if header['flag_doubleprecision']==0:
        precision = float32
    else:
        precision = float64

    # ok so far, lets write the file
    f = open(filename, 'wb')

    # write the header
    np.array(256, uint32).tofile(f)

    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    # final block
    np.array(256, uint32).tofile(f)

    # write the data
    writeu(f, data['pos'].astype(precision))
    writeu(f, data['vel'].astype(precision))
    writeu(f, data['id'])
    writeu(f, data['mass'].astype(precision))

    writeu(f,data['u_therm'].astype(precision) )
    writeu(f, data['rho'].astype(precision))
    
#    if io_flags['time_steps']:
#      writeu(f, data['tsteps'].astype(precision))
    
    if io_flags['mc_tracer']:
      writeu(f, data['numtrace'])
      writeu(f, data['tracerid'])
      writeu(f, data['parentid'])
    
    if io_flags['time_steps']:
      writeu(f, data['tsteps'].astype(precision))
    
    if io_flags['sgchem']:

      if io_flags['variable_metallicity']:
        writeu(f, data['abundz'].astype(precision))
      
      writeu(f, data['chem'].astype(precision))
      writeu(f, data['tdust'].astype(precision))
      
      if io_flags['variable_metallicity']:
        writeu(f, data['dtgratio'].astype(precision))

    # all done!
    f.close()

def write_IC(pos,vel,mass,u_therm,filename):
    # write initial conditions for arepo given:
    # pos = positions
    # vel = velocities 
    # mass = masses
    # u_therm = thermal energy per unit mass
    npts = mass.size
    header = {'num_particles': np.array([npts,       0,       0,       0,       0,       0]).astype(uint32),
             'mass': np.array([ 0.,  0.,  0.,  0.,  0.,  0.]).astype(float64),
             'time': np.array([0.]).astype(float64),
             'redshift': np.array([0.]).astype(float64),
             'flag_sfr': np.array([0]).astype(uint32),
             'flag_feedback': np.array([0]).astype(uint32),
             'num_particles_total': np.array([npts,       0,       0,       0,       0,       0]).astype(uint32),
             'flag_cooling': np.array([0]).astype(uint32),
             'num_files': np.array([1]).astype(uint32),
             'boxsize': np.array([ 0.]).astype(float64),
             'omega0': np.array([ 0.]).astype(float64),
             'omegaLambda': np.array([ 0.]).astype(float64),
             'hubble0': np.array([ 1.]).astype(float64),
             'flag_stellarage': np.array([0]).astype(uint32),
             'flag_metals': np.array([0]).astype(uint32),
             'npartTotalHighWord': np.array([0, 0, 0, 0, 0, 0]).astype(uint32),
             'flag_entropy_instead_u': np.array([0]).astype(uint32),
             'flag_doubleprecision': np.array([0]).astype(uint32),
             'flag_lpt_ics': np.array([0]).astype(uint32),
             'lpt_scalingfactor': np.array([0]).astype(float32),
             'flag_tracer_field': np.array([0]).astype(uint32),
             'composition_vector_length': np.array([0]).astype(uint32),
             'buffer': np.empty([40]).astype(np.uint8)}

    ID = np.arange(npts) + 1
    f = open(filename, 'wb')
    precision = float32
    idprecision = np.int32
  
    np.array(256, uint32).tofile(f)
    for name, size in zip(header_names, header_sizes):
        np.array(header[name]).astype(size[0]).tofile(f)
    np.array(256, uint32).tofile(f)
    writeu(f, pos.astype(precision))
    writeu(f, vel.astype(precision))
    writeu(f, ID.astype(idprecision))
    writeu(f, mass.astype(precision))
    writeu(f, u_therm.astype(precision) )
    f.close()
    return
