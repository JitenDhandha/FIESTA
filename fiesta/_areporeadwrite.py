######################################################################
# Authors: Robin Tress
# Last edited: ---
######################################################################

import numpy as np
from numpy import uint32, uint64, float64, float32

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

########## necessary READ functions #########

def read_header(f, h_names, h_sizes):
    """ Read the binary header file into a dictionary """
    block_size = np.fromfile(f, uint32, 1)[0]
    header = dict(((name, np.fromfile(f, dtype=size[0], count=size[1])) \
                 for name, size in zip(h_names, h_sizes)))
                 
    assert(np.fromfile(f, uint32, 1)[0] == 256)

    return header

def readu(f, dtype=None, components=1):
    """ Read a numpy array from the unformatted fortran file f """
    data_size = np.fromfile(f, uint32, 1)[0]

    count = data_size//np.dtype(dtype).itemsize
    arr = np.fromfile(f, dtype, count)

    final_block = np.fromfile(f, uint32, 1)[0]

    # check the flag at the beginning corresponds to that at the end
    assert(data_size == final_block)

    return arr

def readIDs(f, count):
    """ Read a the ID block from a binary snapshot file """
    data_size = np.fromfile(f, uint32, 1)[0]
    f.seek(-4, 1)

    count = int(count)
    #print(data_size//4,data_size//8,count)
    if data_size // 4 == count: dtype = uint32
    elif data_size // 8 == count: dtype = uint64
    else: raise Exception('Incorrect number of IDs requested')

    #print("ID type: ", dtype)

    return readu(f, dtype, 1)


########## necessary WRITE functions #########

def writeu(f, data):
    """ Write a numpy array to the unformatted fortran file f """
    data_size = data.size * data.dtype.itemsize
    data_size = np.array(data_size, dtype=uint32)

    data_size.tofile(f)

    data.tofile(f)

    data_size.tofile(f)

####### binary read functions #########

def return_header(filename):
    f = open(filename, mode='rb')
    header = read_header(f, header_names, header_sizes)
    f.close()
    return header

def read_IC(filename):
    f = open(filename, mode='rb')
    print("Loading IC file %s" % filename)

    data = {} # dictionary to hold data
  
    # read the header
    header = read_header(f,IC_header_names, IC_header_sizes)
  
    nparts = header['num_particles']
  
    npts = nparts.sum()
    precision = float32
    ranges=None
  
    data['pos'] = readu(f, precision, components=3).reshape(npts, 3)
    data['vel'] = readu(f, precision, components=3).reshape(npts, 3)
    data['id'] = readIDs(f, npts)
  
    # there are two methods of defining the masses in arepo, either by defining the mass in the mass array in the header
    # (then all particles of that type are given that constant mass) or by defining the mass of each particle 
    if (header['mass'].sum() == 0.):
      data['mass'] = readu(f, precision)
      data['u_therm'] = readu(f, precision)
    
    f.close()
  
    return data, header

def read_snapshot(filename, io_flags):
    """ Reads a binary snapshot file """
    f = open(filename, mode='rb')
    #print ("Loading file %s" % filename)

    data = {} # dictionary to hold data

    # read the header
    header = read_header(f, header_names, header_sizes)

    nparts = header['num_particles']
    masses = header['mass']
    #print ('Particles', nparts)
    #print ('Masses', masses)

    n_gas = nparts[0]
    n_sinks = nparts[5]
    n_tracer = nparts[2]
    total = n_gas + n_sinks

    #print ('Gas particles', n_gas)

    #if n_tracer != 0:
        #print ('Tracer particles', n_tracer)
    #if n_sinks != 0:
        #print ('Sink particles', n_sinks)

    #print ('Time = ', header['time'])

    if header['flag_doubleprecision']:
        precision = float64
        #print ('Precision: Double')
    else:
        precision = float32
        #print ('Precision: Float')
    
    """Now starts the reading!"""

    data['pos'] = readu(f, precision, 3).reshape((total, 3))
    data['vel'] = readu(f, precision, 3).reshape((total, 3))
    data['id'] = readIDs(f, total)
    data['mass'] = readu(f, precision)
    
    data['u_therm'] = readu(f, precision)
    data['rho'] = readu(f, precision)
    
    if io_flags['time_steps']:
      data['tsteps'] = readu(f, precision)
    
    if io_flags['mc_tracer']:
      data['numtrace'] = readu(f, uint32)
      data['tracerid'] = readIDs(f, n_tracer)
      data['parentid'] = readIDs(f, n_tracer)
    
    if io_flags['sgchem']:
      
      if io_flags['variable_metallicity']:
        data['abundz'] = readu(f, precision, 4).reshape((n_gas, 4))
      
      if io_flags['sgchem_NL99']:
        data['chem'] = readu(f, precision, 9).reshape((n_gas, 9))
      else:
        data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
      
      data['tdust'] = readu(f, precision)
      
      if io_flags['variable_metallicity']:
        data['dtgratio'] = readu(f, precision)

    f.close()
    return data, header

def read_image(filename):
    f = open(filename, mode='rb')
    #print ("Loading file %s" % filename)

    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)
    
    npix_x=int(npix_x)
    npix_y=int(npix_y)

    #print (npix_x, npix_y)
    arepo_image = np.fromfile(f, float32, npix_x*npix_y).reshape((int(npix_x), int(npix_y)))
    arepo_image = np.rot90(arepo_image)
    f.close()
    return arepo_image

def read_vector_image(filename):
    f = open(filename, mode='rb')
    #print ("Loading file %s" % filename)

    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)

    #print (npix_x, npix_y)
    arepo_image = np.fromfile(f, float32, npix_x*npix_y*3).reshape((npix_x, npix_y,3))
    arepo_image = np.rot90(arepo_image)
    f.close()
    return arepo_image
    
def read_grid(filename):
    f = open(filename, mode='rb')
    #print ("Loading file %s" % filename)
    
    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)
    npix_z = np.fromfile(f, uint32, 1)
    #print (npix_x, npix_y,npix_z)
    
    arepo_grid = np.fromfile(f, float32, npix_x[0]*npix_y[0]*npix_z[0]).reshape((npix_x[0], npix_y[0],npix_z[0]))
    
    f.close()
    
    return arepo_grid
    
def read_sink_evolution_file(filename):
    f = open(filename, mode='rb')
    #print ("Loading file %s" %filename)

    time = np.fromfile(f, float64, 1)

    #print ("Time = ", time)

    NSinks = np.fromfile(f, uint32, 1)

    #print ("Number of sink particles = ", NSinks)

    SinkP = {'Pos':[],
             'Vel':[],
             'Acc':[],
             'Mass':[],
             'MassOld':[],
             'FormationMass':[],
             'FormationTime':[],
             'ID':[],
             'HomeTask':[],
             'Index':[],
             'FormationOrder':[]}

    for i in np.arange(NSinks):

      app = SinkP['Pos']
      app.append(np.fromfile(f, float64, 3).reshape(1,3))
      SinkP['Pos'] = app

      app = SinkP['Vel']
      app.append(np.fromfile(f, float64, 3).reshape(1,3))
      SinkP['Vel'] = app

      app = SinkP['Acc']
      app.append(np.fromfile(f, float64, 3).reshape(1,3))
      SinkP['Acc'] = app

      app = SinkP['Mass']
      app.append(np.fromfile(f, float64, 1))
      SinkP['Mass'] = app
        
      app = SinkP['MassOld']
      app.append(np.fromfile(f, float64, 1))
      SinkP['MassOld'] = app

      app = SinkP['FormationMass']
      app.append(np.fromfile(f, float64, 1))
      SinkP['FormationMass'] = app

      app = SinkP['FormationTime']
      app.append(np.fromfile(f, float64, 1))
      SinkP['FormationTime'] = app

      app = SinkP['ID']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['ID'] = app

      app = SinkP['HomeTask']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['HomeTask'] = app

      app = SinkP['Index']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['Index'] = app

      app = SinkP['FormationOrder']
      app.append(np.fromfile(f, uint32, 1))
      SinkP['FormationOrder'] = app

    return time, SinkP

def return_gas_particles(data, header):
    ngas = header['num_particles'][0]
    
    data['pos'] = data['pos'][0:ngas,:]
    data['vel'] = data['vel'][0:ngas,:]
    data['id'] = data['id'][0:ngas]
    data['mass'] = data['mass'][0:ngas]

    return data

########## necessary WRITE functions #########

def writeu(f, data):
    """ Write a numpy array to the unformatted fortran file f """
    data_size = data.size * data.dtype.itemsize

    data_size = np.array(data_size, dtype=uint32)

    data_size.tofile(f)

    data.tofile(f)

    data_size.tofile(f)
    
####### binary write functions #########

def write_snapshot(filename, header, data, io_flags):
    """ Write a binary snapshot file (unformatted fortran) """
    # do some checks on the header
    for name, size in zip(header_names, header_sizes):
      if name not in header:
        #print('Missing %s in header file' % name)
        raise Exception('Missing %s in header file' % name)

      if np.array(header[name]).size != size[1]:
        msg = 'Header %s should contain %d elements, %d found' % \
              (name, size[1], np.array(header[name]).size)
        #print(msg)
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