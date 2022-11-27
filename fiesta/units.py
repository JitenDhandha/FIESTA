######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains the AREPO units used throughout 
the code for unit conversions. 

The three base `~astropy.units.Unit`'s defined (in CGS) are:

- `~fiesta.units.AREPO_LENGTH`
- `~fiesta.units.AREPO_MASS`
- `~fiesta.units.AREPO_VELOCITY`

along with the following derived `~astropy.units.Unit`'s, for ease of use:

- `~fiesta.units.AREPO_TIME`
- `~fiesta.units.AREPO_DENSITY`
- `~fiesta.units.AREPO_ENERGY`

It additionally contains `~fiesta.units.AREPO_xHe`, the helium fractional
abundance assumed in AREPO. 

The module allows for changing these variables as required, 
which is then propagated elsewhere in the code. 
For each of the units, it is possible to see its value through 
the ``.represents`` attribute of `~astropy.units.Unit` class. 
For example, ``AREPO_LENGTH.represents``.
"""

#Standard libs
#Astropy
from astropy import units as u
from astropy import constants as const

#Fiesta
from fiesta import _utils as utils

######################################################################
#                            AREPO units                             #
######################################################################

#Base units (in CGS)
AREPO_LENGTH = u.def_unit('AREPO_LENGTH', 1.0000000e+17 * u.cm  ) #: Base AREPO `~astropy.units.Unit` for *length*.
AREPO_MASS = u.def_unit('AREPO_MASS', 1.991e33 * u.g) #: Base AREPO `~astropy.units.Unit` for *mass*.
AREPO_VELOCITY = u.def_unit('AREPO_VELOCITY', 36447.268 * u.cm / u.s) #: Base AREPO `~astropy.units.Unit` for *velocity*.

#Derived units
AREPO_TIME = u.def_unit('AREPO_TIME', AREPO_LENGTH/AREPO_VELOCITY)    #: Derived AREPO `~astropy.units.Unit` for *time*.
AREPO_DENSITY = u.def_unit('AREPO_DENSITY', AREPO_MASS/AREPO_LENGTH/AREPO_LENGTH/AREPO_LENGTH) #: Derived AREPO `~astropy.units.Unit` for *density*.
AREPO_ENERGY = u.def_unit('AREPO_ENERGY', AREPO_MASS*AREPO_VELOCITY*AREPO_VELOCITY)  #: Derived AREPO `~astropy.units.Unit` for *energy*.    

AREPO_xHe = 0.1 << u.dimensionless_unscaled #: Helium fractional abundance assumed in AREPO.

#For the units, use .represents attribute of `~astropy.units.Unit` class to see its
#current value. For example, "AREPO_LENGTH.represents"

######################################################################
#               Functions to change the global units                 #
######################################################################

def _reset_derived_units():
    global AREPO_LENGTH, AREPO_MASS, AREPO_VELOCITY
    global AREPO_TIME, AREPO_DENSITY, AREPO_ENERGY

    AREPO_TIME = u.def_unit('AREPO_TIME', AREPO_LENGTH/AREPO_VELOCITY)
    AREPO_DENSITY = u.def_unit('AREPO_DENSITY', AREPO_MASS/AREPO_LENGTH/AREPO_LENGTH/AREPO_LENGTH)
    AREPO_ENERGY = u.def_unit('AREPO_ENERGY', AREPO_MASS*AREPO_VELOCITY*AREPO_VELOCITY)

def set_AREPO_LENGTH(length):
    """
    Function to set the value of base unit `~fiesta.units.AREPO_LENGTH`.
    
    Parameters
    ----------
    
    length : `~astropy.units.Quantity`
        New value for `~fiesta.units.AREPO_LENGTH`.
    
    """
    global AREPO_LENGTH
    utils.check_quantity(length,u.cm)
    AREPO_LENGTH = u.def_unit('AREPO_LENGTH', length)
    _reset_derived_units()

def set_AREPO_MASS(mass):
    """
    Function to set the value of base unit `~fiesta.units.AREPO_MASS`.
    
    Parameters
    ----------
    
    mass : `~astropy.units.Quantity`
        New value for `~fiesta.units.AREPO_MASS`.
    
    """
    global AREPO_MASS
    utils.check_quantity(mass,u.g)
    AREPO_MASS = u.def_unit('AREPO_MASS', mass)
    _reset_derived_units()

def set_AREPO_VELOCITY(vel):
    """
    Function to set the value of base unit `~fiesta.units.AREPO_VELOCITY`.
    
    Parameters
    ----------
    
    vel : `~astropy.units.Quantity`
        New value for `~fiesta.units.AREPO_VELOCITY`.
    
    """
    global AREPO_VELOCITY
    utils.check_quantity(vel,u.cm/u.s)
    AREPO_VELOCITY = u.def_unit('AREPO_VELOCITY', vel)
    _reset_derived_units()

def set_AREPO_xHe(xHe):
    """
    Function to set the Helium fractional abundance 
    `~fiesta.units.AREPO_xHe`.
    
    Parameters
    ----------
    
    xHe : `~astropy.units.Quantity`
        New value for `~fiesta.units.AREPO_xHe`.
    
    """
    global AREPO_xHe
    utils.check_quantity(xHe,u.dimensionless_unscaled)
    AREPO_xHe = xHe << u.dimensionless_unscaled

######################################################################
#                       [FUTURE FUNCTIONALITY]                       #
######################################################################
'''
class GridScale:
    def __init__(self, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
        self.nx = nx
        self.ny = ny 
        self.nz = nz
        self.xmin = xmin 
        self.ymin = ymin 
        self.zmin = zmin
        self.xmax = xmax 
        self.ymax = ymax
        self.zmax = zmax
        self.xequiv = [(u.pix, 
                        AREPO_LENGTH,
                        lambda val: self.xmin + (self.xmax-self.xmin)/self.nx * val, 
                        lambda val: (val - self.xmin) * self.nx/(self.xmax-self.xmin))]
        self.yequiv = [(u.pix, 
                        AREPO_LENGTH,
                        lambda val: self.ymin + (self.ymax-self.ymin)/self.nx * val, 
                        lambda val: (val - self.ymin) * self.nx/(self.ymax-self.ymin))]
        self.zequiv = [(u.pix, 
                        AREPO_LENGTH,
                        lambda val: self.zmin + (self.zmax-self.zmin)/self.nx * val, 
                        lambda val: (val - self.zmin) * self.nx/(self.zmax-self.zmin))]
'''