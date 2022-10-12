######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains the AREPO base units that are used throughout 
the code for unit conversions, and also some additional fundamental
constants. The module allows for changing the AREPO base units,
which is then propagated elsewhere in the code.
"""

######################################################################
#                            AREPO units                             #
######################################################################

#Base units (in CGS)
ulength = 1.0000000e+17 #: AREPO *length* unit in cm (base).
umass = 1.991e33 		#: AREPO *mass* unit in g (base).
uvel = 36447.268 		#: AREPO *velocity* unit in g/s (base).

#Derived units
utime = ulength/uvel                     #: AREPO *time* unit in s (derived).
udensity = umass/ulength/ulength/ulength #: AREPO *density* unit in g/cm^3 (derived).
uenergy= umass*uvel*uvel                 #: AREPO *energy* unit in ergs (derived).
ucolumn = umass/ulength/ulength          #: AREPO *column density* unit in g/cm^2 (derived).
uparsec=ulength/3.0856e18                #: Conversion from ``ulength`` to parsec.
usolarmass = umass/1.988e+33             #: Conversion from ``umass`` to solar masses.

xHe = 0.1 #: Helium fractional abundance.

######################################################################
#                      Fundamental constants                         #
######################################################################

mp = 1.6726231e-24 #: Mass of proton in g.
kb = 1.3806485e-16 #: Boltzmann consant in erg/K.
c = 29979245800 #: Speed of light in cm/s

######################################################################
#               Functions to change the global units                 #
######################################################################

def _reset_derived_units():
    global ulength, umass, uvel
    global utime, udensity, uenergy, ucolumn, uparsec, usolarmass
    utime = ulength/uvel
    udensity = umass/ulength/ulength/ulength
    uenergy = umass*uvel*uvel
    ucolumn = umass/ulength/ulength
    uparsec = ulength/3.0856e18
    usolarmass = umass/1.988e+33

def set_ulength(length_new):
    """
    Function to set the value of AREPO's ``ulength``.
    
    Parameters
    ----------
    
    length_new : float
        The value to set ``ulength`` to.
    
    """
    global ulength
    ulength = length_new
    _reset_derived_units()

def set_umass(mass_new):
    """
    Function to set the value of AREPO's ``umass``.
    
    Parameters
    ----------
    
    mass_new : float
        The value to set ``umass`` to.
    
    """
    global umass
    umass = mass_new
    _reset_derived_units()

def set_uvel(vel_new):
    """
    Function to set the value of AREPO's ``uvel``.
    
    Parameters
    ----------
    
    vel_new : float
        The value to set ``uvel`` to.
    
    """
    global uvel
    uvel = vel_new
    _reset_derived_units()

def set_xHe(xHe_new):
    """
    Function to set the Helium fractional abundance.
    
    Parameters
    ----------
    
    xHe_new : float
        The value to set ``xHe`` to.
    
    """
    global xHe
    xHe = xHe_new