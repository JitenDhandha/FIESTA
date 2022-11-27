######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains the internal functions that can be used across 
the package.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import numpy as np
#Astropy
from astropy import units as u

######################################################################
#                               FUNCTIONS                            #
######################################################################

def check_unit(unit, expected_unit):
    try:
        unit = u.Unit(unit)
    except ValueError:
        raise u.UnitsError(_prestring() + "Got incompatible/unexpected units.")
    if not unit.is_equivalent(expected_unit):
        raise u.UnitsError(_prestring() + "Got incompatible/unexpected units.")

def check_quantity(obj, expected_unit, string="Object"):
    if not isinstance(obj, u.Quantity):
        raise TypeError(_prestring() + string + " must be a Quantity!")
    if not obj.unit.is_equivalent(expected_unit):
        raise u.UnitsError(_prestring() + string + " has incompatible/unexpected units.")

def _prestring():
    return "FIESTA >> "