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

#Own libs
from .units import *

######################################################################
#                         PHYSICAL FUNCTIONS                         #
######################################################################

xHe = 0.1 #Fraction of helium. Assumed throughout.

#Input: cs in km/s and ndens in cm^-3
#Output: Jmass in solar masses
def calc_jeans_mass(cs, ndens):
    Jmass = 2.0 * (cs / 0.2)**3.0 * (ndens / 1000.0)**-0.5
    return Jmass

#Input: ndens in cm^-3
#Output: fft in years
def calc_free_fall_time(ndens):
    fft = 2.0e6 * (ndens/1000.0)**-0.5
    return fft

#Input: cs in km/s and ndens in cm^-3
#Output: Jlength in parsec
def calc_jeans_length(cs, ndens):
    Jlength = 0.4 * (cs/0.2) * (ndens/1000.0)**-0.5
    return Jlength

#Input: xH2 and xHp in unitless quantities
#Output: molw in unitless quantity
def calc_molecular_weight(chem):
    xH2, xHp, _ = chem.T
    #Molecular weight given molecular and ionised hydrogen abundance
    molw = (1.0 + 4.0*xHe) / (1.0 + xHe - xH2 + xHp)
    return molw

#Input: temp in K and molw in unitless quantity
#Output: cs in km/s
def calc_sound_speed(temp, molw):
    cs = 1.0e-5 * np.sqrt(kb*temp/(molw*mp))
    return cs

#Fractional quantity stuff below:
#The x's here are with respect to total number of hydrogen atoms (nHtot).
#nHtot = nHI + nHp + 2*nH2
#nTOT = nHI + nH2 + nHp + ne + nHe

#Input: rho in codeunits
#Output: ndens in cm^-3
def calc_number_density(rho):
    rho_cgs = rho * udensity
    ndens = rho_cgs/((1.0 + 4.0 * xHe) * mp)
    return ndens

#Input: chem in codeunits, ndens in cm^-3
#Output: nHI, nHp, nH2, nCO, nTOT in cm^-3
def calc_chem_density(chem, ndens):
    xH2, xHp, xCO = chem.T
    xHI = 1.0 - xHp - 2.0*xH2
    xTOT = 1.0 + xHp - xH2 + xHe
    nHp = xHp*ndens
    nH2 = xH2*ndens
    nCO = xCO*ndens
    nHI = xHI*ndens
    nTOT = xTOT*ndens
    return [nHI, nHp, nH2, nCO, nTOT]
    
#Input: utherm in codeunits, rho in codeunits, nTOT in cm^-3
#Output: temp in Kelvin
def calc_temperature(rho, utherm, nTOT):
    rho_cgs = rho * udensity
    energy_per_unit_mass_cgs = utherm * uenergy/umass
    avg_mass = rho_cgs/(nTOT*mp)
    temp = (2.0/3.0)*energy_per_unit_mass_cgs*avg_mass*mp/kb
    return temp

#Input: mass in codeunits, vel in codeunits
#Output: E in ergs
def calc_kinetic_energy(mass, vel):
    Ekin = 0.0
    for i in range(len(mass)):
        Ei = mass[i]/2.0 * (vel[i][0]**2.0 + vel[i][1]**2.0 + vel[i][2]**2.0)
        Ekin += Ei
    Ekin *= uenergy #converting to cgs here
    return Ekin

#Input: mass in codeunits, rho in codeunits
#Output: E in ergs
def calc_gravitational_potential_energy(mass, rho):
    G = 1.0 #code units
    M = sum(mass)
    #radius assuming spherical, constant density cloud
    rho_avg = np.average(rho)
    R = (3.0*M / (4.0*np.pi*rho_avg))**(1.0/3.0)
    #gravitational potential energy with same assumption
    Egrav = 3.0/5.0*G*M**2.0/R * uenergy #converting to cgs here
    return Egrav

#Input: mass in codeunits
#Output: M in g
def calc_total_mass(mass):
    M = np.sum(mass)
    M *= umass #converting to cgs here
    return M

#Input: utherm in codeunits
#Output: E in ergs
def calc_thermal_energy(utherm):
    E = np.sum(utherm)
    E *= uenergy #converting to cgs here
    return E

#Input: vel in codeunits, cs in km/s 
#Output: mach_number in unitless quantity
def calc_mach_numbers(vel, cs):
    vel_disp_x = vel[:,0].std()*uvel/10.0**5 #km/s
    vel_disp_y = vel[:,1].std()*uvel/10.0**5 #km/s
    vel_disp_z = vel[:,2].std()*uvel/10.0**5 #km/s
    vel_disp = np.sqrt(vel_disp_x**2.0 + vel_disp_y**2.0 + vel_disp_z**2.0)
    mach = vel_disp / cs
    return mach