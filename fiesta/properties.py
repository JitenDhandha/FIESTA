######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains functions for calculating physical properties of
AREPO simulations such as number density, sound speed, Jeans mass,
Mach number, and more.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
import numpy as np

#FIESTA
from fiesta import units as ufi

######################################################################
#                              FUNCTIONS                             #
######################################################################


def calc_jeans_mass(cs, ndens):

    """
    
    Calculates Jeans mass using
    :math:`M_\\mathrm{J} = 2 \\big(\\frac{c_s}{0.2~\\mathrm{km/s}}\\big)^3 \\big(\\frac{n}{1000~\\mathrm{cm}^{-3}}\\big)^{-0.5} ~\\mathrm{M}_\\odot`.

    Parameters
    ----------

    cs : numpy.ndarray
        Array of sound speeds :math:`c_s` in km/s.

    ndens : numpy.ndarray
        Array of number densities :math:`n` in 1/cm^3.

    Returns
    -------

    jeans_mass : numpy.ndarray
        1D array of Jeans masses :math:`M_\\mathrm{J}` in solar masses.

    """

    jeans_mass = 2.0 * (cs / 0.2)**3.0 * (ndens / 1000.0)**-0.5
    return jeans_mass


def calc_free_fall_time(ndens):
    """
    
    Calculates free-fall time using
    :math:`t_\\mathrm{ff} = 2 \\times 10^6 \\big(\\frac{n}{1000~\\mathrm{cm}^{-3}}\\big)^{-0.5} ~\\mathrm{years}`.

    Parameters
    ----------

    ndens : numpy.ndarray
        Array of number densities :math:`n` in 1/cm^3.

    Returns
    -------

    free_fall_time : numpy.ndarray
        Array of free fall times :math:`t_\\mathrm{ff}` in years.

    """
    free_fall_time = 2.0e6 * (ndens/1000.0)**-0.5
    return free_fall_time


def calc_jeans_length(cs, ndens):
    """
    
    Calculates Jeans length using
    :math:`\\lambda_\\mathrm{J} = 0.4 \\big(\\frac{c_s}{0.2~\\mathrm{km/s}}\\big) \\big(\\frac{n}{1000~\\mathrm{cm}^{-3}}\\big)^{-0.5} ~\\mathrm{pc}`.

    Parameters
    ----------

    cs : numpy.ndarray
        Array of sound speeds :math:`c_s` in km/s.

    ndens : numpy.ndarray
        Array of number densities :math:`n` in 1/cm^3.

    Returns
    -------

    jeans_length : numpy.ndarray
        Array of Jeans lengths :math:`\\lambda_\\mathrm{J}` in parsec.

    """
    jeans_length = 0.4 * (cs/0.2) * (ndens/1000.0)**-0.5
    return jeans_length


def calc_molecular_weight(chem):
    """
    
    Calculates molecular weight using
    :math:`\\mathrm{molw} = \\frac{1 - 4 x_\\mathrm{He}}{1+x_\\mathrm{He}-x_\\mathrm{H_2}+x_\\mathrm{H^+}}`.

    Parameters
    ----------

    chem : numpy.ndarray
        See variable ``chem`` in :attr:`~fiesta.arepo.ArepoVoronoiGrid`.

    Returns
    -------

    molw : numpy.ndarray
        Array of molecular weights :math:`\\mathrm{molw}` (unitless).

    """
    xH2, xHp, _ = chem.T
    molw = (1.0 + 4.0*ufi.xHe) / (1.0 + xHe - xH2 + xHp)
    return molw


def calc_sound_speed(temp, molw):
    """
    
    Calculates sound speed using 
    :math:`c_\\mathrm{s} = 10^{-5} \\sqrt{\\frac{k_\\mathrm{b}T}{\\mathrm{molw}\\times m_\\mathrm{p}}} ~\\mathrm{km/s}`.

    Parameters
    ----------

    temp : numpy.ndarray
        Array of temperatures :math:`T` in Kelvin.

    molw : numpy.ndarray
        Array of molecular weights :math:`\\mathrm{molw}` (unitless).

    Returns
    -------

    sounds_speed : numpy.ndarray
        Array of sound speeds :math:`c_\\mathrm{s}` in km/s.

    """
    sounds_speed = 1.0e-5 * np.sqrt(ufi.kb*temp/(molw*ufi.mp))
    return sounds_speed


def calc_number_density(rho):
    """
    
    Calculates number density using 
    :math:`n = \\frac{\\rho}{(1+4x_\\mathrm{He})m_\\mathrm{p}}`
    in 1/cm^3.

    Parameters
    ----------

    rho : numpy.ndarray
        Array of density :math:`\\rho` in AREPO ``umass/ulength**3`` units.

    Returns
    -------

    ndensity : numpy.ndarray
        Array of number densities :math:`n` in 1/cm^3.

    """
    ndensity = rho * ufi.udensity/((1.0 + 4.0 * ufi.xHe) * ufi.mp)
    return ndensity


#A note on fractional abundances:
#The x's here are with respect to total number of hydrogen atoms (nHtot).
#nHtot = nHI + nHp + 2*nH2
#nTOT = nHI + nH2 + nHp + ne + nHe

def calc_chem_density(chem, ndens):
    """
    
    Calculates chemical densities 
    :math:`n_\\mathrm{HI}`, :math:`n_\\mathrm{H^+}`, :math:`n_\\mathrm{H_2}`, 
    :math:`n_\\mathrm{CO}`, :math:`n_\\mathrm{tot}` 
    in 1/cm^3.

    Parameters
    ----------

    chem : numpy.ndarray
        See variable ``chem`` in :attr:`~fiesta.arepo.ArepoVoronoiGrid`.

    ndens : numpy.ndarray
        Array of number densities :math:`\\mathrm{n}` in 1/cm^3.

    Returns
    -------

    nHI : numpy.ndarray
        Array of neutral hydrogen number densities :math:`n_\\mathrm{HI}` in 1/cm^3.

    nHp : numpy.ndarray
        Array of ionized hydrogen number densities :math:`n_\\mathrm{H^+}` in 1/cm^3.

    nH2 : numpy.ndarray
        Array of molecular hydrogen number densities :math:`n_\\mathrm{H_2}` in 1/cm^3.

    nCO : numpy.ndarray
        Array of CO number densities :math:`n_\\mathrm{CO}` in 1/cm^3.

    nTOT : numpy.ndarray
        Array of total number densities :math:`n_\\mathrm{tot}` in 1/cm^3.

    """
    xH2, xHp, xCO = chem.T
    xHI = 1.0 - xHp - 2.0*xH2
    xTOT = 1.0 + xHp - xH2 + xHe
    nHp = xHp*ndens
    nH2 = xH2*ndens
    nCO = xCO*ndens
    nHI = xHI*ndens
    nTOT = xTOT*ndens
    return [nHI, nHp, nH2, nCO, nTOT]


def calc_temperature(rho, utherm, nTOT):
    """
    
    Calculates temperature using
    :math:`T = \\frac{2}{3}\\frac{u_\\mathrm{therm} m_\\mathrm{avg} m_\\mathrm{p}}{k_\\mathrm{b}}`
    in Kelvin.

    Parameters
    ----------

    rho : numpy.ndarray
        Array of density :math:`\\rho` in AREPO ``umass/ulength**3`` units.

    utherm : numpy.ndarray
        Array of energy per unit mass in AREPO ``1/ulength**3`` units.

    nTOT : numpy.ndarray
        Array of total number densities :math:`n_\\mathrm{tot}` in 1/cm^3.

    Returns
    -------

    temperature : numpy.ndarray
        Array of temperatures :math:`T` in Kelvin.

    """
    rho_cgs = rho * ufi.udensity
    energy_per_unit_mass_cgs = ufi.utherm * ufi.uenergy/ufi.umass
    avg_mass = rho_cgs/(nTOT*ufi.mp)
    temperature = (2.0/3.0)*energy_per_unit_mass_cgs*avg_mass*ufi.mp/ufi.kb
    return temperature


def calc_kinetic_energy(mass, vel):
    """
    
    Calculates kinetic energy using
    :math:`E_\\mathrm{kin} = \\frac{1}{2}mv^2`
    in ergs.

    Parameters
    ----------

    mass : numpy.ndarray
        Array of masses :math:`m` in AREPO ``umass`` units.

    vel : numpy.ndarray
        Array of velocities :math:`v` in AREPO ``uvel`` units.

    Returns
    -------

    Ekin : numpy.ndarray
        Array of kinetic energies :math:`E_\\mathrm{kin}` in ergs.

    """
    Ekin = np.sum(ufi.mass*(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)*ufi.uenergy)
    return Ekin


def calc_gravitational_potential_energy(mass, rho):
    """
    
    Calculates gravitational potential energy using
    :math:`E_\\mathrm{grav} = \\frac{3}{5}\\frac{GM^2}{R}`
    in ergs.

    Parameters
    ----------

    mass : numpy.ndarray
        Array of masses :math:`m` in AREPO ``umass`` units.

    rho : numpy.ndarray
        Array of density :math:`\\rho` in AREPO ``umass/ulength**3`` units.

    Returns
    -------

    Egrav : numpy.ndarray
        Array of gravitational potential energies :math:`E_\\mathrm{grav}` in ergs.

    """
    G = 1.0 #TO DO: This is bad.
    M = sum(mass)
    rho_avg = np.average(rho) #assuming ~constant density cloud
    R = (3.0*M / (4.0*np.pi*rho_avg))**(1.0/3.0) #assuming spherical
    Egrav = 3.0/5.0*G*M**2.0/R * ufi.uenergy 
    return Egrav


def calc_thermal_energy(utherm):
    """
    
    Calculates thermal energy using :math:`E_\\mathrm{therm} = \\sum u_\\mathrm{therm}` 
    in ergs.

    Parameters
    ----------

    utherm : numpy.ndarray
        Array of energy per unit mass in AREPO ``1/ulength**3`` units.

    Returns
    -------

    Etherm : numpy.ndarray
        Array of thermal energies :math:`E_\\mathrm{therm}` in ergs.

    """
    Etherm = np.sum(utherm)*ufi.uenergy
    return Etherm


def calc_mach_numbers(vel, cs):
    """
    
    Calculates Mach numbers using :math:`\\mathcal{M} = \\frac{\\sigma_v}{c_\\mathrm{s}}`.

    Parameters
    ----------

    vel : numpy.ndarray
        Array of velocities :math:`v` in AREPO ``uvel`` units.

    cs : numpy.ndarray
        Array of sound speeds :math:`c_s` in km/s.

    Returns
    -------

    mach : numpy.ndarray
        Array of Mach numbers :math:`\\mathcal{M}` (unitless).

    """
    vel_disp_x = vel[:,0].std()*ufi.uvel/10.0**5 #km/s
    vel_disp_y = vel[:,1].std()*ufi.uvel/10.0**5 #km/s
    vel_disp_z = vel[:,2].std()*ufi.uvel/10.0**5 #km/s
    vel_disp = np.sqrt(vel_disp_x**2.0 + vel_disp_y**2.0 + vel_disp_z**2.0)
    mach = vel_disp / cs
    return mach