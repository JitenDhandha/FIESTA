######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

"""
This module contains functions for calculating physical properties 
such as number density, sound speed, Jeans mass, Mach number, and more.
"""

######################################################################
#                            LIBRARIES                               #
######################################################################

#Standard libs
#Numpy
import numpy as np
#Astropy
from astropy import units as u
from astropy import constants as const

#FIESTA
from fiesta import units as ufi
from fiesta import _utils as utils

######################################################################
#                              FUNCTIONS                             #
######################################################################


def calc_jeans_mass(cs, ndens, unit=u.g):

    """
    
    Calculates Jeans mass using
    :math:`M_\\mathrm{J} = 2 \\big(\\frac{c_s}{2~\\mathrm{km/s}}\\big)^3 \\big(\\frac{n}{1000~\\mathrm{cm}^{-3}}\\big)^{-0.5} ~\\mathrm{M}_\\odot`.

    Parameters
    ----------

    cs : `~astropy.units.Quantity`
        Sound speed :math:`c_s`.

    ndens : `~astropy.units.Quantity`
        Number density :math:`n`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.g``.

    Returns
    -------

    jeans_mass : `~astropy.units.Quantity`
       Jeans mass :math:`M_\\mathrm{J}`.

    """
    utils.check_quantity(cs, u.cm/u.s, "cs")
    utils.check_quantity(ndens, u.cm**-3, "ndens")
    cs = cs.to(u.km/u.s)
    ndens = ndens.to(u.cm**-3)
    jeans_mass = 2.0 * np.power(cs/(0.2 * u.km/u.s),3) * np.power(ndens/(1000.0 * u.cm**-3),-0.5) << u.Msun
    utils.check_unit(unit, u.g)
    return jeans_mass.to(unit)

def calc_free_fall_time(ndens, unit=u.s):
    """
    
    Calculates free-fall time using
    :math:`t_\\mathrm{ff} = 2 \\times 10^6 \\big(\\frac{n}{1000~\\mathrm{cm}^{-3}}\\big)^{-0.5} ~\\mathrm{yrs}`.

    Parameters
    ----------

    ndens : `~astropy.units.Quantity`
        Number density :math:`n`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.s``.

    Returns
    -------

    free_fall_time : `~astropy.units.Quantity`
        Free fall time :math:`t_\\mathrm{ff}`.

    """
    utils.check_quantity(ndens, u.cm**-3, "ndens")
    ndens = ndens.to(u.cm**-3)
    free_fall_time = 2.0e6 * np.power(ndens/(1000.0 * u.cm**-3),-0.5) << u.yr
    utils.check_unit(unit, u.s)
    return free_fall_time.to(unit)


def calc_jeans_length(cs, ndens, unit=u.cm):
    """
    
    Calculates Jeans length using
    :math:`\\lambda_\\mathrm{J} = 0.4 \\big(\\frac{c_s}{2~\\mathrm{km/s}}\\big) \\big(\\frac{n}{1000~\\mathrm{cm}^{-3}}\\big)^{-0.5} ~\\mathrm{pc}`.

    Parameters
    ----------

    cs : `~astropy.units.Quantity`
        Sound speed :math:`c_s`.

    ndens : `~astropy.units.Quantity`
        Number density :math:`n`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.cm``.

    Returns
    -------

    jeans_length : `~astropy.units.Quantity`
        Jeans length :math:`\\lambda_\\mathrm{J}`.

    """
    utils.check_quantity(cs, u.cm/u.s, "cs")
    utils.check_quantity(ndens, u.cm**-3, "ndens")
    cs = cs.to(u.km/u.s)
    ndens = ndens.to(u.cm**-3)
    jeans_length = 0.4 * (cs / (0.2 * u.km/u.s)) * np.power(ndens/(1000.0 * u.cm**-3),-0.5) << u.pc
    utils.check_unit(unit, u.cm)
    return jeans_length.to(unit)


def calc_molecular_weight(chem):
    """
    
    Calculates molecular weight using
    :math:`\\mathrm{molw} = \\frac{1 - 4 x_\\mathrm{He}}{1+x_\\mathrm{He}-x_\\mathrm{H_2}+x_\\mathrm{H^+}}`.

    Parameters
    ----------

    chem : `~astropy.units.Quantity`
        See variable `~fiesta.arepo.ArepoVoronoiGrid.chem` in `~fiesta.arepo.ArepoVoronoiGrid`.

    Returns
    -------

    molw : `~astropy.units.Quantity`
        Molecular weight :math:`\\mathrm{molw}` (unitless).

    """
    utils.check_quantity(chem, u.dimensionless_unscaled, "chem")
    xH2, xHp, _ = chem.T.to(u.dimensionless_unscaled)
    molw = (1.0 + 4.0*ufi.AREPO_xHe) / (1.0 + ufi.AREPO_xHe - xH2 + xHp) << u.dimensionless_unscaled
    return molw


def calc_sound_speed(temp, molw, unit=u.cm/u.s):
    """
    
    Calculates sound speed using 
    :math:`c_\\mathrm{s} = 10^{-5} \\sqrt{\\frac{k_\\mathrm{b}T}{\\mathrm{molw}\\times m_\\mathrm{p}}} ~\\mathrm{km/s}`.

    Parameters
    ----------

    temp : `~astropy.units.Quantity`
        Temperature :math:`T`.

    molw : `~astropy.units.Quantity`
        Molecular weight :math:`\\mathrm{molw}` (unitless).

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.cm/u.s``.

    Returns
    -------

    cs : `~astropy.units.Quantity`
        Sound speed :math:`c_\\mathrm{s}`.

    """
    utils.check_quantity(temp, u.K, "temp")
    utils.check_quantity(molw, u.dimensionless_unscaled, "molw")
    temp = temp.to(u.K)
    molw = molw.to(u.dimensionless_unscaled)
    cs = np.sqrt((const.k_B.cgs * temp) / (molw * const.m_p.cgs))
    utils.check_unit(unit, u.cm/u.s)
    return cs.to(unit)


def calc_number_density(rho, unit=u.cm**-3):
    """
    
    Calculates number density using 
    :math:`n = \\frac{\\rho}{(1+4x_\\mathrm{He})m_\\mathrm{p}}`.

    Parameters
    ----------

    rho : `~astropy.units.Quantity`
        Density :math:`\\rho`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.cm**-3``.

    Returns
    -------

    ndensity : `~astropy.units.Quantity`
        Number density :math:`n`.

    """
    utils.check_quantity(rho, u.g/u.cm**3, "rho")
    rho = rho.to(u.g/u.cm**3)
    ndensity = rho/((1.0 + 4.0 * ufi.AREPO_xHe) * const.m_p.cgs)
    utils.check_unit(unit, u.cm**-3)
    return ndensity.to(unit)


#A note on fractional abundances:
#The x's here are with respect to total number of hydrogen atoms (nHtot).
#nHtot = nHI + nHp + 2*nH2
#nTOT = nHI + nH2 + nHp + ne + nHe

def calc_chem_density(chem, ndens, unit=u.cm**-3):
    """
    
    Calculates chemical densities 
    :math:`n_\\mathrm{HI}`, :math:`n_\\mathrm{H^+}`, :math:`n_\\mathrm{H_2}`, 
    :math:`n_\\mathrm{CO}`, :math:`n_\\mathrm{tot}`.

    Parameters
    ----------

    chem : `~astropy.units.Quantity`
        See variable `~fiesta.arepo.ArepoVoronoiGrid.chem` in `~fiesta.arepo.ArepoVoronoiGrid`.

    ndens : `~astropy.units.Quantity`
        Number density :math:`n`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.cm**-3``.

    Returns
    -------

    nHI : `~astropy.units.Quantity`
        Neutral hydrogen number density :math:`n_\\mathrm{HI}`.

    nHp : `~astropy.units.Quantity`
       Ionized hydrogen number density :math:`n_\\mathrm{H^+}`.

    nH2 : `~astropy.units.Quantity`
        Molecular hydrogen number density :math:`n_\\mathrm{H_2}`.

    nCO : `~astropy.units.Quantity`
       CO number density :math:`n_\\mathrm{CO}`.

    nTOT : `~astropy.units.Quantity`
        Total number density :math:`n_\\mathrm{tot}`.

    """
    utils.check_quantity(chem, u.dimensionless_unscaled, "chem")
    xH2, xHp, xCO = chem.T.to(u.dimensionless_unscaled)
    ndens = ndens.to(u.cm**-3)
    xHI = 1.0 - xHp - 2.0*xH2
    xTOT = 1.0 + xHp - xH2 + ufi.AREPO_xHe
    nHp = xHp*ndens
    nH2 = xH2*ndens
    nCO = xCO*ndens
    nHI = xHI*ndens
    nTOT =xTOT*ndens
    utils.check_unit(unit, u.cm**-3)
    return nHI.to(unit), nHp.to(unit), nH2.to(unit), nCO.to(unit), nTOT.to(unit)

def calc_temperature(rho, utherm, nTOT, unit=u.K):
    """
    
    Calculates temperature using
    :math:`T = \\frac{2}{3}\\frac{u_\\mathrm{therm} m_\\mathrm{avg} m_\\mathrm{p}}{k_\\mathrm{b}}`.

    Parameters
    ----------

    rho :  `~astropy.units.Quantity`
        Density :math:`\\rho`.

    utherm :  `~astropy.units.Quantity`
        Thermal energy per unit mass :math:`u_\\mathrm{therm}`.

    nTOT : `~astropy.units.Quantity`
        Total number density :math:`n_\\mathrm{tot}`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.K``.

    Returns
    -------

    temperature :  `~astropy.units.Quantity`
        Temperature :math:`T`.

    """
    utils.check_quantity(rho, u.g/u.cm**3, "rho")
    utils.check_quantity(nTOT, u.cm**-3, "nTOT")
    utils.check_quantity(utherm, u.erg/u.g, "utherm")
    rho = rho.to(u.g/u.cm**3)
    nTOT = nTOT.to(u.cm**-3)
    utherm = utherm.to(u.erg/u.g)
    avg_mass = rho/nTOT
    temperature = (2.0/3.0) * utherm * avg_mass / const.k_B.cgs
    utils.check_unit(unit, u.K)
    return temperature.to(unit)


def calc_total_kinetic_energy(mass, vel, unit=u.erg):
    """
    
    Calculates kinetic energy using
    :math:`E_\\mathrm{kin} = \\sum_{i} \\frac{1}{2}m_i v_i^2`.

    Parameters
    ----------

    mass : `~astropy.units.Quantity`
        Mass :math:`m`.

    vel : `~astropy.units.Quantity`
        Velocity :math:`v`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.erg``.

    Returns
    -------

    Ekin : `~astropy.units.Quantity`
        Kinetic energy :math:`E_\\mathrm{kin}`.

    """
    utils.check_quantity(mass, u.g, "mass")
    utils.check_quantity(vel, u.cm/u.s, "vel")
    mass = mass.to(u.g)
    vel = vel.to(u.cm/u.s)
    Ekin = np.sum(1.0/2.0 * mass * (vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2))
    utils.check_unit(unit, u.erg)
    return Ekin.to(unit)


def calc_total_gravitational_potential_energy(mass, rho, unit=u.erg):
    """
    
    Calculates gravitational potential energy using
    :math:`E_\\mathrm{grav} = \\frac{3}{5}\\frac{GM^2}{R}`.

    Parameters
    ----------

    mass : `~astropy.units.Quantity`
        Mass :math:`m`.

    rho : `~astropy.units.Quantity`
        Density :math:`\\rho`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.erg``.

    Returns
    -------

    Egrav : numpy.ndarray
        Gravitational potential energy :math:`E_\\mathrm{grav}`.

    """
    utils.check_quantity(mass, u.g, "mass")
    utils.check_quantity(rho, u.g/u.cm**3, "rho")
    mass = mass.to(u.g)
    rho = rho.to(u.g/u.cm**3)
    total_mass = np.sum(mass)
    rho_avg = np.average(rho) #assuming roughly constant density
    radius = np.power((3.0*total_mass)/(4.0*np.pi*rho_avg),1.0/3.0) #assuming spherical
    Egrav = 3.0/5.0 * const.G.cgs * np.power(total_mass,2.0) / radius
    utils.check_unit(unit, u.erg)
    return Egrav.to(unit)

def calc_total_thermal_energy(utherm, mass, unit=u.erg):
    """
    
    Calculates thermal energy using :math:`E_\\mathrm{therm} = \\sum_{i} u_\\mathrm{i,therm} m_i`.

    Parameters
    ----------

    utherm :  `~astropy.units.Quantity`
        Thermal energy per unit mass :math:`u_\\mathrm{therm}`.

    mass : `~astropy.units.Quantity`
        Mass :math:`m`.

    unit : `~astropy.units.Unit`, optional
        Unit to output the result in. Default value is ``u.erg``.

    Returns
    -------

    Etherm : `~astropy.units.Quantity`
        Thermal energy :math:`E_\\mathrm{therm}`.

    """
    utils.check_quantity(utherm, u.erg/u.g, "utherm")
    utils.check_quantity(mass, u.g, "mass")
    utherm = utherm.to(u.erg/u.g)
    mass = mass.to(u.g)
    Etherm = np.sum(utherm*mass)
    utils.check_unit(unit, u.erg)
    return Etherm.to(unit)


def calc_mach_numbers(vel, cs):
    """
    
    Calculates Mach numbers using :math:`\\mathcal{M} = \\frac{\\sigma_v}{c_\\mathrm{s}}`.

    Parameters
    ----------

    vel : `~astropy.units.Quantity`
       Velocity :math:`v`.

    cs : `~astropy.units.Quantity`
        Sounds peed :math:`c_s`.

    Returns
    -------

    mach : `~astropy.units.Quantity`
        Mach number :math:`\\mathcal{M}` (unitless).

    """
    utils.check_quantity(vel, u.cm/u.s, "vel")
    utils.check_quantity(cs, u.cm/u.s, "cs")
    vel = vel.to(u.km/u.s)
    cs = cs.to(u.km/u.s)
    vel_disp_x = vel[:,0].std()
    vel_disp_y = vel[:,1].std()
    vel_disp_z = vel[:,2].std()
    vel_disp = np.sqrt(vel_disp_x**2.0 + vel_disp_y**2.0 + vel_disp_z**2.0)
    mach = vel_disp / cs
    return mach